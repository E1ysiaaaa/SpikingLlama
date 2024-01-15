"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from flash_attn import flash_attn_func
from src.config import Config
KVCache = Tuple[torch.Tensor, torch.Tensor]
#FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")

import math
from typing import Optional, Tuple
from einops import repeat

T_total = 63
time_step = T_total
wait_time = T_total

class IF(nn.Module):
    SOP = 0
    def __init__(self, T=T_total, step=time_step):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(8.0))
        self.T = T
        self.step = step

    def forward(self, x):
        # x [T, B, S, D]
        device = torch.cuda.current_device()
        threshold = self.alpha
        membrane = 0.5 * threshold
        spikes = torch.zeros(x.shape).to(device)

        for i in range(0, self.T, self.step):
            for j in range(0, min(self.step, self.T - i)):
                membrane = membrane + x[i+j]
            for j in range(0, min(self.step, self.T - i)):
                spike = membrane > threshold
                membrane[spike] = membrane[spike] - threshold
                spikes[i+j] = spike.float()
        
        IF.SOP += spikes.sum().item()
        return threshold * spikes

class SpikeInnerProduct(nn.Module):
    SOP = 0
    def __init__(self, T=T_total, step=time_step):
        super().__init__()
        self.T = T
        self.step = step

    def forward(self, x, y):
        T, B, n_heads, S, head_dim = x.shape
        x_th = torch.max(x.flatten()).item().to(x.device)
        y_th = torch.max(y.flatten()).item().to(y.device)
        out_weight = torch.zeros([T, B, n_heads, S, S]).to(x.device)
        
        for i in range(0, self.T, self.step):
            x_add = 0
            y_add = 0
            for j in range(min(self.step, self.T - i)):
                x_add = x_add + x[i+j] / self.step
                y_add = y_add + y[i+j] / self.step
            weight = x_add @ y_add
            SOP += weight.sum().item() * (self.step * self.step / x_th / y_th)
            for j in range(min(self.step, self.T - i)):
                out_weight[i+j] = weight
        return out_weight

class SpikeGPT(nn.Module):
    def __init__(self, config: Config, T=T_toal) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        self.T = T

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.poe = nn.Parameter(torch.zeros([config.block_size, config.n_embd]))
        self.out_if = IF()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            # RWKV: set it to 1e-4
            # torch.nn.init.uniform_(module.weight,  -1e-4, 1e-4)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # GPT-NeoX       
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (name == "w3.weight" and isinstance(module, SwiGLU) or (name=="proj.weight" and isinstance(module, CausalSelfAttention))):  #if use xformer swiglu, fc2 layer will be renamed to w3
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd)  /  n_layer)
        

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-gpt/pull/83#issuecomment-1558150179
            self.mask_cache = None

    def forward(
        self, idx: torch.Tensor, max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        T, B, S = idx.size()
        use_kv_cache = input_pos is not None

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:  # not relevant otherwise
            assert (
                max_seq_length >= S
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert block_size >= S, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        if use_kv_cache:
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :, :max_seq_length]
        else:
            mask = None
        
        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = x + self.poe[:S, :].unsqueeze(0)
            
        if not use_kv_cache:
            for block in self.transformer.h:
                x, *_ = block(x, max_seq_length)
        else:
            self.kv_caches = self.kv_caches or self.build_kv_caches(x, max_seq_length)
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(x, max_seq_length, mask, input_pos, self.kv_caches[i])

        x = self.transformer.ln_f(x)
        x = self.out_if(x)
        x = x.mean(dim=0)

        x = self.lm_head(x)  # (b, t, vocab_size)
        return x

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(self, idx: torch.Tensor, max_seq_length: int, rope_cache_length: int) -> List[KVCache]:
        B = idx.size(1)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_query_groups

        k_cache_shape = (self.T, B, max_seq_length, heads, self.config.head_size)
        v_cache_shape = (self.T, B, max_seq_length, heads, self.config.head_size)
        device = idx.device
        return [
            (torch.zeros(k_cache_shape, device=device), torch.zeros(v_cache_shape, device=device))
            for _ in range(self.config.n_layer)
        ]

class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.act_1 = IF()
        self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
            self.act_2 = IF()
        self.mlp = config.mlp_class(config)
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:

        n_1 = self.norm_1(x)
        n_1 = self.act_1(n_1)
        h, new_kv_cache = self.attn(n_1, max_seq_length, mask, input_pos, kv_cache)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.act_2(self.norm_2(x))
            x = x + h + self.mlp(n_2)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )
            
            x = x + h
            x = x + self.mlp(self.act_2(self.norm_2(x)))
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        self.act_1 = IF()
        self.act_2 = IF()
        self.inner_product1 = SpikeInnerProduct()
        self.inner_product2 = SpikeInnerProduct()
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        T, B, S, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.act_1(self.attn(x))

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(T, B, S, self.config.n_query_groups, total_qkv, self.config.head_size) # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        # repeat k and v if necessary
        # Peiyuan: we do not need to do this as flash attention 2 already support GQA
        # if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
        #     # for MHA this is a no-op
        #     k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
        #     v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(T, B, S, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(T, B, S, -1, self.config.head_size)  
        v = v.reshape(T, B, S, -1, self.config.head_size)  

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)

            k = cache_k.index_copy_(2, input_pos, k)
            v = cache_v.index_copy_(2, input_pos, v)
            kv_cache = k, v

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(T, B, S, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y, kv_cache

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)
        
        '''
        if (
            #FlashAttention2Available
            mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=True)
        '''
        q = q.transpose(3, 2)
        k = k.transpose(3, 2)
        v = v.transpose(3, 2)
        if q.size() != k.size():
             k = k.repeat_interleave(q.shape[2]//k.shape[2], dim=2)
             v = v.repeat_interleave(q.shape[2]//v.shape[2], dim=2)

        L, S = q.size(-2), k.size(-2)
        scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
        if mask is None:
            temp_mask = torch.ones(L, S, dtype=torch.bool, device=q.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)
        else:
            if mask.dtype == torch.bool:
                mask.masked_fill_(mask.logical_not(), float("-inf"))
            else:
                attn_bias += mask
        attn_weight = self.inner_product1(q, k.transpose(-2, -1)) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        #attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        attn_weight = self.act_2(attn_weight)
        y = self.inner_product2(attn_weight, v)
        return y.transpose(1, 2)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.act = IF()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.act(x)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        # self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        # self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        # self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=False, _pack_weights=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_fc_1 = self.fc_1(x)
        # x_fc_2 = self.fc_2(x)
        # x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        # return self.proj(x)
        return self.swiglu(x)
