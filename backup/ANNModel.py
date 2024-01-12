import torch 
from torch import nn
import torch.nn.functional as F 
import math
from typing import Optional, Tuple

from .args import *

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        Q, K = apply_rotary_emb(Q, K, freq_cis)
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MySpikeGPT(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args 

        self.encode_layer = nn.Embedding(self.args.vocab_size, self.args.embed)
        self.transformer = [TransformerBlock(i, self.args) for i in range(self.args.n_layers)]
        self.out = OutputLayer(self.args)

        self.pos = nn.Parameter(torch.zeros([self.args.ctx_len, self.args.embed]))
        self.register_parameter('pos', self.pos)

        for i in range(self.args.n_layers):
            self.register_module('transformer_block'+str(i), self.transformer[i])
        self.register_module('out', self.out)

    def forward(self, x, y=None):
        # x: [B, S], out: [B, S, vocab]
        assert x.shape[1] == self.args.ctx_len, "input sequence length is not equal to ctx_len!"
        out = self.encode_layer(x)
        out = out + self.pos
        out = F.relu(out)
        for i in range(self.args.n_layers):
            out = self.transformer[i](out)
        out = self.out(out)

        if y is not None and y.shape[1] != 2:
            return F.cross_entropy(out.view(-1, self.args.vocab_size), y.view(-1))
        elif y is not None and y.shape[1] == 2:
            return out
        else:
            return out[:, -1, :]

class TransformerBlock(nn.Module):
    def __init__(self, i, model_args=args):
        super().__init__()
        self.args = model_args
        self.ffn = FFN(self.args)
        self.sdsa = SDSA(i, self.args)
        self.sdsa_norm = nn.LayerNorm(self.args.embed)
        self.ffn_norm = nn.LayerNorm(self.args.embed)

        self.register_module('ffn', self.ffn)
        self.register_module('sdsa', self.sdsa)
        #self.register_module('sdsa_norm', self.sdsa_norm)
        #self.register_module('ffn_norm', self.ffn_norm)

    def forward(self, x):
        # x: [B, S, D], out: [B, S, D]
        h = x + self.sdsa(self.sdsa_norm(x))
        out = h + self.ffn(self.ffn_norm(h))
        return out 

class SDSA(nn.Module):
    def __init__(self, i, model_args=args):
        super().__init__()
        self.args = model_args
        self.n_heads = self.args.n_heads
        self.head_dim = self.args.head_dim
        self.dim = self.args.embed

        self.wk = nn.Linear(self.dim, self.dim, bias=False)
        self.wv = nn.Linear(self.dim, self.dim, bias=False)
        self.wq = nn.Linear(self.dim, self.dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

        #self.q_if = nn.ReLU()
        #self.k_if = nn.ReLU()
        #self.v_if = nn.ReLU()
        #self.o_if = nn.ReLU()

        #self.freq_cis = precompute_freqs_cis(self.head_dim, self.args.ctx_len)

    def forward(self, x):
        B, S, D = x.shape

        Q = self.wq(x)
        V = self.wv(x)
        K = self.wk(x)

        Q = Q.reshape(B, -1, self.n_heads, self.head_dim)
        V = V.reshape(B, -1, self.n_heads, self.head_dim)
        K = K.reshape(B, -1, self.n_heads, self.head_dim)

        Q = Q.transpose(2, 1)
        V = V.transpose(2, 1)
        K = K.transpose(2, 1)  # [B, n_heads, S, head_dim]
        K = F.relu(K)
        Q = F.relu(Q)
        V = F.relu(V)

        QK = Q @ K.transpose(-1, -2) / math.sqrt(self.dim) # [B, n_heads, S, S]
        mask = torch.full(
                (1, 1, S, S), float("-inf"), device=self.args.device
            )
        mask = torch.triu(mask, diagonal=1).type_as(x)
        QK = QK + mask
        QK = torch.softmax(QK, dim=-1) # [B, n_heads, S, S]
        #QK = F.relu(QK)
        QKV = QK @ V # [B, n_heads, S, head_dim]

        QKV = QKV.transpose(2, 1)
        QKV = QKV.reshape(B, S, -1)
        QKV = self.wo(QKV)
        QKV = F.relu(QKV)
        return QKV

class FFN(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.hidden = model_args.ffn_hidden_layer
        self.dim = model_args.embed 
        #self.spike1 = nn.SiLU()
        #self.spike1 = nn.ReLU()
        #self.init_spike = nn.SiLU()
        #self.init_spike = nn.ReLU()
        self.linear1 = nn.Linear(self.dim, self.hidden, bias=False)
        self.linear2 = nn.Linear(self.hidden, self.dim, bias=False)
        
    def forward(self, x):
        out = F.relu(x)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        return out 

class OutputLayer(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args 
        self.output = nn.Linear(self.args.embed, self.args.vocab_size, bias=False)
        self.norm = nn.LayerNorm(self.args.embed)
        #self.out_if = nn.ReLU()
        #self.register_module("norm", self.norm)

    def forward(self, x):
        out = self.norm(x)
        #out = self.out_if(out)
        out = self.output(out)
        return out
