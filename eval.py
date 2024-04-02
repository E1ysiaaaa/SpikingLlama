from src.model import GPT
from src.spike_model import SpikeGPT, IF, SpikeInnerProduct
from src.model import GPT
from src.quant_model import QuantGPT
from src.config import Config
from src.tokenizer import Tokenizer

import torch
from torch import nn
import os
from collections import namedtuple

from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import math
from typing import List, Literal, Optional, Tuple, TypedDict

import transformers
from transformers import AutoTokenizer
from tokenizers import Tokenizer as HFTokenizer

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate

class SpikeGPTFull(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        model_name = "tiny_LLaMA_1b"
        self.config = Config.from_name(model_name)
        self.model = GPT(self.config)
        checkpoint = torch.load(pretrained)
        self.model.load_state_dict(checkpoint, strict=False)
        tokenizer_path = Path("checkpoints/")
        self.tokenizer = Tokenizer(tokenizer_path)

    def forward(self, x, max_len=None, input_pos=None):
        lm_logits = self.model.forward(x, max_len, input_pos)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        config = self.config
        max_seq_len = 512
        bsz = len(prompt_tokens)
        assert bsz <= max_seq_len, (bsz, max_seq_len)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= max_seq_len
        total_len = min(max_seq_len, max_gen_len + max_prompt_len)

        pad_id = 0
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            '''
            if tokenizer.eos_id in toks:
                eos_idx = toks.index(tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            '''
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def sample_top_p(self, probs, p):
        """
        Perform top-p (nucleus) sampling on a probability distribution.

        Args:
            probs (torch.Tensor): Probability distribution tensor.
            p (float): Probability threshold for top-p sampling.

        Returns:
            torch.Tensor: Sampled token indices.

        Note:
            Top-p sampling selects the smallest set of tokens whose cumulative probability mass
            exceeds the threshold p. The distribution is renormalized based on the selected tokens.

        """
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

@register_model("spikellama")
class SpikeLlamaWrapper(HFLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained="out/spiking-llama-1b/teacher.pth", max_length=512, batch_size=None, device="cuda",
                 dtype=torch.float32):
        LM.__init__(self)
        self._model = SpikeGPTFull(pretrained).to(device=device, dtype=dtype)

        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()

if __name__ == "__main__":
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
    cli_evaluate()
