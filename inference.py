from src.model import GPT
from src.config import Config
from src.tokenizer import Tokenizer
from src.spike_model import SpikeGPT, IF, SpikeInnerProduct
from src.quant_model import QuantGPT

from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
import torch



@torch.inference_mode()
def generate(
    model, 
    tokenizer,
    prompt_token,
    max_gen_len: int,
    temperature: float = 0.6,
    top_p: float = 0.9
):

    config = model.config
    max_seq_len = config.block_size
    total_len = min(max_seq_len, len(prompt_token)+max_gen_len)

    token = prompt_token 
    token = token.unsqueeze(0).to("cuda")

    for cur_pos in range(len(prompt_token), total_len):
        logits = model.forward(token[:, 0:cur_pos], max_seq_length=cur_pos)
        if temperature > 0:
            probs = torch.softmax(logits[-1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[-1], dim=-1)

        # only replace token if prompt has already been generated
        token = torch.cat((token, next_token[-1].unsqueeze(0)), dim=-1)

        if next_token[-1, 0] == tokenizer.eos_id:
            break

    return token

def sample_top_p(probs, p):
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

def main():
    model_name = "tiny_LLaMA_1b"
    checkpoint_path = "out/spiking-llama-1b/iter-060000-ckpt.pth"
    #checkpoint_path = "out/spiking-llama-1b/teacher.pth"
    tokenizer_path = Path("checkpoints/")
    tokenizer = Tokenizer(tokenizer_path)
    config = Config.from_name(model_name)
    model = QuantGPT(config).to("cuda")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"], strict=False)

    input_text = "All at once she understood why this whole day she'd felt even more consumed by Daniel than usual, and also even more conflicted about Cam. She'd dreamed about them both last night. She'd been wandering through a dusty fog, someone holding her hand. She'd turned, thinking it would be "
    max_gen_len = 20
    input_tokens = tokenizer.encode(input_text)
    prompt_tokens = input_tokens[:-1]
    out_tokens = generate(model, tokenizer, prompt_tokens, max_gen_len)
    out_text = tokenizer.decode(torch.tensor(out_tokens[0]))
    print(out_text)

if __name__ == "__main__":
    main()
