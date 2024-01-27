from src.model import GPT
from src.spike_model import SpikeGPT, IF, SpikeInnerProduct
from src.config import Config
from src.tokenizer import Tokenizer
from inference import generate, sample_top_p

import torch
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import math

def lambada(model, tokenizer):
    dataset = load_dataset("lambada")
    dataset = dataset['test']

    temperature = 0.6
    top_p = 0.9
    total_count = 10
    acc = 0
    ce = 0
    small_set = dataset[:total_count]['text']
    loss_fn = torch.nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(total_count)):
            tokens = tokenizer.encode(small_set[i])
            input_token = torch.tensor(tokens[:-2], device="cuda")
            answer = tokens[-2].item()
            cur_pos = input_token.shape[0]
            pred = model.forward(input_token.unsqueeze(0), cur_pos)
            ce += loss_fn(pred[0], torch.tensor(tokens[1: -1], device="cuda", dtype=torch.long))
            probs = torch.softmax(pred / temperature, dim=-1)
            next_token = sample_top_p(probs[:, -1, :], top_p)
            if next_token[0] == answer:
                acc += 1
    acc = 1.0 * acc / total_count
    ce = ce / total_count
    print(f"lambada accuracy: {acc}, ppl: {math.exp(ce)}, SOP: {(IF.SOP + SpikeInnerProduct.SOP) / total_count}")
    
def wikitext2(model, tokenizer):
    dataset = load_dataset("wikitext")
    dataset = dataset['wikitext-2-v1']

    temperature = 0.6
    top_p = 0.9
    total_count = 100
    ce = 0
    small_set = dataset[:total_count]
    loss_fn = torch.nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(total_count)):
            tokens = tokenizer.encode(small_set[i]).to("cuda")
            input_token = tokens[:-2]
            pred = model.forward(input_token.unsqueeze(0))
            ce += loss_fn(pred[0], torch.tensor(tokens[1: -1], device="cuda", dtype=torch.long))

    ce = ce / total_count
    print(f"wikitext2 ppl: {ce}")

def main(task: str):
    model_name = "tiny_LLaMA_120M"
    checkpoint_path = "out/spiking-llama-120m/iter-075000-ckpt.pth"
    tokenizer_path = Path("checkpoints/")
    tokenizer = Tokenizer(tokenizer_path)
    config = Config.from_name(model_name)
    model = SpikeGPT(config).to("cuda")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'], strict=False)

    if task == "lambada":
        lambada(model, tokenizer)

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
