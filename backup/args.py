import json
import types 
import torch

args = types.SimpleNamespace()
def load_params(args, file_name):
    with open(file_name, 'r') as f:
        params = json.load(f)
    for key in params.keys():
        setattr(args, key, params[key])
    setattr(args, "head_dim", args.embed // args.n_heads)
    setattr(args, "ffn_hidden_layer", 4 * args.embed)

    if torch.cuda.is_available():
        setattr(args, "device", "cuda")
    elif torch.backends.mps.is_available():
        setattr(args, "device", "mps")
    else:
        setattr(args, "device", "cpu")

load_params(args, "params.json")
