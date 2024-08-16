import torch
from src.quant_model import QuantGPT, act_quantization
from csrc.quant_relu.interface import *
from src.model import GPT
from src import Config
from time import time
import random

'''
model_name = "tiny_LLaMA_120M"
config = Config.from_name(model_name)
model = QuantGPT(config).to("cuda")

torch.manual_seed(0)
x = torch.randint(1, 3000, [1, 2048])
x = x.cuda()
'''

y = random.random()
x = torch.rand([2048, 768], dtype=torch.bfloat16).cuda().requires_grad_(True)
alpha = torch.tensor(y, dtype=torch.bfloat16).cuda().requires_grad_(True)

print("----------forward-----------")
torch.cuda.synchronize()
start = time()
y1 = act_quantization(4)(x, alpha)
torch.cuda.synchronize()
end = time()
print(f"original: {end-start}s")
print(y1)
torch.cuda.synchronize()
start = time()
y2 = quant_relu(x, alpha, 4)
torch.cuda.synchronize()
end = time()
print(f"custom: {end-start}s")
print(y2)

print("------------backward----------")
dummy_grad = torch.rand(y1.shape, dtype=torch.bfloat16).cuda()
alpha.grad = torch.zeros(alpha.shape, dtype=torch.bfloat16).cuda()
torch.cuda.synchronize()
start = time()
y1.backward(dummy_grad)
torch.cuda.synchronize()
end = time()
print(f"original: {end-start}s")
print(alpha.grad)
alpha.grad = torch.zeros(alpha.shape, dtype=torch.bfloat16).cuda()
torch.cuda.synchronize()
start = time()
grad2 = y2.backward(dummy_grad)
torch.cuda.synchronize()
end = time()
print(f"custom: {end-start}s")
print(alpha.grad)
