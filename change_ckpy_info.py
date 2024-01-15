import torch

state = torch.load('out/spiking-llama-120m/iter-035000-ckpt.pth')
state['iter_num'] = 0
state['step_count'] = 0
torch.save(state, 'out/spiking-llama-120m/iter-035000-ckpt.pth')
