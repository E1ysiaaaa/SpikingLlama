import torch
import torch.nn as nn
import IF_cppcuda

class IFFunc(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, x, alpha, step, SOP):
        assert x.device == alpha.device, f"input and threshold tensor not on the same device, {x.device} and {alpha.device}"
        return IF_cppcuda.forward_cu(x, alpha, step, SOP)
    
    @staticmethod
    def backward(ctx, grad_out):
        return IF_cppcuda.backward_cu(x, alpha, step, SOP)

class IF(nn.Module):
    SOP = 0
    def __init__(self, step=255):
        super().__init__()
        self.step = step
        self.alpha = nn.Parameter(torch.tensor(8.0))

    def forward(self, x):
        out = IFFunc.apply(x.contiguous(), self.alpha, self.step, IF.SOP)
        return out
