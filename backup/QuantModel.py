import torch 
from torch import nn
import torch.nn.functional as F 
import math
from typing import Optional, Tuple

from .args import *

def act_quantization(b):
    def uniform_quant(x, b):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard
    
    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            x = x.div(alpha)
            x_c = x.clamp(min=0, max=1)
            x_q = uniform_quant(x_c, b)
            ctx.save_for_backward(x, x_q)
            x_q = x_q.mul(alpha)
            return x_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            x, x_q = ctx.saved_tensors
            i = (x > 1).float()
            grad_alpha = (grad_output * (i + (x_q - x) * (1 - i))).sum()
            grad_input = grad_input * (1 - i)
            return grad_input, grad_alpha

    return _uq().apply

class QuantReLU(nn.ReLU):
    def __init__(self):
        super().__init__()
        self.bit = 4
        self.act_alq = act_quantization(self.bit)
        self.act_alpha = torch.nn.Parameter(torch.tensor(8.0))

    def forward(self, x):
        x = F.relu(x)
        return self.act_alq(x, self.act_alpha)

class MySpikeGPT(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args 

        self.encode_layer = nn.Embedding(self.args.vocab_size, self.args.embed)
        self.transformer = [TransformerBlock(i, self.args) for i in range(self.args.n_layers)]
        self.out = OutputLayer(self.args)

        self.pos = nn.Parameter(torch.zeros([self.args.ctx_len, self.args.embed]))
        self.register_parameter('pos', self.pos)

        self.init_spike = QuantReLU()

        for i in range(self.args.n_layers):
            self.register_module('transformer_block'+str(i), self.transformer[i])
        self.register_module('out', self.out)

    def forward(self, x, y=None):
        # x: [B, S], out: [B, S, vocab]
        assert x.shape[1] == self.args.ctx_len, "input sequence length is not equal to ctx_len!"
        out = self.encode_layer(x)
        out = out + self.pos
        out = self.init_spike(out)
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

        self.q_if = QuantReLU()
        self.k_if = QuantReLU()
        self.v_if = QuantReLU()
        self.o_if = QuantReLU()
        self.softmax_if = QuantReLU()
        self.weight_if = QuantReLU()

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
        K = self.k_if(K)
        Q = self.q_if(Q)
        V = self.v_if(V)

        QK = Q @ K.transpose(-1, -2) / math.sqrt(self.dim) # [B, n_heads, S, S]
        mask = torch.full(
                (1, 1, S, S), float("-inf"), device=self.args.device
            )
        mask = torch.triu(mask, diagonal=1).type_as(x)
        QK = QK + mask
        QK = torch.softmax(QK, dim=-1) # [B, n_heads, S, S]
        QK = self.softmax_if(QK)
        QKV = QK @ V # [B, n_heads, S, head_dim]

        QKV = QKV.transpose(2, 1)
        QKV = QKV.reshape(B, S, -1)
        QKV = self.weight_if(QKV)
        QKV = self.wo(QKV)
        QKV = self.o_if(QKV)
        return QKV

class FFN(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args
        self.hidden = model_args.ffn_hidden_layer
        self.dim = model_args.embed 
        #self.spike1 = nn.SiLU()
        self.spike1 = QuantReLU()
        #self.init_spike = nn.SiLU()
        self.init_spike = QuantReLU()
        self.linear1 = nn.Linear(self.dim, self.hidden, bias=False)
        self.linear2 = nn.Linear(self.hidden, self.dim, bias=False)
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.init_spike(out)
        out = self.linear2(out)
        out = self.spike1(out)
        return out 

class OutputLayer(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args 
        self.output = nn.Linear(self.args.embed, self.args.vocab_size, bias=False)
        #self.norm = nn.LayerNorm(self.args.embed)
        #self.out_if = QuantReLU()
        #self.register_module("norm", self.norm)

    def forward(self, x):
        #out = self.norm(x)
        #out = self.out_if(out)
        out = self.output(x)
        return out
