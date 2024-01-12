import torch 
from torch import nn
import torch.nn.functional as F 
import math
from typing import Optional, Tuple
from einops import repeat

from .args import *

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
    def __init__(self, T=T_total, step=time_step):
        super().__init__()
        self.T = T
        self.step = step

    def forward(self, x, y):
        T, B, n_heads, S, head_dim = x.shape
        device = torch.cuda.current_device()
        out_weight = torch.zeros([T, B, n_heads, S, S]).to(device)
        
        for i in range(0, self.T, self.step):
            x_add = 0
            y_add = 0
            for j in range(min(self.step, self.T - i)):
                x_add = x_add + x[i+j] / self.step
                y_add = y_add + y[i+j] / self.step
            weight = x_add @ y_add
            for j in range(min(self.step, self.T - i)):
                out_weight[i+j] = weight
        return out_weight

class RestrictLN(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(dim)

    def forward(self, x):
        device = torch.cuda.current_device()
        T, B, S, D = x.shape
        std = x.std(dim=-1, keepdim=True)
        one = torch.ones(x.shape).to(device)
        restrict_std = torch.where(std > 1, std, one)
        out = (x - x.mean(dim=-1, keepdim=True)) / restrict_std
        return out

class MySpikeGPT(nn.Module):
    def __init__(self, model_args=args):
        super().__init__()
        self.args = model_args 

        self.encode_layer = nn.Embedding(self.args.vocab_size, self.args.embed)
        self.transformer = [TransformerBlock(i, self.args) for i in range(self.args.n_layers)]
        self.out = OutputLayer(self.args)

        self.pos = nn.Parameter(torch.zeros([self.args.ctx_len, self.args.embed]))
        self.register_parameter('pos', self.pos)

        self.init_spike = IF(step=time_step)

        for i in range(self.args.n_layers):
            self.register_module('transformer_block'+str(i), self.transformer[i])
        self.register_module('out', self.out)

    def forward(self, x, y=None):
        # x: [B, S], out: [B, S, vocab]
        assert x.shape[1] == self.args.ctx_len, "input sequence length is not equal to ctx_len!"
        out = self.encode_layer(x)
        out = out + self.pos
        out = repeat(out, "b s d -> t b s d", t=T_total)
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
        #self.sdsa_norm = nn.LayerNorm(self.args.embed)
        self.sdsa_norm = RestrictLN(self.args.embed)
        #self.ffn_norm = nn.LayerNorm(self.args.embed)
        self.ffn_norm = RestrictLN(self.args.embed)

        self.register_module('ffn', self.ffn)
        self.register_module('sdsa', self.sdsa)
        #self.register_module('sdsa_norm', self.sdsa_norm)
        #self.register_module('ffn_norm', self.ffn_norm)

    def forward(self, x):
        # x: [B, S, D], out: [B, S, D]
        #x_real = x.mean(dim=0)
        #h_real = x_real + self.sdsa(repeat(self.sdsa_norm(x_real), 'b s l -> t b s l', t=T_total)).mean(dim=0)
        #out = repeat(h_real, 'b s l -> t b s l', t=T_total) + self.ffn(repeat(self.ffn_norm(h_real), 'b s l -> t b s l', t=T_total))
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

        self.q_if = IF(step=time_step)
        self.k_if = IF(step=time_step)
        self.v_if = IF(step=time_step)
        self.o_if = IF(step=time_step)
        self.softmax_if = IF(step=time_step)
        self.weight_if = IF(step=time_step)
        self.inner_product = SpikeInnerProduct(step=time_step)

        #self.freq_cis = precompute_freqs_cis(self.head_dim, self.args.ctx_len)

    def forward(self, x):
        T, B, S, D = x.shape

        Q = self.wq(x)
        V = self.wv(x)
        K = self.wk(x)

        Q = Q.reshape(T, B, -1, self.n_heads, self.head_dim)
        V = V.reshape(T, B, -1, self.n_heads, self.head_dim)
        K = K.reshape(T, B, -1, self.n_heads, self.head_dim)

        Q = Q.transpose(2, 3)
        V = V.transpose(2, 3)
        K = K.transpose(2, 3)  # [T, B, n_heads, S, head_dim]
        K = self.k_if(K)
        Q = self.q_if(Q)
        V = self.v_if(V)

        QK = self.inner_product(Q, K.transpose(-1, -2)) / math.sqrt(self.dim) # [T, B, n_heads, S, S]
        mask = torch.full(
                (1, 1, 1, S, S), float("-inf"), device=self.args.device
            )
        mask = torch.triu(mask, diagonal=1).type_as(x)
        QK = QK + mask
        QK = torch.softmax(QK, dim=-1) # [T, B, n_heads, S, S]
        QK = QK.clamp(0, self.softmax_if.alpha.item())
        QKV = QK @ V # [T, B, n_heads, S, head_dim]

        QKV = QKV.transpose(2, 3)
        QKV = QKV.reshape(T, B, S, -1)
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
        self.spike1 = IF(step=time_step)
        #self.init_spike = nn.SiLU()
        self.init_spike = IF(step=time_step)
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
        out = out[:wait_time].mean(dim=0)
        return out
