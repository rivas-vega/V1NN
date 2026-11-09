"""Causal positive-rate RNN with no pretrained visual encoder or readout bypass."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .graph import Graph


class FixedRetina(nn.Module):
    """Stateless centre-surround ON/OFF filter, not a fitted retina.

    [B,T,1,H,W] -> [B,T,2*grid*grid]. No temporal kernels, learned weights,
    normalization across time, or pretrained parameters.
    """
    def __init__(self, grid=4):
        super().__init__()
        if grid < 1:
            raise ValueError("grid must be positive")
        self.grid = grid

    def forward(self, frames):
        if frames.ndim != 5 or frames.shape[2] != 1:
            raise ValueError("Expected [batch,time,1,height,width]")
        b,t,_,height,width = frames.shape
        if min(height,width) < 5:
            raise ValueError("Images must be at least 5x5")
        flat = frames.reshape(b*t,1,height,width)
        centre = F.avg_pool2d(F.pad(flat,(1,1,1,1),mode="reflect"),3,stride=1)
        surround = F.avg_pool2d(F.pad(flat,(2,2,2,2),mode="reflect"),5,stride=1)
        contrast = centre-surround
        onoff = torch.cat([F.relu(contrast),F.relu(-contrast)],dim=1)
        return F.adaptive_avg_pool2d(onoff,self.grid).flatten(1).reshape(b,t,-1)


class ConnectomeRNN(nn.Module):
    """One trainable weight per directed neuronal pair; Dale sign is presynaptic.

    Unknown outgoing signs are rejected. Row absolute-sum cap is a sufficient
    contraction condition, an engineering safeguard, NOT a cortical finding.
    Fixed random sensor routing is diagnostic, NOT measured retinotopy.
    """
    def __init__(self, graph: Graph, input_dim: int, output_dim: int,
                 alpha=0.3, gain_cap=0.95, fan_in=4, routing_seed=0):
        super().__init__()
        if not 0 < alpha <= 1 or not 0 < gain_cap < 1:
            raise ValueError("Require 0<alpha<=1 and 0<gain_cap<1")
        if min(input_dim,output_dim,fan_in) < 1:
            raise ValueError("Feature dimensions and fan_in must be positive")
        if len(graph.src) and (graph.signs[graph.src] == 0).any():
            raise ValueError("Unknown outgoing E/I signs: document a policy first")
        self.n,self.input_dim = graph.n,input_dim
        self.alpha,self.gain_cap = float(alpha),float(gain_cap)
        self.register_buffer("src",torch.tensor(graph.src,dtype=torch.long))
        self.register_buffer("dst",torch.tensor(graph.dst,dtype=torch.long))
        self.register_buffer("edge_sign",torch.tensor(graph.signs[graph.src]))
        degree = np.bincount(graph.dst,minlength=graph.n)
        magnitude = 0.5*gain_cap/np.maximum(1,degree[graph.dst])
        theta = np.log(np.expm1(magnitude)).astype(np.float32)
        self.theta = nn.Parameter(torch.tensor(theta))
        generator = torch.Generator().manual_seed(routing_seed)
        self.register_buffer("routing",torch.randint(input_dim,(graph.n,fan_in),generator=generator))
        self.log_input_gain = nn.Parameter(torch.zeros(graph.n))
        self.bias = nn.Parameter(torch.zeros(graph.n))
        self.readout = nn.Linear(graph.n,output_dim)

    def effective_weights(self):
        magnitudes = F.softplus(self.theta)
        row_mass = magnitudes.new_zeros(self.n).index_add(0,self.dst,magnitudes)
        denominator = (row_mass[self.dst]/self.gain_cap).clamp_min(1.0)
        return self.edge_sign*magnitudes/denominator

    def recurrent_current(self,state,weights=None):
        if weights is None:
            weights = self.effective_weights()
        return state.new_zeros(state.shape).index_add(1,self.dst,state[:,self.src]*weights[None,:])

    def forward(self,inputs,state=None,reset_each_frame=False,edge_scale=1.0):
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim or inputs.shape[1] == 0:
            raise ValueError("Expected nonempty [batch,time,input_dim]")
        if state is None:
            state = inputs.new_zeros((inputs.shape[0],self.n))
        if state.shape != (inputs.shape[0],self.n):
            raise ValueError("Invalid recurrent state shape")
        weights = self.effective_weights()*edge_scale
        gain = F.softplus(self.log_input_gain)/math.log(2.0)
        states = []
        for x in inputs.unbind(1):
            if reset_each_frame:
                state = torch.zeros_like(state)
            sensory = x[:,self.routing].mean(-1)*gain
            drive = self.recurrent_current(state,weights)+sensory+self.bias
            state = (1-self.alpha)*state+self.alpha*F.relu(drive)
            states.append(state)
        hidden = torch.stack(states,dim=1)
        return self.readout(hidden),hidden
