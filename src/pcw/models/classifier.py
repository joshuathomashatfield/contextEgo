"""Matched visual head with optional trainable causal GRU and memory fusion."""
import torch
from torch import nn

class StepModel(nn.Module):
    def __init__(self, features, classes, memory_dim, aggregation="mean", hidden=32):
        super().__init__(); self.aggregation=aggregation
        self.temporal=nn.GRU(features,hidden,batch_first=True) if aggregation=="gru" else None
        self.head=nn.Linear((hidden if self.temporal else features)+memory_dim,classes)
    def forward(self, sequence, memory):
        visual=self.temporal(sequence.unsqueeze(0))[1][-1,0] if self.temporal else sequence.mean(0)
        return self.head(torch.cat([visual,memory],dim=-1))
