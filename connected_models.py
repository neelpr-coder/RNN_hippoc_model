import torch
import torch.nn as nn


class connected_models(nn.module):
    def __innit__(self, input_size=625, hidden_size=10, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self):
        pass