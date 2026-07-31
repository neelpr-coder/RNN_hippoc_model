import torch
import torch.nn as nn


class connected_models(nn.module):
    def __innit__(self, in_size1=625, in_size2=625, hidden_size1=10, hidden_size2=20):
        super().__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        # connect the layers of the two networks.
        self.in_current_a = nn.Linear(in_size1, hidden_size1)
        self.in_current_b = nn.Linear(in_size2, hidden_size2)
        self.connectAB = nn.Linear(hidden_size1, hidden_size2)
        self.connectBA = nn.Linear(hidden_size2, hidden_size1)
        self.connectAA = nn.Linear(hidden_size1, hidden_size1)
        self.connectBB = nn.Linear(hidden_size2, hidden_size2)

    def forward(self, x_a, x_b, h_a, h_b):
        I_aa = self.connectAA(h_a)
        I_ab = self.connectAB(h_b)
        I_bb = self.connectBB(h_b)
        I_ba = self.connectBA(h_a)

        input_current_a = self.in_current_a(x_a)
        input_current_b = self.in_current_b(x_b)

        # incomplete finish later