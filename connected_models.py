import torch
import torch.nn as nn


class connected_models(nn.Module):
    def __init__(self, in_size1=625, in_size2=625, hidden_size1=10, hidden_size2=20, num_inputs_next_layer=3):
        super().__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_inputs_next_layer = num_inputs_next_layer

        # connect the layers of the two networks.
        self.in_current_a = nn.Linear(in_size1, hidden_size1)
        self.in_current_b = nn.Linear(in_size2, hidden_size2)
        
        self.connectAA = nn.Linear(hidden_size1, hidden_size1, bias=False)
        self.connectBB = nn.Linear(hidden_size2, hidden_size2, bias=False)

        ab_mapping = torch.stack([torch.randperm(hidden_size1)[:num_inputs_next_layer] for _ in range(hidden_size2)])
        ba_mapping = torch.stack([torch.randperm(hidden_size2)[:num_inputs_next_layer] for _ in range(hidden_size1)])

        self.register_buffer("ab_mapping", ab_mapping)
        self.register_buffer("ba_mapping", ba_mapping)

    def connection_current(self, h_source, map):
        if h_source.dim() == 2: 
            h_source = h_source.unsqueeze(0)

        summed_activations = h_source[map].sum(dim=1)
        current = summed_activations

        return current

    def forward(self, x_a, x_b, h_a, h_b):
        I_aa = self.connectAA(h_a)
        I_ab = self.connection_current(h_a, self.ab_mapping)
        I_bb = self.connectBB(h_b)
        I_ba = self.connection_current(h_b, self.ba_mapping)

        if I_ab.dim() == 1: 
            I_ab = I_ab.unsqueeze(0)
        if I_ba.dim() == 1: 
            I_ba = I_ba.unsqueeze(0)

        input_current_a = self.in_current_a(x_a)
        input_current_b = self.in_current_b(x_b)

        h_a_next = torch.tanh(input_current_a + I_aa + I_ba)
        h_b_next = torch.tanh(input_current_b + I_bb + I_ab)

        return h_a_next, h_b_next, I_aa, I_ab, I_bb, I_ba
