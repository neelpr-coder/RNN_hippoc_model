import torch
import torch.nn as nn


class connected_models(nn.Module):
    def __init__(self, in_size1=625, in_size2=625, hidden_size1=10, hidden_size2=10, num_inputs_next_layer=3, cross_strength=0.1):
        super().__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_inputs_next_layer = num_inputs_next_layer
        self.cross_strength = cross_strength

        # connect the layers of the two networks.
        self.in_current_a = nn.Linear(in_size1, hidden_size1)
        self.in_current_b = nn.Linear(in_size2, hidden_size2)
        
        self.connectAA = nn.Linear(hidden_size1, hidden_size1, bias=False)
        self.connectBB = nn.Linear(hidden_size2, hidden_size2, bias=False)

        ab_mapping = torch.stack([torch.randperm(hidden_size1)[:num_inputs_next_layer] for _ in range(hidden_size2)])
        ba_mapping = torch.stack([torch.randperm(hidden_size2)[:num_inputs_next_layer] for _ in range(hidden_size1)])

        self.register_buffer("ab_mapping", ab_mapping)
        self.register_buffer("ba_mapping", ba_mapping)

    def connection_current(self, h_source, mapping):
        if h_source.dim() == 2: 
            h_source = h_source.squeeze(0)

        summed_activations = h_source[mapping].sum(dim=1)
        current = summed_activations

        return current

    def forward(self, x_a, x_b, h_a, h_b):
        x_a = x_a.reshape(1, -1)
        x_b = x_b.reshape(1, -1)
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

        h_a_next = torch.tanh(input_current_a + I_aa + self.cross_strength * I_ba)
        h_b_next = torch.tanh(input_current_b + I_bb + self.cross_strength * I_ab)

        return h_a_next, h_b_next, I_aa, I_ab, I_bb, I_ba

    def knockout_neuron(self, index, region="both"):
        if region not in ("a", "b", "both"):
            raise ValueError("region must be 'a', 'b', or 'both'")
        if region in ("a", "both") and not 0 <= index < self.hidden_size1:
            raise IndexError(f"RNN A index {index} is outside [0, {self.hidden_size1 - 1}]")
        if region in ("b", "both") and not 0 <= index < self.hidden_size2:
            raise IndexError(f"RNN B index {index} is outside [0, {self.hidden_size2 - 1}]")

        def knockout_hook(module, inputs, output):
            h_a, h_b, I_aa, I_ab, I_bb, I_ba = output
            perturbed_h_a = h_a.clone()
            perturbed_h_b = h_b.clone()

            if region in ("a", "both"):
                perturbed_h_a[..., index] = 0.0
            if region in ("b", "both"):
                perturbed_h_b[..., index] = 0.0
            return perturbed_h_a, perturbed_h_b, I_aa, I_ab, I_bb, I_ba
        return self.register_forward_hook(knockout_hook)
