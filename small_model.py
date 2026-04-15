import torch 
import torch.nn as nn

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class RNN(nn.Module):
    def __init__(self, input_size=625, hidden_size=10, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
                          batch_first=False
                          )
        
    def forward(self, X, h_prev=None):
        if X.dim() == 2: 
            X = X.reshape(-1)
        
        X = X.unsqueeze(0).unsqueeze(0)

        if h_prev is None:
            h_prev = torch.zeros(
                self.num_layers, 1, self.hidden_size,
                device=X.device, dtype=X.dtype
            )

        out, h_next = self.rnn(X, h_prev)

        neural_state = h_next[-1, 0, :]   # shape: (hidden_size,)
        return neural_state, h_next