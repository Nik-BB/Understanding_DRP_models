from torch import nn
import torch.nn.functional as F
class MLP(nn.Module):
    '''MLP model with same number of nodes (hid_dim) for each hidden layer'''
    def __init__(self, inp_dim, hid_dim, num_hid_layers=2):
        super().__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.out = nn.Linear(hid_dim, 1)
        self.hid_list = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for _ in range(num_hid_layers)])

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        for hidden in self.hid_list:
            x = F.relu(hidden(x))
        return F.sigmoid(self.out(x))
        