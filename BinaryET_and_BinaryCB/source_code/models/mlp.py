import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaModel

class MLP(nn.Module):
    '''MLP model with differt number of nodes (hid_dim) for each hidden layer'''
    def __init__(self, inp_dim, hid_dims):
        super().__init__()
        self.inp_dim = inp_dim
        self.hid_dims = hid_dims
        #num_hid_layers = len(hid_dims)
        self.fc1 = nn.Linear(inp_dim, hid_dims[0])
        #self.out = nn.Linear(hid_dims[-1], 1) #removed  31/3/24 (bug)
        top_hid_dim_idx = len(hid_dims) - 1
        self.hid_list = nn.ModuleList([nn.Linear(hid_dims[i], hid_dims[i + 1]) 
                                       for i in range(top_hid_dim_idx)])

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        for hidden in self.hid_list:
            x = F.relu(hidden(x))
        return x
    
class OmicMlpMarker(nn.Module):
    '''Omic mlp input and marker drug input '''
    def __init__(self, hps, inp_dim):
        super().__init__()
        omic_n_nodes = hps['omic_n_nodes']
        n_nodes1 = hps['n_nodes1']
        n_nodes2 = hps['n_nodes2']
        self.omic_encoder = MLP(inp_dim=inp_dim, hid_dims=omic_n_nodes)
        #177 is number of drugs we have
        self.reg = BinaryRegressor(hps, inp_dim=omic_n_nodes[-1] + 177)

    def forward(self, xo, xd):
        xo = self.omic_encoder(xo)
        # concat
        xc = torch.cat((xo, xd), 1)
        out = self.reg(xc)

        return out

    
    
class BinaryRegressor(torch.nn.Module):
    '''Does regression using an mlp with sigmoid output'''
    #only made drop out optinal 18/4
    def __init__(self, hps, inp_dim=192, drop_out=True):
        super().__init__()

        n_nodes1 = hps['n_nodes1']
        n_nodes2 = hps['n_nodes2']
        if drop_out:
            self.drop_out = hps['drop_out']

        # combined layers
        #self.fc1 = nn.LazyLinear(n_nodes1)
        self.fc1 = nn.Linear(inp_dim, n_nodes1)
        self.fc2 = nn.Linear(n_nodes1, n_nodes2)
        self.out = nn.Linear(n_nodes2, 1)
        self.dropout = nn.Dropout(self.drop_out)

    def forward(self, xr):
        xr = self.fc1(xr)
        if self.drop_out:
            xr = self.dropout(xr)
        xr = F.relu(xr)
        xr = self.fc2(xr)
        if self.drop_out:
            xr = self.dropout(xr)
        xr = F.relu(xr)
        if self.drop_out:
            xr = self.dropout(xr)
        xr = self.out(xr)
        return F.sigmoid(xr)
    
class MarkerOmicRo(nn.Module):
    def __init__(self, hps, db_model_version='seyonec/ChemBERTa-zinc-base-v1'):
        super().__init__()
        self.drug_encoder = RobertaModel.from_pretrained(
            db_model_version, output_attentions=True, output_hidden_states=True)
        self.reg = BinaryRegressor(hps)
        
    def forward(self, xo, xd):

        xd = self.drug_encoder(**xd)
        xd = torch.mean(xd['last_hidden_state'], 1)
        
        xc = torch.cat((xo, xd), 1)
        out = self.reg(xc)

        return out
    
class MarkerBoth(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.reg = BinaryRegressor(hps)
        
    def forward(self, xo, xd):
        xc = torch.cat((xo, xd), 1)
        out = self.reg(xc)
        return out


    
class OmicMlpRo(nn.Module):

    def __init__(self, hps, inp_dim, db_model_version='seyonec/ChemBERTa-zinc-base-v1'):
        super().__init__()

        list_omic_nodes = hps['omic_n_nodes']
        if db_model_version in ['DeepChem/ChemBERTa-10M-MLM', 'DeepChem/ChemBERTa-77M-MLM', 'DeepChem/ChemBERTa-5M-MLM']:
            reg_inp_dim = list_omic_nodes[-1] + 384
        else:
            reg_inp_dim = list_omic_nodes[-1] + 768
        self.omic_encoder = MLP(inp_dim=inp_dim, hid_dims=list_omic_nodes)
        self.reg = BinaryRegressor(hps, inp_dim=reg_inp_dim)
        self.drug_encoder =  RobertaModel.from_pretrained(
            db_model_version, output_attentions=True, output_hidden_states=True)


    def forward(self, xo, xd):

        xd = self.drug_encoder(**xd)
        xd = torch.mean(xd['last_hidden_state'], 1)

        xo = self.omic_encoder(xo)
        
        xc = torch.cat((xo, xd), 1)
        out = self.reg(xc)

        return out