'''transfomer drug branch model'''
import torch
from torch import nn

import os 
from models.mlp import MLP, BinaryRegressor

class TransformerDB(nn.Module):
    '''Transformer model with embeding layer'''
    def __init__(self, hps, vocab_size=880, att_mask=False, xdict=True, pos_enc=False):
        super().__init__()

        #vocab_size = 880 #vocab_size of tokensized with chemberta for smile reps
        drug_emb_dim = hps['drug_emb_dim']
        max_position_size = 128 #max smile length after padding (50 in tta with bpe encoding)
        n_tf_layers = hps['n_tf_layers']
        nhead = hps['tf_atten_heads']
        dim_feedforward = hps['tf_ff_dim']

        self.embedding = nn.Embedding(vocab_size, drug_emb_dim)
        self.pos_enc = pos_enc
        if self.pos_enc:
            self.pos_embeddings = nn.Embedding(max_position_size, drug_emb_dim) #postinal embeddings
        encoder_layer = nn.TransformerEncoderLayer(d_model=drug_emb_dim, nhead=nhead, batch_first=True, dim_feedforward=dim_feedforward)
        self.tf = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_tf_layers)
        self.att_mask = att_mask
        self.xdict = xdict

    def forward(self, x):
        if self.att_mask:
            att_mask = ~x['attention_mask'].bool() #note ~ used to invert mask
        else:
            att_mask = None
        if self.xdict:
            if self.pos_enc:
                pos = torch.arange(x['input_ids'].shape[1], device=x['input_ids'].device)
                pos = pos.unsqueeze(0).expand_as(x['input_ids'])
                pos = self.pos_embeddings(pos)
            xm = self.embedding(x['input_ids'])
        else:
            if self.pos_enc: 
                pos = torch.arange(x.shape[1], device=x.device)
                pos = pos.unsqueeze(0).expand_as(x)
                pos = self.pos_embeddings(pos)
            xm = self.embedding(x)
            
        if self.pos_enc:
            xm = xm + pos
            
        out = self.tf(xm, src_key_padding_mask=att_mask)

        return out

class OmicMLPDrugTF(nn.Module):
    '''drp model with mlp omics branch and Transformer drug branch '''
    def __init__(self, hps, o_inp_dims, vocab_size=880, att_mask=False, xdict=True, pos_enc=False):
        super().__init__()

        omic_n_nodes = hps['omic_n_nodes']
        self.omic_encoder = MLP(inp_dim=o_inp_dims, hid_dims=omic_n_nodes)
        self.drug_encoder = TransformerDB(hps, vocab_size=vocab_size, att_mask=att_mask, xdict=xdict, pos_enc=pos_enc)
        self.reg = BinaryRegressor(hps, inp_dim=omic_n_nodes[-1] + hps['drug_emb_dim'])

    
    def forward(self, xo, xd):
        
        xo = self.omic_encoder(xo)
        xd = self.drug_encoder(xd)
        xd = torch.mean(xd, 1) #average over token embeddings
        #xd = xd[:, 0] #takes rep of first smies token
        xc = torch.cat((xo, xd), 1)
        out = self.reg(xc)
        
        return out