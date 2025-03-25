import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Twin_CNNS(nn.Module):
    '''Twin CNN (tCNN) implementation in torch
     from paper Improving prediction of phenotypic drug response on
     cancer cell lines using deep convolutional network
     original implementation is in Keras
    '''
    
    def __init__(self, in_channels=188, drug_branch=True):
        super().__init__()
        self.drug_branch = drug_branch
        conv_width = 7
        conv_stride = 1
        self.pool_width  = 3
        #pool_stride = 3 same as pool_width by defalt
        num_channels = [40, 80, 60]
        hid_neurons = 1024
        self.drop_out = 0.5 #drop out prob defalt already 0.5
        
        #omic cnn layers
        self.ocnn1 = nn.Conv1d(in_channels=1, out_channels=num_channels[0], 
                               kernel_size=conv_width, stride=conv_stride)
        self.ocnn2 = nn.Conv1d(in_channels=num_channels[0], out_channels=num_channels[1], 
                                kernel_size=conv_width, stride=conv_stride)
        self.ocnn3 = nn.Conv1d(in_channels=num_channels[1], out_channels=num_channels[2], 
                                kernel_size=conv_width, stride=conv_stride)
        self.oflaten = nn.Flatten()
        
        #drug cnn layers
        self.dcnn1 = nn.Conv1d(in_channels=in_channels, out_channels=num_channels[0], 
                               kernel_size=conv_width, stride=conv_stride)
        self.dcnn2 = nn.Conv1d(in_channels=num_channels[0], out_channels=num_channels[1], 
                                kernel_size=conv_width, stride=conv_stride)
        self.dcnn3 = nn.Conv1d(in_channels=num_channels[1], out_channels=num_channels[2], 
                                kernel_size=conv_width, stride=conv_stride)
        self.dflaten = nn.Flatten()
        
        #feed forward layers after omic / drug concat
        self.ff1 = nn.LazyLinear(hid_neurons)
        self.ff2 = nn.Linear(hid_neurons, hid_neurons)
        self.ff3 = nn.Linear(hid_neurons, 1)
        
    def forward(self, xo, xd):
        
        #encode omic data xo
        xo = F.relu(self.ocnn1(xo))
        xo = F.max_pool1d(xo, self.pool_width)
        xo = F.relu(self.ocnn2(xo))
        xo = F.max_pool1d(xo, self.pool_width)
        xo = F.relu(self.ocnn3(xo))
        xo = F.max_pool1d(xo, self.pool_width)
        xo = self.oflaten(xo)
        
        if self.drug_branch:
            #encode drug data xd
            xd = F.relu(self.dcnn1(xd))
            xd = F.max_pool1d(xd, self.pool_width)
            xd = F.relu(self.dcnn2(xd))
            xd = F.max_pool1d(xd, self.pool_width)
            xd = F.relu(self.dcnn3(xd))
            xd = F.max_pool1d(xd, self.pool_width)
            xd = self.dflaten(xd)
        
        concat = torch.cat((xo, xd), dim=1) #dim=0 gives batchsize
        #feed forward 
        output = self.ff1(concat)
        output = F.relu(output)
        output = F.dropout(output, p=self.drop_out) #defalt 0.5
        output = self.ff2(output)
        output = F.relu(output)
        output = F.dropout(output, p=self.drop_out)
        output = self.ff3(output)
        #output = F.sigmoid(output)
        
        return output