'''Model  from Graph Convolutional Networks for DrugResponse Prediction paper
https://github.com/hauldhut/GraphDRP/blob/master/models/gat.py

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, GCNConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, 
                 omic_type='exp', drug_branch=True):

        super(GINConvNet, self).__init__()

        self.drug_branch = drug_branch
        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)

        # cell line feature
        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)

        if omic_type == 'exp':
            n_fc1_nodes = 2944
        elif omic_type == 'rna':
            n_fc1_nodes = 82048
            
        self.fc1_xt = nn.Linear(n_fc1_nodes, output_dim)

        # combined layers
        n_comb_nodes = 2*output_dim
        if drug_branch == False:
            n_comb_nodes = 309
        self.fc1 = nn.Linear(n_comb_nodes, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, g_data, xo):
        if self.drug_branch:
            x, edge_index, batch = g_data.x, g_data.edge_index, g_data.batch
            #print(x)
            #print(data.target)
            x = F.relu(self.conv1(x, edge_index))
            x = self.bn1(x)
            x = F.relu(self.conv2(x, edge_index))
            x = self.bn2(x)
            x = F.relu(self.conv3(x, edge_index))
            x = self.bn3(x)
            x = F.relu(self.conv4(x, edge_index))
            x = self.bn4(x)
            x = F.relu(self.conv5(x, edge_index))
            x = self.bn5(x)
            x = global_add_pool(x, batch)
            x = F.relu(self.fc1_xd(x))
            x = F.dropout(x, p=0.2, training=self.training)

        else:
            x = g_data


        # 1d conv layers
        conv_xt = self.conv_xt_1(xo)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)
        
        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)
        
        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        #removed sigmoid as not using normed target vals (nb)
        #out = nn.Sigmoid()(out)
        return out



# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.5):

        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # cell line feature
        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)
        self.fc1_xt = nn.Linear(2944, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, self.n_output)

    def forward(self, g_data, xo):
        # get graph input
        x, edge_index, batch = g_data.x, g_data.edge_index, g_data.batch

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # 1d conv layers
        conv_xt = self.conv_xt_1(xo)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # print('conv_xt', conv_xt.size())

        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        #removed sigmoid as not using normed target vals (nb)
        #out = nn.Sigmoid()(out)
        return out 