import torch
import numpy as np
import pandas as pd
import torch_geometric.data as tgd

def create_graphs(drugs_with_smiles, y):
    edge_indexs = pd.read_pickle('data/graph_edge_indices_gdrp.pkl')
    graph_features = pd.read_pickle('data/graph_features_gdrp.pkl')
    drugs_to_graphs = {}

    for d, ft, edg_inds in zip(drugs_with_smiles, graph_features, edge_indexs):
        edg_inds = torch.tensor(edg_inds, dtype=int).t().contiguous()
        ft = torch.tensor(ft.astype(np.float32))
        drugs_to_graphs[d] = tgd.Data(x=ft, edge_index=edg_inds)

    
    pairs_to_graphs = {}
    for pair in y.index:
        d = pair.split('::')[1]
        y_g = y.loc[pair].astype(np.float32)
        y_g = np.expand_dims(y_g, -1)
        graph = tgd.Data.clone(drugs_to_graphs[d])
        graph.y = torch.tensor(y_g)
        pairs_to_graphs[pair] = graph
        
    x_drug = pd.Series(pairs_to_graphs)

    return x_drug
