'''Run tcnns and graphDRP with rna instead of genomics omic proflies'''

import os
import sys
import torch
import numpy as np 
import pandas as pd
import pubchempy as pcp
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as DataLoaderGeo
import torch_geometric.data as tgd
import data_loading, data_processing  
import ttc_data_processing, tcnn_data_processing, graphdrp_data_processing
import training, inference
from models import tta, twin_model, graphdrp

epochs = int(sys.argv[1])
example_data = False
supported_models = ['tcnn_rna', 'graphdrp_rna', 'tcnn_rna-db', 'graphdrp_rna-db']
supported_splits = ['c_blind', 'mixed', 'd_blind']


models_to_run = ['graphdrp_rna-db']#, 'tcnn_rna-db',]
split_type  = 'c_blind'

#path setting
hpc=False
#if os.getcwd()[0] == 'C': #working locally
    #hpc = False
#genral data loading
if hpc:
    omic_dir_path = '../../GDSC/downloaded_data' #hpc
    
else:
    omic_dir_path = 'data' 
    
gdsc2_target_path = 'data/GDSC2_Wed Aug GDSC2_30_15_49_31_2023.csv'
pubchem_ids_path = 'data/drugs_gdsc_to_pubID.csv'

if hpc:
    os.chdir('..')
print(f'using hpc: {hpc}')

def main():
    for model_type in models_to_run:
        #checks
        print(f'Running {model_type} with {split_type} split')
        if model_type not in supported_models:
            raise Exception(f'{model_type} not a supported model')
        if split_type not in supported_splits:
            raise Exception(f'{split_type} not a supported test train split')
        
        if model_type.split('_')[0] == 'tcnn':
            batch_size = 100
        elif model_type.split('_')[0] == 'graphdrp' or model_type.split('_')[0] == 'gcn_gdrp':
            batch_size=102#4
            

        if example_data:
            rna = pd.read_csv('data/example_data/xpr_sub.csv', index_col=0).astype(np.float32)
            ic50 = pd.read_csv('data/example_data/ic50_sub.csv', index_col=0).astype(np.float32)
            drugs_to_smiles = pd.read_csv('data/drugs_to_smiles_gdsc2.csv', index_col=0)['0']
            pairs_path = f'data/example_data/train_test_pairs/{split_type}/'
        #set save=true to create drugs_to_smiles, 
        #and once saved can use save=False to load the same drugs_to_smiles dict
        else:
            rna, ic50, drugs_to_smiles = data_loading.load_omics_drugs_target(
                omic_dir_path, gdsc2_target_path, pubchem_ids_path, save=False)
            pairs_path = f'data/train_test_pairs/{split_type}/'
        drugs_with_smiles = drugs_to_smiles.index

        one_hot_drugs = np.zeros((len(drugs_with_smiles), len(drugs_with_smiles)))
        np.fill_diagonal(one_hot_drugs, 1)
        one_hot_drugs = pd.DataFrame(
            one_hot_drugs, index=drugs_with_smiles, columns=drugs_with_smiles)
        
        if model_type == 'tcnn_rna':
            dts_pad = tcnn_data_processing.pad_smiles(drugs_to_smiles)
            drug_to_hot_smile = tcnn_data_processing.hot_encode_smiles(dts_pad)
            x_omic, x_drug, y = data_processing.create_all_drugs(rna, drug_to_hot_smile, ic50)


        if model_type == 'graphdrp_rna' or model_type == 'gcn_gdrp_rna':
            x_omic, _, y = data_processing.create_all_drugs(rna, one_hot_drugs, ic50)
            x_drug = graphdrp_data_processing.create_graphs(drugs_with_smiles, y)

        if model_type == 'tcnn_rna-db' or model_type == 'graphdrp_rna-db':
            x_omic, x_drug, y = data_processing.create_all_drugs(rna, one_hot_drugs, ic50)
            x_drug = pd.DataFrame(
                {'drug_encoding': [enc.values.astype(np.float32)
                                   for enc in  x_drug['drug_encoding']]},
                index=x_drug.index)
        
        if model_type == 'graphdrp_rna-db':
            pass



        # train and eval model for 3 train test splits and 3 model seeds
        for seed in range(1, 4):

            train_pairs = pd.read_csv(
                f'{pairs_path}seed_{seed}_train', header=None)[0].values
            val_pairs = pd.read_csv(
                f'{pairs_path}seed_{seed}_val', header=None)[0].values
            test_pairs = pd.read_csv(
                f'{pairs_path}seed_{seed}_test', header=None)[0].values

            xd_train, xo_train = x_drug.loc[train_pairs], x_omic.loc[train_pairs]
            xd_val, xo_val = x_drug.loc[val_pairs], x_omic.loc[val_pairs]
            xd_test, xo_test = x_drug.loc[test_pairs], x_omic.loc[test_pairs]
            y_train, y_test = y.loc[train_pairs], y.loc[test_pairs]
            y_val =  y.loc[val_pairs]
            
            # xd_train.shape, xo_train.shape, y_train.shape, \
            #     xd_val.shape, xo_val.shape, y_val.shape, \
            #         xd_test.shape, xo_test.shape, y_test.shape

            if model_type == 'tcnn_rna' or  model_type =='graphdrp_rna' or model_type == 'gcn_gdrp_rna':
                xo_train = DataLoader(np.expand_dims(xo_train.values, 1), 
                                    batch_size=batch_size)
                y_train = DataLoader(np.expand_dims(y_train, 1), 
                                    batch_size=batch_size)
            
                xo_val = DataLoader(np.expand_dims(xo_val.values, 1), 
                                    batch_size=batch_size)
                y_val = DataLoader(np.expand_dims(y_val, 1), 
                                batch_size=batch_size)
            
                xo_test = DataLoader(np.expand_dims(xo_test.values, 1), 
                                    batch_size=batch_size)
                y_test = np.expand_dims(y_test, 1)

            if model_type == 'tcnn_rna-db' or model_type == 'graphdrp_rna-db':

                xo_train = DataLoader(np.expand_dims(xo_train.values, 1), 
                                    batch_size=batch_size)
                y_train = DataLoader(np.expand_dims(y_train, 1), 
                                    batch_size=batch_size)
            
                xo_val = DataLoader(np.expand_dims(xo_val.values, 1), 
                                    batch_size=batch_size)
                y_val = DataLoader(np.expand_dims(y_val, 1), 
                                batch_size=batch_size)
            
                xo_test = DataLoader(np.expand_dims(xo_test.values, 1), 
                                    batch_size=batch_size)
                y_test = np.expand_dims(y_test, 1)

                xd_train = DataLoader(xd_train['drug_encoding'], 
                                    batch_size=batch_size)
                xd_val = DataLoader(xd_val['drug_encoding'], 
                                    batch_size=batch_size)
                xd_test = DataLoader(xd_test['drug_encoding'], 
                                    batch_size=batch_size)

            if model_type == 'tcnn_rna':
                xd_train = DataLoader(xd_train['drug_encoding'], 
                                    batch_size=batch_size)
                xd_val = DataLoader(xd_val['drug_encoding'], 
                                    batch_size=batch_size)
                xd_test = DataLoader(xd_test['drug_encoding'], 
                                    batch_size=batch_size)
                
            if model_type == 'graphdrp_rna' or model_type == 'gcn_gdrp_rna':
                xd_train = DataLoaderGeo(
                    tgd.Batch().from_data_list(xd_train.values), batch_size=batch_size)
                xd_val = DataLoaderGeo(
                    tgd.Batch().from_data_list(xd_val.values), batch_size=batch_size)
                xd_test = DataLoaderGeo(
                    tgd.Batch().from_data_list(xd_test.values), batch_size=batch_size)
            

            #model train
            save_path = f'results/tt_split_{seed}/predictions/{model_type}/'
            model_path =  f'saved_models/{model_type}'
            
            #model traning for 3 seeds
            metrics = {'mse': [], 'spear': []}
            for model_seed in range(3):
                torch.manual_seed(model_seed)
                model_dir = f'{model_path}/{split_type}/'
                
                if model_type == 'tcnn_rna':
                    tcnn = twin_model.Twin_CNNS(in_channels=30)
                    hist = training.tl_multi_dls( 
                        train_dls=[xo_train, xd_train],
                        y_train=y_train,
                        val_dls=[xo_val, xd_val],
                        y_val=y_val,
                        model=tcnn,
                        loss_fn=torch.nn.MSELoss(), 
                        optimiser=torch.optim.Adam(tcnn.parameters(), lr=1e-4),
                        epochs=epochs,
                        early_stopping_dict={'patience': 300, 'delta': 0.0},
                        ms_path=f'{model_dir}_tcnn'
                        )
                    y_pred = inference.predict(xo_test, xd_test, tcnn)

                if model_type == 'graphdrp_rna-db':
                    graph_drp = graphdrp.GINConvNet(drug_branch=False, omic_type='rna')
                    tl, vl = training.tl_multi_dls(
                        train_dls=[xd_train, xo_train],
                        y_train=y_train,
                        val_dls=[xd_val, xo_val],
                        y_val=y_val,
                        model=graph_drp,
                        loss_fn=torch.nn.MSELoss(),
                        optimiser=torch.optim.Adam(graph_drp.parameters(), lr=1e-4),
                        epochs=epochs, 
                        early_stopping_dict={'patience': 300, 'delta': 0.0},
                        ms_path=f'{model_dir}_graph'
                        )
                    y_pred = inference.predict(xd_test, xo_test, graph_drp)
                    print('min val loss')
                    print(min(vl), np.argmin(vl))

                if model_type == 'tcnn_rna-db':
                    tcnn = twin_model.Twin_CNNS(drug_branch=False)
                    hist = training.tl_multi_dls( 
                        train_dls=[xo_train, xd_train],
                        y_train=y_train,
                        val_dls=[xo_val, xd_val],
                        y_val=y_val,
                        model=tcnn,
                        loss_fn=torch.nn.MSELoss(), 
                        optimiser=torch.optim.Adam(tcnn.parameters(), lr=1e-4),
                        epochs=epochs,
                        early_stopping_dict={'patience': 300, 'delta': 0.0},
                        ms_path=f'{model_dir}_tcnn'
                        )
                    y_pred = inference.predict(xo_test, xd_test, tcnn)

                if model_type == 'graphdrp_rna':
                    graph_drp = graphdrp.GINConvNet(num_features_xd=78, omic_type='rna')
                    tl, vl = training.tl_dual_graph(
                        train_dl1=xd_train,
                        train_dl2=xo_train,
                        val_dl1=xd_val,
                        val_dl2=xo_val,
                        model=graph_drp,
                        loss_fn=torch.nn.MSELoss(),
                        optimiser=torch.optim.Adam(graph_drp.parameters(), lr=1e-4),
                        epochs=epochs, 
                        early_stopping_dict={'patience': 300, 'delta': 0.0},
                        ms_path=f'{model_dir}_graph'
                        )
                    y_pred = inference.predict(xd_test, xo_test, graph_drp)

                if model_type == 'gcn_gdrp_rna':
                    graph_drp = graphdrp.GCNNet(num_features_xd=78)
                    tl, vl = training.tl_dual_graph(
                        train_dl1=xd_train,
                        train_dl2=xo_train,
                        val_dl1=xd_val,
                        val_dl2=xo_val,
                        model=graph_drp,
                        loss_fn=torch.nn.MSELoss(),
                        optimiser=torch.optim.Adam(graph_drp.parameters(), lr=1e-4),
                        epochs=epochs, 
                        early_stopping_dict={'patience': 300, 'delta': 0.0},
                        ms_path=f'{model_dir}_graph'
                        )
                    y_pred = inference.predict(xd_test, xo_test, graph_drp)

                
                np.savetxt(f'{save_path}{split_type}_seed{model_seed}',
                        np.array(y_pred, np.float32))
                metrics['mse'].append(mean_squared_error(y_test, y_pred))
                metrics['spear'].append(spearmanr(y_test, y_pred)[0])

            metrics = pd.DataFrame(metrics)
            metrics.to_csv(f'{save_path}mets_ttsplit_{seed}')

if __name__ == '__main__':
    main()