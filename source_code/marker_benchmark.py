import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
from models import marker
from torch.utils.data import DataLoader
import data_loading, data_processing, training, inference

model_type = 'marker'
split_type = 'mixed'
epochs = int(sys.argv[1])
batch_size = 128

hpc = False if os.getcwd()[0] == 'C' else True
if hpc:
    omic_dir_path = '../../GDSC/downloaded_data' #hpc    
else:
    omic_dir_path = '../Downloaded_data'

gdsc2_target_path = 'data/GDSC2_Wed Aug GDSC2_30_15_49_31_2023.csv'
pubchem_ids_path = 'data/drugs_gdsc_to_pubID.csv'

if hpc:
    os.chdir('..')
print(f'using hpc: {hpc}')


rna, ic50, drugs_to_smiles = data_loading.load_omics_drugs_target(
    omic_dir_path, gdsc2_target_path, pubchem_ids_path, save=False)
drugs_with_smiles = drugs_to_smiles.index


one_hot_drugs = np.zeros((len(drugs_with_smiles), len(drugs_with_smiles)))
np.fill_diagonal(one_hot_drugs, 1)
one_hot_drugs = pd.DataFrame(
    one_hot_drugs, index=drugs_with_smiles, columns=drugs_with_smiles, 
    dtype=np.float32)

one_hot_cls = np.zeros((len(rna.index), len(rna.index)))
np.fill_diagonal(one_hot_cls, 1)
one_hot_cls = pd.DataFrame(one_hot_cls, index=rna.index, dtype=np.float32)

x_omic, x_drug, y = data_processing.create_all_drugs(one_hot_cls, one_hot_drugs, ic50)

x_drug = pd.DataFrame(
    [enc.values for enc in x_drug['drug_encoding']], index=x_drug.index)

 # train and eval model for 3 train test splits and 3 model seeds
for seed in range(1, 4):
    pairs_path = f'data/train_test_pairs/{split_type}/'

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

    num_features = len(xd_train.columns) + len(xo_train.columns)

    x_train = DataLoader(pd.concat((xo_train, xd_train), axis=1).values,
                         batch_size=batch_size)
    x_val = DataLoader(pd.concat((xo_val, xd_val), axis=1).values,
                       batch_size=batch_size)
    x_test = DataLoader(pd.concat((xo_test, xd_test), axis=1).values, 
                        batch_size=batch_size)
    y_train = DataLoader(np.expand_dims(y_train, 1), 
                        batch_size=batch_size)
    y_val = DataLoader(np.expand_dims(y_val, 1), 
                    batch_size=batch_size)
    y_test = np.expand_dims(y_test, 1)

    #model train
    save_path = f'results/tt_split_{seed}/predictions/{model_type}/'
    model_path =  f'saved_models/{model_type}'
    
    #model traning for 3 seeds
    metrics = {'mse': [], 'spear': []}
    for model_seed in range(3):
        torch.manual_seed(model_seed)
        model_dir = f'{model_path}/{split_type}/'

        marker_model = marker.MLP(inp_dim=num_features, hid_dim=4096, 
                            num_hid_layers=2)

        hist = training.tl_multi_dls(
            train_dls=[x_train],
            y_train=y_train,
            val_dls=[x_val],
            y_val=y_val,
            model=marker_model,
            loss_fn=torch.nn.MSELoss(), 
            optimiser=torch.optim.RMSprop(marker_model.parameters(), lr=1e-4),
            epochs=epochs,
            early_stopping_dict={'patience': 300, 'delta': 0.0},
            ms_path=f'{model_dir}_marker'
        )
        #with open('results/marker_hist_temp', 'wb') as handel:
            #pickle.dump(hist, handel)

        y_pred = inference.one_imp_predict(x_test, model=marker_model)
        np.savetxt(f'{save_path}{split_type}_seed{model_seed}', 
                   np.array(y_pred, np.float32))