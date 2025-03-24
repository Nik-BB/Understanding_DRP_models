import os
import numpy as np
import pandas as pd
import data_loading 
import data_processing
from models import mean_model
from inference import find_metrics, find_stratifed_mets

split_type = 'c_blind'
example_data = True 
hpc = False if os.getcwd()[0] == 'C' else True
#genral data loading
# if hpc:
#     omic_dir_path = '../../GDSC/downloaded_data' #hpc
    
# else:
#     omic_dir_path = '../../Downloaded_data' #local
omic_dir_path = 'data'
gdsc2_target_path = 'data/GDSC2_Wed Aug GDSC2_30_15_49_31_2023.csv'
pubchem_ids_path = 'data/drugs_gdsc_to_pubID.csv'

if hpc:
    os.chdir('..')
print(f'using hpc: {hpc}')

if example_data:
    rna = pd.read_csv('data/example_data/xpr_sub.csv', index_col=0)
    ic50 = pd.read_csv('data/example_data/ic50_sub.csv', index_col=0)
    drugs_to_smiles = pd.read_csv('data/drugs_to_smiles_gdsc2.csv', index_col=0)['0']
    pairs_path = f'data/example_data/train_test_pairs/{split_type}/'

else:
    rna, ic50, drugs_to_smiles = data_loading.load_omics_drugs_target(
        omic_dir_path, gdsc2_target_path, pubchem_ids_path, save=False)
    pairs_path = f'data/train_test_pairs/{split_type}/'

print(rna.shape, ic50.shape, len(drugs_to_smiles))
drugs_with_smiles = drugs_to_smiles.index


one_hot_drugs = np.zeros((len(drugs_with_smiles), len(drugs_with_smiles)))
np.fill_diagonal(one_hot_drugs, 1)
one_hot_drugs = pd.DataFrame(
    one_hot_drugs, index=drugs_with_smiles, columns=drugs_with_smiles)

_, _, y = data_processing.create_all_drugs(rna, one_hot_drugs, ic50)

# 3 train test splits 
splits_mets = [] 
for seed in range(1, 4):
    train_pairs = pd.read_csv(
        f'{pairs_path}seed_{seed}_train', header=None)[0].values
    val_pairs = pd.read_csv(
        f'{pairs_path}seed_{seed}_val', header=None)[0].values
    test_pairs = pd.read_csv(
        f'{pairs_path}seed_{seed}_test', header=None)[0].values

    y_train, y_test = y.loc[train_pairs], y.loc[test_pairs]
    y_val =  y.loc[val_pairs]

    if split_type == 'c_blind':
        mm = mean_model.MeanModel(y_train=y_train, drugs=drugs_with_smiles)
        by = 'cl'
    if split_type == 'd_blind':
        mm = mean_model.MMDblind(y_train=y_train, cells=rna.index)
        by = 'drug'
    mm_pred = mm.predict(y_test.index)
    pd.Series(mm_pred).to_csv(f'results/mm_mets/preds_{split_type}_{seed}')
    splits_mets.append(find_metrics(mm_pred, y_test))
    stratifed_mets = find_stratifed_mets(mm_pred, y_test, by=by)
    stratifed_mets.to_csv(f'results/mm_mets/strat_{split_type}_split{seed}')

splits_mets = pd.DataFrame(splits_mets)
splits_mets.to_csv(f'results/mm_mets/mm_{split_type}')

    

