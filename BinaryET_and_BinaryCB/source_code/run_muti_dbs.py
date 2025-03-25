'''run mlp oimc encoder with diffrent drug branches'''


import os
import sys
import pickle
import wandb
import torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import ParameterGrid
import torch
import create_dataset
import binary_truth
import db_prep
import training
import inference
import utils
from models import mlp
import bpe_tokenisation

supported_models = ['omlp_db_chemro', 'db_marker']

device = "cuda" if torch.cuda.is_available() else "cpu"

owd = os.getcwd()
log = True
hpc = False
if owd[0].upper() == 'C':
    hpc=False
if hpc:
    os.chdir(owd)
    os.chdir('..')
    #log = True



def process_model_specific_data(drug_data_type, db_architecture, binary_ic50=True):
    '''reutrn drug cell line and y data for all drugs for a given model'''
    ds = create_dataset.ClDrugIntegrator(omics=['rna'])

    if binary_ic50:
        #binarise ic50 vals
        drug_to_max_con, multi_con_drugs = binary_truth.find_drug_to_max_con_gdsc2()
        #drop drugs with mulitple concentrations (4)
        drugs_to_drop = [d for d in multi_con_drugs if d in ds.y]
        ds.y = ds.y.drop(columns=drugs_to_drop)
        ds.drugs_to_smiles = ds.drugs_to_smiles.loc[ds.y.columns]
        ds.y = binary_truth.binarise_ic50(ds.y, drug_to_max_con)

    marker_drugs = create_dataset.one_hot_encode(ds.drugs_to_smiles.index)
    marker_drugs.index = marker_drugs.columns
    # x_all_omic, x_hot_drug, y_list = utils.create_all_drugs(
    #     ds.rna_omic, marker_drugs, ds.y)

    if drug_data_type == 'smiles' and db_architecture == 'cnn':
        dts_pad = db_prep.pad_smiles(ds.drugs_to_smiles['0'])
        drug_to_hot_smile = db_prep.hot_encode_smiles(dts_pad)
        x_omic, x_drug, y = utils.create_all_drugs(ds.rna_omic, drug_to_hot_smile, ds.y)

        return x_omic, x_drug, y, ds

    elif drug_data_type == 'graph' and db_architecture == 'gnn':
        x_omic, _, y = utils.create_all_drugs(ds.rna_omic, marker_drugs, ds.y)
        x_drug = db_prep.create_graphs(ds.drugs_to_smiles.index, y)
        

        return x_omic, x_drug, y, ds

    elif drug_data_type == 'smiles' and db_architecture == 'tf' or db_architecture == 'ro':
        x_omic, x_drug, y = utils.create_all_drugs(ds.rna_omic, marker_drugs, ds.y)

        return x_omic, x_drug, y, ds 

    elif drug_data_type == 'bpe_smiles' and db_architecture == 'tf':
        #not keeping atten mask here
        encoded_smiles = { 
            drug: bpe_tokenisation.smile_encoder(ds.drugs_to_smiles.loc[drug]['0'])[0] #[1] gives atten mask
            for drug in ds.y.columns}
        encoded_smiles = pd.Series(encoded_smiles)
        x_omic, x_drug, y = utils.create_all_drugs(ds.rna_omic, encoded_smiles, ds.y)       

        return x_omic, x_drug, y, ds 
    
        
    elif drug_data_type == 'marker':
        marker_drugs = marker_drugs.astype(np.float32)
        x_omic, x_drug, y = utils.create_all_drugs(ds.rna_omic, marker_drugs, ds.y)
        return x_omic, x_drug, y, ds 


def run_model(model_type, config, save_name, save_dir=None):
    '''runs one model defiend by model_type '''
    if save_dir:
        p_path = f'results/mets/{save_dir}/{save_name}_'
    else:
        p_path = f'results/mets/{save_name}_'
    with open(f'{p_path}config.pickle', 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    bs = config['batch_size']
    hps = config['hps']
    model_seed = config['model_seed']
    epochs = config['max_epochs']
    opt_type = config['opt_type']
    if 'xav_init' in config.keys():
        xav_init = config['xav_init']
    else:
        xav_init=False
    split_seed = 1
    print(f'Note split seed is {split_seed}')

    if model_type == 'omlp_db_chemro':
        drug_data_type = 'smiles'
        db_architecture = 'ro'

    elif model_type == 'db_marker':
        drug_data_type = 'marker'
        db_architecture = None


    x_omic, x_drug, y, ds = process_model_specific_data(drug_data_type, db_architecture)
    inp_dim = x_omic.shape[1]


    if drug_data_type == 'smiles' and db_architecture == 'tf' or db_architecture == 'ro':
        pairs_to_smiles = pd.Series(
            {pair: ds.drugs_to_smiles.loc[pair.split('::')[1]]['0'] for pair in y.index})
        pairs_to_smiles = pairs_to_smiles.loc[y.index]
        db_model_version = config['db_model_version']
        tok_version = db_model_version
        tokenizer = RobertaTokenizer.from_pretrained(tok_version)
        train_dls, val_dls, test_dls, y_test = create_dataset.load_data_smile(
            x_omic, pairs_to_smiles, y, tokenizer, split_seed, split_type='c_blind', 
            expand_xo_dims=False, batch_size=bs)

    elif drug_data_type == 'bpe_smiles' and db_architecture == 'tf':
        train_dls, val_dls, test_dls, y_test = create_dataset.load_data_omic_df(
            x_omic, x_drug, y, split_seed, split_type=split_type, expand_xo_dims=False, batch_size=bs)
        
    elif drug_data_type == 'marker':
        train_dls, val_dls, test_dls, y_test = create_dataset.load_data_omic_df(
            x_omic, x_drug, y, split_seed, mm_data=False, split_type=split_type, 
            expand_xo_dims=False, batch_size=bs)
            
    torch.manual_seed(model_seed)

    print(db_architecture)
    lr_scheduler=None
    def get_lrs():
        n_train_steps = np.floor(len(train_dls[0]) * epochs) #typcally train for < 100 epochs before overfitting so myb change
        n_warmups = len(train_dls[0]) * 3
        if config['lrs'] == 'cosine_schedule_with_warmup':
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=opt, num_warmup_steps=n_warmups, 
                num_training_steps=n_train_steps)
        return lr_scheduler

    if db_architecture == 'ro':

        model = mlp.OmicMlpRo(config['hps'], inp_dim, db_model_version=db_model_version)
        model = model.to(device)  

        opt = opt_type(model.parameters(), lr=hps['lr'])
        if config['lrs']:
            lr_scheduler = get_lrs()
        hist = training.tl_multi_dls(
            train_dls=train_dls[: 2],
            y_train=train_dls[-1],
            val_dls=val_dls[: 2],
            y_val=val_dls[-1],
            model=model,
            loss_fn=torch.nn.BCELoss(),
            optimiser=opt,
            lr_scheduler_tran=lr_scheduler,
            epochs=epochs,
            early_stopping_dict={'patience': 50, 'delta': 0.0},
            ms_path=f'{model_type}_{REP}_temp'
            )
        
    elif model_type == 'db_marker':
        model = mlp.OmicMlpMarker(hps, inp_dim)
        opt = opt_type(model.parameters(), lr=hps['lr'])
        if config['lrs']:
            lr_scheduler = get_lrs()

        hist = training.tl_multi_dls(
            train_dls=train_dls[: 2],
            y_train=train_dls[-1],
            val_dls=val_dls[: 2],
            y_val=val_dls[-1],
            model=model,
            loss_fn=torch.nn.BCELoss(),
            optimiser=opt,
            lr_scheduler_tran=lr_scheduler,
            epochs=epochs,
            early_stopping_dict={'patience': 50, 'delta': 0.0},
            ms_path=f'{model_type}_temp'
            )
     
    y_pred = inference.predict(test_dls[0], test_dls[1], model)
    y_val_pred = inference.predict(val_dls[0], val_dls[1], model)


    y_val = []
    for yv in val_dls[-1]:
        yv = yv.cpu().detach().numpy()
        yv = yv.reshape(len(yv))
        y_val.extend(yv)
    y_val = np.array(y_val)
        
    print('min val loss')
    print(min(hist['val_loss']), np.argmin(hist['val_loss']))
    met_names = ['auc', 'aupr']
    metrics = inference.find_metrics(y_pred, y_test, m_list=met_names)
    metrics_val = inference.find_metrics(y_val_pred, y_val, m_list=met_names)
    metrics['val_auc'] = metrics_val['auc']
    metrics['val_aupr'] = metrics_val['aupr']
    metrics['min_val'] = min(hist['val_loss'])
    metrics['argmin'] = np.argmin(hist['val_loss'])
    pd.DataFrame(metrics, index=[f'{save_name}']).to_csv(f'{p_path}mets.csv')
    np.savetxt(f'results/preds/{save_name}',
        np.array(y_pred, np.float32))
    pd.DataFrame(hist).to_csv(f'results/train_hist/{save_name}')

    return metrics


if hpc:
    REP = int(sys.argv[1])
else:
    REP = 0 
epochs = 1
split_type = 'c_blind'
model_type = 'omlp_db_chemro'
hp_opt=False

config = {
    'split_type': split_type,
    'hps': {'omic_n_nodes': [1024, 256, 64],
            'n_nodes1': 1024, 
            'n_nodes2': 512,
            'lr': 5e-5,
        'drop_out': 0.0,
        'drug_emb_dim': 128,
        'n_tf_layers': 8,
        'tf_atten_heads': 8,
        'tf_ff_dim': 2048
        },
    'max_epochs': epochs,
    'batch_size': 128, 
    'lrs': 'cosine_schedule_with_warmup',
    'model_seed': REP,
    'opt_type': torch.optim.AdamW,
    'db_model_version': 'seyonec/ChemBERTa-zinc-base-v1', #can be interchanged with model versions below to run different chemberta models 
    'm_type': model_type,
    'xav_init': False
    }


#db_vs = ['DeepChem/ChemBERTa-5M-MLM', 'DeepChem/ChemBERTa-10M-MLM', 'DeepChem/ChemBERTa-77M-MLM']
#save_names = ['5M-MLM', '10M-MLM', '77M-MLM']
#db_vs = ['seyonec/ChemBERTa-zinc-base-v1', 'seyonec/ChemBERTa-zinc250k-v1', 'seyonec/ChemBERTa_zinc250k_v2_40k']
#save_names = ['base-v1', 'zinc250k-v1', 'zinc40k-v1']

def main():
    run_model(model_type, config, f'robase-v1_r{REP}', save_dir='chemro') 


if __name__ == '__main__':
    main()



