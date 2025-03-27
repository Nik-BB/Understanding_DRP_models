'''run model for mutiple tt splits and seeds '''

import os
import sys
import pickle
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
from models import drp_tf, mlp
import bpe_tokenisation
#import wandb
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = 'cpu'
owd = os.getcwd()
log = False
hpc = False
# if owd[0].upper() == 'C':
#     hpc=False
#     log = False
if hpc:
    os.chdir(owd)
    os.chdir('..')
    log = True

supported_models = ['db_tf', 'db_marker']
if __name__ == '__main__':
    models_to_run = str(sys.argv[1]) # e.g. db_tf
    split_type = str(sys.argv[2]) #e.g. c_blind
    epochs = int(sys.argv[3])    

else:
    models_to_run = 'db_tf'
    split_type = 'c_blind'
    epochs = 100

if models_to_run == 'multiple':
    models_to_run = supported_models
else:
    models_to_run = [models_to_run]

def process_model_specific_data(drug_data_type, db_architecture, binary_ic50=True):
    '''reutrn drug cell line and y data for all drugs for a given model'''

    ds = create_dataset.ClDrugIntegrator(omics=['rna'])
    if binary_ic50 and ds.full_data:
        #binarise ic50 vals
        drug_to_max_con, multi_con_drugs = binary_truth.find_drug_to_max_con_gdsc2()
        #drop drugs with mulitple concentrations (4)
        drugs_to_drop = [d for d in multi_con_drugs if d in ds.y]
        ds.y = ds.y.drop(columns=drugs_to_drop)
        ds.drugs_to_smiles = ds.drugs_to_smiles.loc[ds.y.columns]
        ds.y = binary_truth.binarise_ic50(ds.y, drug_to_max_con)

    marker_drugs = create_dataset.one_hot_encode(ds.drugs_to_smiles.index)
    marker_drugs.index = marker_drugs.columns
    marker_drugs = marker_drugs.astype(np.float32)
    # x_all_omic, x_hot_drug, y_list = utils.create_all_drugs(
    #     ds.rna_omic, marker_drugs, ds.y)


    #can remove this and just use smiles encoding above
    if drug_data_type == 'smiles' and db_architecture == 'tf':
        x_omic, x_drug, y = utils.create_all_drugs(ds.rna_omic, marker_drugs, ds.y)

        return x_omic, x_drug, y, ds 

    elif drug_data_type == 'bpe_smiles' and db_architecture == 'tf':
        #not keeping atten mask here
        encoded_smiles = { 
            drug: bpe_tokenisation.smile_encoder(ds.drugs_to_smiles.loc[drug]['0'])[0] #[1] gives atten mask
            for drug in ds.y.columns}
        encoded_smiles = pd.DataFrame(encoded_smiles)
        x_omic, x_drug, y = utils.create_all_drugs(ds.rna_omic, encoded_smiles, ds.y)       

        return x_omic, x_drug, y, ds 
    
    elif drug_data_type == 'marker':
        marker_drugs = marker_drugs.astype(np.float32)
        x_omic, x_drug, y = utils.create_all_drugs(ds.rna_omic, marker_drugs, ds.y)
        return x_omic, x_drug, y, ds 

    

def run_model(model_type, config, save_name, save_dir='multi_splits'):
    '''runs one model defiend by model_type '''
    print(os.getcwd())
    p_path = f'results/{save_dir}/mets/{save_name}_'
    m_dir = f'results/{save_dir}/mets'
    p_dir = f'results/{save_dir}/preds'
    th_dir = f'results/{save_dir}/train_hist'
    if not os.path.exists(m_dir):
        os.makedirs(m_dir)
    if not os.path.exists(p_dir):
        os.makedirs(p_dir)
    if not os.path.exists(th_dir):
        os.makedirs(th_dir)

    with open(f'{p_path}config.pickle', 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    bs = config['batch_size']
    hps = config['hps']
    epochs = config['max_epochs']
    opt_type = config['opt_type']
    if 'xav_init' in config.keys():
        xav_init = config['xav_init']
    else:
        xav_init=False

    #if model_type == 'omlp_db_smileBpeTf' or model_type == 'tta_rec':
    if model_type == 'db_tf':
        drug_data_type = 'smiles'
        db_architecture = 'tf'
        xdict = True
    elif model_type == 'db_marker':
        drug_data_type = 'marker'
        db_architecture = None

    x_omic, x_drug, y, ds = process_model_specific_data(drug_data_type, db_architecture)

    if drug_data_type != 'marker': del x_drug
    
    inp_dim = x_omic.shape[1]

    for split_seed in range(1, 4): 

        if drug_data_type == 'smiles' and db_architecture == 'tf':
            pairs_to_smiles = pd.Series(
                {pair: ds.drugs_to_smiles.loc[pair.split('::')[1]]['0'] for pair in y.index})
            pairs_to_smiles = pairs_to_smiles.loc[y.index]
            tok_version = 'seyonec/ChemBERTa_zinc250k_v2_40k'
            tokenizer = RobertaTokenizer.from_pretrained(tok_version)
            train_dls, val_dls, test_dls, y_test = create_dataset.load_data_smile(
                x_omic, pairs_to_smiles, y, tokenizer, split_seed, split_type=split_type, 
                expand_xo_dims=False, batch_size=bs)
            
        elif drug_data_type == 'marker':
            train_dls, val_dls, test_dls, y_test = create_dataset.load_data_omic_df(
                x_omic, x_drug, y, split_seed, mm_data=False, split_type=split_type, 
                expand_xo_dims=False, batch_size=bs)
            
        for model_seed in range(3): 
            torch.manual_seed(model_seed)

            lr_scheduler=None
            def get_lrs():
                n_train_steps = np.floor(len(train_dls[0]) * epochs) 
                n_warmups = len(train_dls[0]) * 3
                if config['lrs'] == 'cosine_schedule_with_warmup':
                    lr_scheduler = get_cosine_schedule_with_warmup(
                        optimizer=opt, num_warmup_steps=n_warmups, 
                        num_training_steps=n_train_steps)
                return lr_scheduler

            if model_type == 'db_tf':
                if drug_data_type == 'smiles':
                    vocab_size = 880 
                elif drug_data_type == 'bpe_smiles':
                    vocab_size = 2584
                model = drp_tf.OmicMLPDrugTF(hps, inp_dim, vocab_size, att_mask=False, xdict=xdict, pos_enc=True)
                opt = opt_type(model.parameters(), lr=hps['lr'])
            elif model_type == 'db_marker':
                model = mlp.OmicMlpMarker(hps, inp_dim)
                opt = opt_type(model.parameters(), lr=hps['lr'])

            if config['lrs']:
                lr_scheduler = get_lrs()

            if xav_init: 

                for name, param in model.named_parameters():
                    if 'weight' in name and param.data.dim() == 2:
                        torch.nn.init.xavier_uniform_(param)

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
            path_suffix = f'{split_type}split{split_seed}_ms{model_seed}'
            pd.DataFrame(metrics, index=[f'{save_name}']).to_csv(f'{p_path}mets_{path_suffix}.csv')
            np.savetxt(f'results/{save_dir}/preds/{save_name}{path_suffix}',
                np.array(y_pred, np.float32))
            pd.DataFrame(hist).to_csv(f'results/{save_dir}/train_hist/{save_name}{path_suffix}')

        


#old hps 
'''
config = {
    'split_type': split_type,
    'hps': {'omic_n_nodes': [1024, 256, 64],
            'n_nodes1': 1024, 
            'n_nodes2': 512,
            'lr': 1e-5,
        'drop_out': 0.0,
        'drug_emb_dim': 128,
        'n_tf_layers': 8,
        'tf_atten_heads': 8,
        'tf_ff_dim': 2048}, #can try 512 hp for tta.
    'max_epochs': epochs,
    'batch_size': 128, 
    'lrs': 'cosine_schedule_with_warmup',
    #'model_seed': 0,
    'opt_type': torch.optim.AdamW,
    'db_model_version': 'seyonec/ChemBERTa_zinc250k_v2_40k'
    }
'''
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
        'tf_ff_dim': 2048}, #can try 512 hp for tta.
    'max_epochs': epochs,
    'batch_size': 128, 
    'lrs': 'cosine_schedule_with_warmup',
    #'model_seed': 0,
    'opt_type': torch.optim.AdamW,
    #'db_model_version': 'seyonec/ChemBERTa-zinc-base-v1',
    'm_type': None,
    'xav_init': False
    }

def main():  
    print(models_to_run)
    for model_type in models_to_run:
        if model_type not in supported_models:
            raise Exception(f'{model_type} not a supported model')
        config['m_type'] = model_type
        save_name = f'{model_type}_opt'
        if log:
            wandb.init(project=f'{model_type}')
        run_model(model_type, config, save_name=save_name, save_dir='multi_splits')

if __name__ == '__main__':
    main()