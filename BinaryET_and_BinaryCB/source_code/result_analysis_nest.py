import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
sys.path.insert(0, 'source_code') 
import inference
from models import mean_model
y = pd.read_csv('results/all_y.csv', index_col=0)['0']


cwd = os.getcwd()
#working locally
if cwd[0].upper() == 'C':
    tt_dir = r'C:/Users/Nik/Documents/PhD_code/year3_code/Binary_ic50/Binary_ic50_lit/data/'

else:
    tt_dir = f'/data/home/wpw035/DRP-lit-comp-aberrations/Binary_ic50_lit/data/'

def stratify(pairs, by='cl'):
    '''maps cell line (cl) or drug to the index of all pairs that include that cl or drug 
    --Inputs--
    pairs: list like
    drug cell line pairs in the format cl::drug
    by: str 'cl' or 'drug'
    controls if cell lines or drugs are mapped to pairs
    '''
    if by not in ['cl', 'drug']:
        raise Exception('by needs to be cl or drug')
    mapping = defaultdict(list)
    #cl beofre the '::' drug after 
    delim = 0 if by == 'cl' else 1
    keys = [pair.split('::')[delim] for pair in pairs]
    for i, key in enumerate(keys):
        mapping[key].append(i)
    
    return mapping

def load_mm_strat(split_type):
    mm_strat = []
    for split in range(1, 4):
        path = f'results/mm_mets/strat_{split_type}_split{split}'
        mm_strat.append(pd.read_csv(path, index_col=0).mean())
    return pd.concat(mm_strat, axis=1).T


#mm_mets = pd.read_csv('results/mm_mets/mm_c_blind', index_col=0)
#mm_dblidn_mets = pd.read_csv('results/mm_mets/mm_d_blind', index_col=0)
#mm_strat_dblind = load_mm_strat('d_blind')
#mm_strat_cblind = load_mm_strat('c_blind')


def read_predictions(seed, model='tta', tt_split=1, split_type='c_blind'):
    if model == 'mm':
        path = f'results/multi_splits/preds/{model}_{split_type}split{tt_split}'
    else:
        path = f'results/multi_splits/preds/{model}{split_type}split{tt_split}_ms{seed}'
        
    pred = pd.read_csv(path, header=None)[0]
    tt_path = f'{tt_dir}train_test_pairs/{split_type}/seed_{tt_split}_test'
    test_pairs = pd.read_csv(tt_path, header=None)[0]
    y_test = y.loc[test_pairs]
    return pred, y_test

def find_mets(seed, model='tta', tt_split=1, split_type='c_blind', 
              strat=False, return_strat=False, remove_single_class=False):
    pred, y_test = read_predictions(seed, model=model, tt_split=tt_split, 
                                    split_type=split_type)
    if strat:
        mets = inference.find_stratifed_mets(pred, y_test, by=strat, remove_single_class=remove_single_class)
        if return_strat:
            return mets
        mets = mets.mean()
    else:
        mets = inference.find_metrics(pred, y_test)
    return mets

def add_bench_mets(df, tt_split, split_type, strat=False):
    if split_type == 'c_blind':
        if strat:
            df.loc['mm'] = mm_strat_cblind.iloc[tt_split - 1]
        else:
            df.loc['mm'] = mm_mets.iloc[tt_split - 1]
    if split_type == 'd_blind':
        if strat:
            df.loc['mm'] = mm_strat_dblind.iloc[tt_split - 1]
        else:   
            df.loc['mm'] = mm_dblidn_mets.iloc[tt_split - 1]
    if split_type == 'mixed':
        pass
    return df    

def find_ave_sd(tt_split, split_type='c_blind', strat=False,
                remove_single_class=False,
                models=['tcnn', 'graphdrp', 'tta', 'tta-db']):
    
    if strat not in ['cl', 'drug', False]:
        raise Exception('strat needs to be cl or drug')
    
    model_mets = {}
    for model in models:
        if model == 'mm':
            mets = pd.DataFrame(
            [find_mets(None, model, tt_split, split_type, strat, remove_single_class=remove_single_class)])
        else:
            mets = pd.DataFrame(
                [find_mets(seed, model, tt_split, split_type, strat, remove_single_class=remove_single_class) 
                 for seed in range(3)])
        model_mets[model] = mets

    ave_mets = pd.DataFrame([model_mets[model].mean() for model in models], 
                            index=models)
    #ave_mets = add_bench_mets(ave_mets, tt_split, split_type, strat)

    sd_mets = pd.DataFrame([model_mets[model].std() for model in models], 
                           index=models)
    return ave_mets, sd_mets 

def format_results(mets, sds):
    '''formats results table to be in the form metric +/- sd
    assumes 
    '''
    #mets_errs = {'auc':[], 'aupr': [], 'bce': [], 'acc': [], 'f1': []}
    mets_errs = defaultdict(list)
    for met in sds.columns:
        for model in sds.index:
            sd = '%.1g' % sds.loc[model, met]
            val = mets.loc[model, met]
            if np.isnan(val):
                pass 
            elif '.' in sd:
                num_dps = len(sd.split('.')[1]) 
                val = str(np.round(val, num_dps))
                #add trailing zero, after ., if needed
                if len(val.split('.')[1]) + 1 == num_dps:
                    val = val + '0'
            else:
                num_sf = len(sd)
                val = round(val, 1 - num_sf)
            mets_errs[met].append(f'{str(val)} ± {sd}')

    return pd.DataFrame(mets_errs, index=sds.index)

def round_string(val, dps):
    val = str(np.round(val, dps))
    num_trail_zeros = None
    if '.' in val:
        num_trail_zeros = dps - len(val.split('.')[1])
    if num_trail_zeros:
        val = val + '0' * num_trail_zeros
    return val

def create_results(tt_split, split_type='c_blind', strat=False, 
                   remove_single_class=False,
                   order=['tcnn', 'graphdrp','mm', 'tta', 'tta-db']):
    
    non_mms = [m for m in order if m != 'mm']
    mets, sd = find_ave_sd(tt_split, split_type, strat, remove_single_class, order)
    results_table = format_results(mets.loc[non_mms], sd.loc[non_mms])
    #results_table = add_bench_mets(results_table, tt_split, split_type, strat=strat)
    if 'mm' in order:
        vals = []
        for val in mets.loc['mm']:
            dps = 3
            val = str(np.round(val, dps))
            #add traning zeros after dp
            num_trail_zeros = dps - len(val.split('.')[1])
            if num_trail_zeros and '.' in val:
                val = val + '0' * num_trail_zeros
            vals.append(val)
        results_table.loc['mm'] = vals #(results_table.loc['mm'].astype(np.float32)).round(3)
    results_table = results_table.loc[order]
    return results_table


def find_drug_mappings(true_series):
    drug_to_idx = inference.stratify(true_series.index, by='drug')
    drug_to_cl = {}
    for d in drug_to_idx.keys():
        cells = [pair.split('::')[0] 
                 for pair in true_series.index[drug_to_idx[d]]]
        drug_to_cl[d] = cells
    return drug_to_idx, drug_to_cl


def find_overlapping_cls(drug_to_cl_dict, drugs_to_ignore=['Tretinoin']):
    '''finds cls in all drugs ignoring drugs_to_ignore'''
    drugs = list(drug_to_cl_dict.keys())
    for d in drugs:
        num_cls = len(drug_to_cl_dict[d]) 
        if num_cls < 500 and d not in drugs_to_ignore:
            raise Exception(f'{d} only has {num_cls} cls ')
    overlapping = set(drug_to_cl_dict[drugs[0]])
    for d in drugs[1 : ]:
        if d in drugs_to_ignore:
            continue
        cells = drug_to_cl_dict[d]
        overlapping = overlapping.intersection(cells)
        
    return overlapping

def find_overlaping_d_cl_pairs(truth_vals, drugs_to_ignore):
    drug_to_idx, drug_to_cl = find_drug_mappings(truth_vals)
    overlapping_cls = find_overlapping_cls(drug_to_cl, drugs_to_ignore)

    #find overlapping pairs from overlapping cls
    overlapping_d_cls_pairs = []
    for d in drug_to_cl.keys():
        if d in drugs_to_ignore:
            continue
        overlapping_d_cls_pairs.extend(
            list(pd.Index(overlapping_cls) + '::' + d))
    return overlapping_d_cls_pairs


def compare_metrics(truth_vals, pred, mm_pred, drugs_to_ignore, strat=False):
    overlapping_d_cls_pairs = find_overlaping_d_cl_pairs(
        truth_vals, drugs_to_ignore)
    pairs_idx = truth_vals.index
    pred.index = pairs_idx

    true_overlapping = truth_vals.loc[overlapping_d_cls_pairs]
    pred_overlapping = pred.loc[overlapping_d_cls_pairs]
    mm_overlapping = pd.Series(
        mm_pred, index=pairs_idx).loc[overlapping_d_cls_pairs]

    if strat:
        lit_mets = inference.find_stratifed_mets(
            pred_overlapping, true_overlapping, by=strat).mean()
        mm_mets = inference.find_stratifed_mets(
            mm_overlapping, true_overlapping, by=strat).mean()
    else:
        lit_mets = inference.find_metrics(pred_overlapping, true_overlapping)
        mm_mets = inference.find_metrics(mm_overlapping, true_overlapping)
    
    results = pd.DataFrame({'lit_mets': lit_mets, 'mm_mets': mm_mets})
    return results.T

def find_mm_preds(tt_seed):
    '''find the predicitons for the mean model for a given train test split''' 
    split_type = 'd_blind'
    pairs_path = f'data/train_test_pairs/{split_type}/'
    train_pairs = pd.read_csv(
        f'{pairs_path}seed_{tt_seed}_train', header=None)[0].values
    val_pairs = pd.read_csv(
        f'{pairs_path}seed_{tt_seed}_val', header=None)[0].values
    test_pairs = pd.read_csv(
        f'{pairs_path}seed_{tt_seed}_test', header=None)[0].values

    y_train, y_test = y.loc[train_pairs], y.loc[test_pairs]
    cells = np.unique([idx.split('::')[0] for idx in train_pairs])


    mm = mean_model.MMDblind(y_train=y_train, cells=cells)
    mm_pred = mm.predict(y_test.index)

    return mm_pred

def find_same_cl_results(tt_split, strat=False):
    '''find d blind results with subseted cls such that all drugs have same 
    cls in the test set'''

    mm_pred = find_mm_preds(tt_split)
    same_cls = defaultdict(list)
    m_names = ['tcnn', 'graphdrp', 'tta']
    if tt_split == 3:
        drugs_to_ignore = ['CCT-018159', 'CHIR-99021']
    elif tt_split == 2:
        drugs_to_ignore = ['Tretinoin']
    elif tt_split == 1:
        drugs_to_ignore = ['BX795']
        
    for m in m_names:
        for seed in range(3):
            pred, true = read_predictions(seed, model=m, tt_split=tt_split, split_type='d_blind')
            df = compare_metrics(
                truth_vals=true, pred=pred, mm_pred=mm_pred, 
                drugs_to_ignore=drugs_to_ignore, 
                strat=strat)
            same_cls[m].append(df.loc['lit_mets'])
            mm_mets = df.loc['mm_mets']


    mets = pd.DataFrame([pd.DataFrame(same_cls[m]).mean() for m in m_names],
                         index=m_names)
    sds = pd.DataFrame([pd.DataFrame(same_cls[m]).std() for m in m_names],
                        index=m_names)

    same_cls = format_results(mets, sds)
    same_cls.loc['mm'] = mm_mets

    return same_cls
