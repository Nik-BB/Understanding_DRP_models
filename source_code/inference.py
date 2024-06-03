
import numpy as np 
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from collections import defaultdict
import pandas as pd
import torch


device = "cuda" if torch.cuda.is_available() else "cpu" 

def predict(dl1, dl2, model):
    '''Find regresstion predictions of a pytorch model
    
    where the model takes two dataloaderes as inputs
    
    ---inputs---
    dl1: DataLoader
    dl2: DataLoader
    model: torch model that takes two inputs
    
    ---returns---
    array: np array of predictions of the model, shape = len(dl1)
    
    '''
    model.eval()
    preds = []
    with torch.no_grad():
        for xo, xd in zip(dl1, dl2):
            xo, xd = xo.to(device), xd.to(device)
            pred = model(xo, xd)
            pred = pred.cpu().detach().numpy()
            pred = pred.reshape(len(pred))
            preds.extend(pred)
    return np.array(preds)

def one_imp_predict(dl, model):
    ''' predictions of a pytorch model with one input Data loader'''

    model.eval()
    preds = []
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            pred = model(x)
            pred = pred.cpu().detach().numpy()
            pred = pred.reshape(len(pred))
            preds.extend(pred)
    return np.array(preds)




def find_metrics(pred, true, m_list=['MSE', 'R2', 'Pear', 'Spear']):
    metric_dict = {}
    for metric in m_list:
        if metric == 'Spear':
            s = spearmanr(true, pred)[0]
            metric_dict['Spear'] = s
        if metric == 'Pear':
            p = pearsonr(true, pred)[0]
            metric_dict['Pear'] = p
        if metric == 'MSE':
            mse = mean_squared_error(true, pred)
            metric_dict['MSE'] = mse
        if metric == 'R2':
            r2 = r2_score(true, pred)
            metric_dict['R2'] = r2
    return metric_dict

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

def find_stratifed_mets(pred, true, by='cl', 
                        m_list=['Pear', 'MSE', 'R2', 'Spear']):
    '''stratify metrics by cl or drug''' 

    cl_to_idx = stratify(true.index, by=by)
    mets = [find_metrics(pred[idxs], true[idxs]) 
            for _, idxs in cl_to_idx.items()]
    mets = pd.DataFrame(mets, index=cl_to_idx.keys())
    return mets