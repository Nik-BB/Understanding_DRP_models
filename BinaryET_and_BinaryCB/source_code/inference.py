
import numpy as np 
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score, log_loss, roc_curve
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
            if isinstance(xd, dict):
                pass
            else:
                xd = xd.to(device)
            xo = xo.to(device)
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


def find_opt_cutoff(true, pred):
    '''find optmial cut off threshold using J statistic'''
    fpr, tpr, thresholds = roc_curve(true,pred)
    opt_idx = np.argmax(tpr - fpr)
    opt_tresh = thresholds[opt_idx]
    return opt_tresh



def find_metrics(pred, true, m_list=['auc', 'aupr', 'bce', 'acc', 'f1']):
    metric_dict = {}
    if 'acc' in m_list or 'f1' in m_list:
        opt_tresh = find_opt_cutoff(true, pred)
        pred_classes = np.where(pred >= opt_tresh, int(1), pred)
        pred_classes = np.where(pred < opt_tresh, int(0), pred_classes)

    for metric in m_list:
        if metric == 'aupr':
            aupr = average_precision_score(true, pred)
            metric_dict['aupr'] = aupr
        if metric == 'auc':
            auc = roc_auc_score(true, pred)
            metric_dict['auc'] = auc
        if metric == 'bce':
            metric_dict['bce'] = log_loss(true, pred)
        if metric == 'f1':
            f1 = f1_score(true, pred_classes)
            metric_dict['f1'] = f1
        if metric == 'acc':
            acc = accuracy_score(true, pred_classes)
            metric_dict['acc'] = acc
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
                        m_list=['auc', 'aupr', 'bce', 'acc', 'f1'],
                        remove_single_class=False):
    '''stratify metrics by cl or drug''' 

    cl_to_idx = stratify(true.index, by=by)
    #for binary case  removes drugs or cls that only have one class i.e. all 0's or 1's 
    if remove_single_class:
        mets = []
        keep_drugs = []
        for d, idxs in cl_to_idx.items():
            if (true[idxs] == 0).sum() > 0 and (true[idxs] == 1).sum() > 0:
                mets.append(find_metrics(pred[idxs], true[idxs])) 
                keep_drugs.append(d)
    else:
        mets = [find_metrics(pred[idxs], true[idxs]) 
                for _, idxs in cl_to_idx.items()]
        keep_drugs = cl_to_idx.keys()
    mets = pd.DataFrame(mets, index=keep_drugs)
    return mets