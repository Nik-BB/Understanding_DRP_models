import numpy as np
import pandas as pd


def create_all_drugs(x, xd, y):
    '''Create data for all drug and cell line pairs, for use in models.
    
    With cell line data (x) that is not drug spesfic (i.e. the same for 
    all drugs) copies this data for each drug while removing missing values 
    that are contained in y as nan.
    The indexes in the dataframes created agree with each other. 
    E.g. the zeorth index of the dataframes corrisponds to the 
    drug cell line pair given by x.iloc[0], y.iloc[0].
    
    Inputs
    -------
    x: pd dataframe.
    Omic data (i.e. phospo) where the index is the cell lines
    and cols are features.
    
    xd: pd series.
    One hot encoded representation of the drugs.
    
    y: pd datafame.
    Target values (i.e. ic50 values) where the index is 
    the cell lines and cols are the drugs. 
    
    Returns
    -------
    x_final: pd dataframe.
    Omics data for all drugs and cell lines
    
    X_drug_final: pd dataframe.
    One hot enocding for all drugs and cell lines
    
    y_final: pd index
    Target values for all drugs and cell lines

    '''
    drug_inds = []
    x_dfs = []
    x_drug_dfs = []
    y_final = []
    
    x.astype(np.float32)
    if isinstance(xd, pd.Series):
        d_list = xd.index
    elif isinstance(xd, pd.DataFrame):
         d_list = xd.columns
    for i, d in enumerate(d_list):
        #find cell lines without missing truth values
        y_temp = y[d]
        nona_cells = y_temp.index[~np.isnan(y_temp)]
        #finds the index for the start / end of each drug
        ind_high = len(nona_cells) + i
        drug_inds.append((d, i, ind_high))
        i += len(nona_cells)

        #store vals of the cell lines with truth values
        x_pp = x.loc[nona_cells] 
        x_dfs.append(x_pp)
        if isinstance(xd, pd.Series): 
            X_drug = pd.DataFrame(
                {'drug_encoding' : [xd[d]] * len(x_pp)}, index=[d] * len(x_pp))
        elif isinstance(xd, pd.DataFrame):
            X_drug = pd.DataFrame(
                {'drug_encoding' : [xd[d].values] * len(x_pp)}, index=[d] * len(x_pp))
            
        x_drug_dfs.append(X_drug)
        y_final.append(y_temp.dropna())

    #combine values for all drugs  
    x_final = pd.concat(x_dfs, axis=0)
    X_drug_final = pd.concat(x_drug_dfs, axis=0)
    y_final = pd.concat(y_final, axis=0)
    
    #reformat indexs 
    cls_drugs_index = x_final.index + '::' + X_drug_final.index 
    x_final.index = cls_drugs_index
    X_drug_final.index = cls_drugs_index
    y_final.index = cls_drugs_index
    
    x_final.astype(np.float32)
    #X_drug_final.astype(np.float32)
    y_final.astype(np.float32)
    
    return x_final, X_drug_final, y_final
