'''General data processing used for mutiple model '''

import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    
    xd: pd dataframe.
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
    for i, d in enumerate(xd.index):
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
        X_drug = pd.DataFrame(
            {'drug_encoding' : [xd[d]] * len(x_pp)}, index=[d] * len(x_pp))
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


def into_dls(x: list, batch_size=512):
    '''helper func to put DRP data into dataloaders
    for non gnn x[0], x[1] and x[2] give the omics, drug and target values
    respectively. for gnn x[0] gives graph data
    
    '''
    #checks 
    assert len(x[0]) == len(x[1])
    assert len(x[0]) == len(x[2])
    from torch_geometric.loader import DataLoader as DataLoaderGeo 
    from torch.utils.data import DataLoader
    import torch_geometric.data as tgd
    
    x[1] = DataLoader(x[1], batch_size=batch_size)
    x[2] = DataLoader(x[2], batch_size=batch_size)
    
    if isinstance(x[0], tgd.Batch):
        print('Graph drug data')
        x[0] = DataLoaderGeo(x[0], batch_size=batch_size)
    else:
        x[0] = DataLoader(x[0], batch_size=batch_size)
        
    return x


#train test split
def cblind_split(seed, _all_cls, _all_drugs, all_targets, train_size=0.8):
    '''Train test split for cancer blind testing

    Cancer blind testing means cell lines do not overlap between
    the train test and val sets.  
    '''
    #input cheks
    if type(_all_drugs) == type(_all_cls) != pd.Index:
        print('_all_drugs and _all_cls need to be PD indxes')


    train_cls, test_cls = train_test_split(_all_cls, train_size=train_size, 
                                            random_state=seed)

    assert len(set(train_cls).intersection(test_cls)) == 0

    frac_train_cl = len(train_cls) / len(_all_cls)
    frac_test_cl = len(test_cls) / len(_all_cls)

    print('Fraction of cls in sets, relative to all cls '\
            'before mising values are removed')          
    print(f'train fraction {frac_train_cl}, test fraction {frac_test_cl}')
    print('------')


    #add in the drugs to each cell line. 
    def create_cl_drug_pair(cells):
        all_pairs = []
        for drug in _all_drugs:
            pairs = cells + '::' + drug
            #only keep pair if there is a truth value for it
            for pair in pairs:
                if pair in all_targets:
                    all_pairs.append(pair)


        return np.array(all_pairs)

    train_pairs = create_cl_drug_pair(train_cls)
    test_pairs = create_cl_drug_pair(test_cls)


    train_pairs = sklearn.utils.shuffle(train_pairs, random_state=seed)
    test_pairs = sklearn.utils.shuffle(test_pairs, random_state=seed)
          

    assert len(set(train_pairs).intersection(test_pairs)) == 0

    num_all_examples = len(_all_cls) * len(_all_drugs)
    frac_train_pairs = len(train_pairs) / num_all_examples
    frac_test_pairs = len(test_pairs) / num_all_examples
          
    print('Fraction of cls in sets, relative to all cl drug pairs, after '\
          'mising values are removed')
    print(f'train fraction {frac_train_pairs}, test fraction '\
          f'{frac_test_pairs}')

    #checking split works as expected.  
 
#create mapping of cls to cl drug pairs for test train and val set. 
    def create_cl_to_pair_dict(cells):
        '''Maps a cell line to all cell line drug pairs with truth values

        '''
        dic = {}
        for cell in cells:
            dic[cell] = []
            for drug in _all_drugs:
                pair = cell + '::' + drug
                #filters out pairs without truth values
                if pair in all_targets:
                    dic[cell].append(pair)
        return dic

    train_cl_to_pair = create_cl_to_pair_dict(train_cls)
    test_cl_to_pair = create_cl_to_pair_dict(test_cls)
    #check right number of cls 
    assert len(train_cl_to_pair) == len(train_cls)
    assert len(test_cl_to_pair) == len(test_cls)

    #more checks 
    #unpack dict check no overlap and correct number 
    def flatten(l):
        return [item for sublist in l for item in sublist]

    train_flat = flatten(train_cl_to_pair.values())
    test_flat = flatten(test_cl_to_pair.values())

    assert len(train_flat) == len(train_pairs)
    assert len(test_flat) == len(test_pairs)
    assert len(set(train_flat).intersection(test_flat)) == 0
    
    return train_pairs, test_pairs, train_cls, test_cls

def dblind_split(seed, cell_lines, drugs, targets, train_size=0.8):
    train_drugs, test_drugs = train_test_split(drugs, train_size=train_size)
    assert len(set(train_drugs).intersection(test_drugs)) == 0
    frac_train_drugs = len(train_drugs) / len(drugs)
    frac_test_drugs = len(test_drugs) / len(drugs)

    print('Fraction of cls in sets, relative to all cls '\
            'before mising values are removed')          
    print(f'train fraction {frac_train_drugs}, test fraction {frac_test_drugs}')
    print('------')


    #add in the cls to each drug. 
    def create_cl_drug_pair(drugs):
        all_pairs = []
        for cl in cell_lines:
            pairs = cl + '::' + drugs
            #only keep pair if there is a truth value for it
            all_pairs.extend([pair for pair in pairs if pair in targets])

        return np.array(all_pairs)

    train_pairs = create_cl_drug_pair(train_drugs)
    test_pairs = create_cl_drug_pair(test_drugs)

    train_pairs = sklearn.utils.shuffle(train_pairs, random_state=seed)
    test_pairs = sklearn.utils.shuffle(test_pairs, random_state=seed)

    assert len(set(train_pairs).intersection(test_pairs)) == 0

    num_all_examples = len(cell_lines) * len(drugs)
    frac_train_pairs = len(train_pairs) / num_all_examples
    frac_test_pairs = len(test_pairs) / num_all_examples
            
    print('Fraction of cls in sets, relative to all cl drug pairs, after '\
            'mising values are removed')
    print(f'train fraction {frac_train_pairs}, test fraction '\
            f'{frac_test_pairs}')

    #checking split works as expected.  

    #create mapping of cls to cl drug pairs for test train and val set. 
    def create_cl_to_pair_dict(drugs):
        '''Maps a cell line to all cell line drug pairs with truth values
        '''
        dic = {}
        for d in drugs:
            dic[d] = []
            for cl in cell_lines:
                pair = cl + '::' + d
                #filters out pairs without truth values
                if pair in targets:
                    dic[d].append(pair)
        return dic

    train_cl_to_pair = create_cl_to_pair_dict(train_drugs)
    test_cl_to_pair = create_cl_to_pair_dict(test_drugs)
    #check right number of drugs 
    assert len(train_cl_to_pair) == len(train_drugs)
    assert len(test_cl_to_pair) == len(test_drugs)

    #more checks 
    #unpack dict check no overlap and correct number 
    def flatten(l):
        return [item for sublist in l for item in sublist]

    train_flat = flatten(train_cl_to_pair.values())
    test_flat = flatten(test_cl_to_pair.values())

    assert len(train_flat) == len(train_pairs)
    assert len(test_flat) == len(test_pairs)
    assert len(set(train_flat).intersection(test_flat)) == 0

    return train_pairs, test_pairs, train_drugs, test_drugs