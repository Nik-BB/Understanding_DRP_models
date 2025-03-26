import os
import torch
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset as Dataset_g
from torch.utils.data import Dataset


example_data = True #use and load in example data just a subset of full data


device = "cuda" if torch.cuda.is_available() else "cpu"
#device = 'cpu'
cwd = os.getcwd()

#working locally
if cwd[0].upper() == 'C':
    data_path = r'C:/Users/Nik/Documents/PhD_code/year3_code/Binary_ic50/Binary_ic50_lit/data/'
    #data_path = f'c:/Users/Nik/Documents/PhD_code/year3_code/drp_omic_graph/Prot_rna_graph/data/'
    omic_data_path = r'C:\Users\Nik\Documents\PhD_code\year3_code\Downloaded_data\\'
    rna_dir = r'C:\Users\Nik\Documents\PhD_code\year3_code\Downloaded_data'
    tt_path = data_path

else:
    data_path = f'/data/home/wpw035/drp_omic_graph/Prot_rna_graph/data/'
    omic_data_path = f'/data/home/wpw035/Codebase/downloaded_data_small/'
    rna_dir = r'/data/home/wpw035/GDSC/downloaded_data'
    tt_path = f'/data/home/wpw035/DRP-lit-comp-aberrations/Binary_ic50_lit/data/'

if example_data:
    example_dir = 'data/'
    tt_path = 'data/example_data/'

    
#read and format target values gdsc1 
# def read_targets():
#     df_ic50 = pd.read_csv(f'{data_path}\\GDSC1_ic50.csv')
#     frame = {}
#     for d in np.unique(df_ic50['CELL_LINE_NAME']):
#         cellDf = df_ic50[df_ic50['CELL_LINE_NAME'] == d]
#         cellDf.index = cellDf['DRUG_NAME']
#         frame[d] = cellDf['LN_IC50']
#     def remove_repeats_mean_gdsc1(frame, df_ic50): 
#         new_frame = {}
#         for cell_line in np.unique(df_ic50['CELL_LINE_NAME']):
#             temp_subset = frame[cell_line].groupby(frame[cell_line].index).mean()
#             new_frame[cell_line] = temp_subset
#         return new_frame  

#     new_frame = remove_repeats_mean_gdsc1(frame, df_ic50)
#     ic50_df1 = pd.DataFrame(new_frame).T
    
#     return ic50_df1


def read_gdsc2(path):
    '''reads gdsc2 dataset in as a matrix shape = (drugs, cell lines) '''
    
    raw_df = pd.read_csv(path, index_col=0)

    frame = {}
    for cl in np.unique(raw_df['Cell Line Name']):
        cl_df = raw_df[raw_df['Cell Line Name'] == cl]
        frame[cl] = cl_df['IC50']

    #take the mean of repeated entries
    aved_frame = {}
    for cl in np.unique(raw_df['Cell Line Name']):
        cl_df_aved = frame[cl].groupby(frame[cl].index).mean()
        aved_frame[cl] = cl_df_aved

    return pd.DataFrame(aved_frame)

def one_hot_encode(x):
    '''One hot encodes list like input'''
    frame = {}
    for i, label in enumerate(x):
        hot_vec = np.zeros(len(x))
        hot_vec[i] = 1
        frame[label] = hot_vec
    encoded_df = pd.DataFrame(frame)
    return encoded_df


def read_prot():
    
    prot_raw = pd.read_csv(
        f'{omic_data_path}Proteinomics_large.tsv',
        sep = '\t', header=1
    )
    prot_raw.drop(index=0, inplace=True)

    prot_raw.index = prot_raw['symbol']
    prot_raw.drop(columns=['symbol', 'Unnamed: 1'], inplace=True)

    #replace missing protomics values
    p_miss = prot_raw.isna().sum().sum() / (len(prot_raw) * len(prot_raw.columns))
    print(f'Number of missing prot values {p_miss}')

    assert sum(prot_raw.index.duplicated()) == 0
    assert sum(prot_raw.columns.duplicated()) == 0
    assert sum(prot_raw.index.isna()) == 0
    assert sum(prot_raw.columns.isna()) == 0
    
    return prot_raw
                
def read_rna_gdsc(gdsc_dir_path):
    #read in rna-seq data
    #gdsc_path = '/data/home/wpw035/GDSC'
    rna_raw = pd.read_csv(f'{gdsc_dir_path}/gdsc_expresstion_dat.csv')
    rna_raw.index = rna_raw['GENE_SYMBOLS']
    rna_raw.drop(columns=['GENE_SYMBOLS','GENE_title'], inplace=True)
    cell_names_raw = pd.read_csv(f'{gdsc_dir_path}/gdsc_cell_names.csv', skiprows=1, skipfooter=1)
    cell_names_raw.drop(index=0, inplace=True)

    #chagne ids to cell names
    id_to_cl = {}
    for _, row in cell_names_raw.iterrows():
        cell_line = row['Sample Name']
        ident = int(row['COSMIC identifier'])
        id_to_cl[ident] = cell_line

    ids = rna_raw.columns
    ids = [int(iden.split('.')[1]) for iden in ids] 

    #ids that are in rna_raw but don't have an assocated cl name 
    #from cell_names_raw (not sure why we have these)
    missing_ids = []
    for iden in ids:
        if iden not in id_to_cl.keys():
            missing_ids.append(iden)
    missing_ids = [f'DATA.{iden}' for iden in missing_ids]      
    rna_raw.drop(columns=missing_ids, inplace=True)

    cell_lines = []
    for iden in ids:
        try:
            cell_lines.append(id_to_cl[iden])
        except KeyError:
            pass
    rna_raw.columns = cell_lines
    rna_raw = rna_raw.T

    #take out duplicated cell line
    rna_raw = rna_raw[~rna_raw.index.duplicated()] 
    #take out nan cols
    rna_raw = rna_raw[rna_raw.columns.dropna()]
    #take out duplciated cols
    rna_raw = rna_raw.T[~rna_raw.columns.duplicated()].T

    #check for duplications and missing value in cols and index
    assert sum(rna_raw.index.duplicated()) == 0
    assert sum(rna_raw.columns.duplicated()) == 0
    assert sum(rna_raw.index.isna()) == 0
    assert sum(rna_raw.columns.isna()) == 0
    
    return rna_raw

def create_drug_to_smiles_mapping_gdsc2(pub_ids_path, save=True):
    '''creates pd serise to map drug names to smiles strings'''
    import pubchempy as pcp

    drug_info_df = pd.read_csv(pub_ids_path, index_col=0)
    drug_info_df = drug_info_df.replace('-', np.nan)
    drug_info_df = drug_info_df.replace('None', np.nan)
    drug_info_df = drug_info_df.replace('none', np.nan)

    #finds drugs that have mutiple pub chem ids
    num_drugs = len(drug_info_df['pubchem'])

    idx_mask =[]
    for idx in drug_info_df['pubchem']:
        if type(idx) == str:
            idx_mask.append(idx.isnumeric())
        elif type(idx) == float:
            idx_mask.append(True)
    idx_mask = np.array(idx_mask)

    #6 entries with mutiple pubchem ID's that we drop
    drug_info_df['pubchem'][~idx_mask] = np.nan

    drug_info_df  = drug_info_df.dropna(subset='pubchem')
    #remove duplicate (only 1)
    drug_info_df = drug_info_df[~drug_info_df['pubchem'].duplicated()] 

    num_drug_with_pub_idx = len(drug_info_df['pubchem'].dropna())
    num_drug_with_pub_idx, num_drugs

    print(f'orignally {num_drugs} drugs')
    print(f'{num_drug_with_pub_idx} drugs with smiles')
    
    #finds smile strings using pubchempy
    drug_to_smiles_gdsc2  = {}
    for _, row in drug_info_df.iterrows():
        drug, pub_id = row['drug_name'], row['pubchem']
        c = pcp.get_compounds(pub_id)
        if len(c) > 0:
            smile = c[0].canonical_smiles
            drug_to_smiles_gdsc2[drug] = smile
    drug_to_smiles_gdsc2 = pd.Series(drug_to_smiles_gdsc2)

    if save:
        drug_to_smiles_gdsc2.to_csv('data/drugs_to_smiles_gdsc2.csv')
    
    return drug_to_smiles_gdsc2


class ExampleData():
    '''loads in example data subset of full dataset from data folder'''
    def __init__(self, data_dir='..data/'):
        rna = pd.read_csv(f'{data_dir}example_data/xpr_sub.csv', index_col=0).astype(np.float32)
        ic50 = pd.read_csv(f'{data_dir}example_data/ic50_sub.csv', index_col=0).astype(np.float32)
        drugs_to_smiles = pd.read_csv('data/drugs_to_smiles_gdsc2.csv', index_col=0)['0']
        self.rna_omic = rna
        self.y = ic50
        self.drugs_to_smiles = drugs_to_smiles

class ClDrugIntegrator():
    '''loads in the omics data specified and overlapping drug data'''

    def __init__(self, omics=['prot', 'rna'], drug_rep='smiles'):
        #load in omic and target data
        all_omics = []
        if example_data:
            print('Running with example data, subset of full dataset')
            print('if full dataset has been downloaded, as outlined in ReadME set example_data=False to use full dataset in create_dataset.py')
            rna = pd.read_csv(f'{example_dir}example_data/xpr_sub.csv', index_col=0).astype(np.float32)
            ic50 = pd.read_csv(f'{example_dir}example_data/ic50_sub.csv', index_col=0).astype(np.float32)
            drugs_to_smiles = pd.read_csv('data/drugs_to_smiles_gdsc2.csv', index_col=0)
            self.rna_omic = rna
            self.y = ic50
            self.drugs_to_smiles = drugs_to_smiles       
        else:
            if 'prot' in omics:
                self.prot_omic = read_prot()
                self.prot_omic = read_rna_gdsc(self.prot_omic)
                all_omics.append(self.prot_omic)
            if 'rna' in omics:
                self.rna_omic = read_rna_gdsc(rna_dir)
                self.rna_omic = self.rna_omic.astype(np.float32)
                #drop RH-18 and HCC202 as have over 95% missing values. (next most is 60%)
                self.rna_omic = self.rna_omic.drop(['RH-18', 'HCC202'])
                all_omics.append(self.rna_omic)

            self.y = read_gdsc2(
                f'{data_path}GDSC2_Wed Aug GDSC2_30_15_49_31_2023.csv').T 

            #only keep overlapping cell lines    
            overlap_cls = set(self.y.index)
            for omic in all_omics:
                overlap_cls = overlap_cls.intersection(omic.index)
            overlap_cls  = list(overlap_cls)

            self.y = self.y.loc[overlap_cls]
            if 'prot' in omics:
                self.prot_omic = self.prot_omic.loc[overlap_cls]
            if 'rna' in omics:
                self.rna_omic = self.rna_omic.loc[overlap_cls]

            if drug_rep == 'smiles':
                self.drugs_to_smiles = pd.read_csv(
                    f'{data_path}drugs_to_smiles_gdsc2.csv', index_col=0)
                overlapping_drugs = set(
                    self.drugs_to_smiles.index).intersection(self.y.columns)
                overlapping_drugs = list(overlapping_drugs)
                self.y = self.y[overlapping_drugs]     

    def drop_na_prot(self):
        '''fs method to drop nans from prot omics '''
        self.prot_omic = self.prot_omic.dropna(axis=1)




        

def split(seed, _all_cls, _all_drugs, all_targets, train_size=0.8, 
          split_type='cblind'):
    '''Train test split for cancer or drug blind testing (no val set)
    
    Cancer blind testing means cell lines do not overlap between
    the train test and val sets. 
    Drug blind testing means drugs do not overlap between
    the train test and val sets. 
    '''
    #input cheks
    if type(_all_drugs) == type(_all_cls) != pd.Index:
        print('_all_drugs and _all_cls need to be PD indxes')

    #cancer blind splitting
    if split_type == 'cblind': 
        train_cls, test_cls = train_test_split(_all_cls, train_size=train_size, 
                                               random_state=seed)

        assert len(set(train_cls).intersection(test_cls)) == 0

        frac_train_cl = len(train_cls) / len(_all_cls)
        frac_test_cl = len(test_cls) / len(_all_cls)

        print('Fraction of cls in sets, relative to all cls'\
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
        
    #drug blind splitting    
    if split_type == 'dblind':
        train_ds, test_ds = train_test_split(_all_drugs, train_size=train_size, 
                                           random_state=seed)

        assert len(set(train_ds).intersection(test_ds)) == 0

        frac_train_ds = len(train_ds) / len(_all_drugs)
        frac_test_ds = len(test_ds) / len(_all_drugs)

        print('Fraction of drugs in sets, relative to all drugs'\
              'before mising values are removed')          
        print(f'train fraction {frac_train_ds}, test fraction {frac_test_ds}')
        print('------')

        #add in the drugs to each cell line. 
        def create_cl_drug_pair(drugs):
            all_pairs = []
            for cell in _all_cls:
                pairs = cell + '::' + drugs
                #only keep pair if there is a truth value for it
                for pair in pairs:
                    if pair in all_targets:
                        all_pairs.append(pair)


            return np.array(all_pairs)

        train_pairs = create_cl_drug_pair(train_ds)
        test_pairs = create_cl_drug_pair(test_ds)


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
 
    if split_type == 'cblind':   
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

    if split_type == 'dblind':
        def create_cl_to_pair_dict(drugs):
            '''Maps a drug to all cell line drug pairs with truth values

            '''
            dic = {}
            for drug in drugs:
                dic[drug] = []
                for cell in _all_cls:
                    pair = cell + '::' + drug
                    #filters out pairs without truth values
                    if pair in all_targets:
                        dic[drug].append(pair)
            return dic
        
        train_cl_to_pair = create_cl_to_pair_dict(train_ds)
        test_cl_to_pair = create_cl_to_pair_dict(test_ds)
        #check right number of drugs 
        assert len(train_cl_to_pair) == len(train_ds)
        assert len(test_cl_to_pair) == len(test_ds)
    
    # more checks 
    #unpack dict check no overlap and correct number 
    def flatten(l):
        return [item for sublist in l for item in sublist]

    train_flat = flatten(train_cl_to_pair.values())
    test_flat = flatten(test_cl_to_pair.values())

    assert len(train_flat) == len(train_pairs)
    assert len(test_flat) == len(test_pairs)
    assert len(set(train_flat).intersection(test_flat)) == 0
    
    return train_pairs, test_pairs


class SmileTokDataset(Dataset):
    '''custom dataset for tokenising smiles data'''
    def __init__(self, drug_cl_pairs, pairs_to_smiles, tokenizer, max_len=128):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.pairs_to_smiles = pairs_to_smiles
        self.drug_cl_pairs = drug_cl_pairs
        #subset pairs_to_smiles
        self.pairs_to_smiles = pairs_to_smiles.loc[drug_cl_pairs]

    def __len__(self):
        return len(self.drug_cl_pairs)
    
    def __getitem__(self, idx):
        smile = self.pairs_to_smiles.iloc[idx]
        tokens = self.tokenizer(smile, truncation=True, padding='max_length', 
                           max_length=self.max_len, return_tensors='pt', 
                           add_special_tokens=True)
        return {'input_ids': tokens['input_ids'].flatten().to(device), 
                'attention_mask': tokens['attention_mask'].flatten().to(device)}

def load_tt_split(df, train_pairs, val_pairs, test_pairs):
    return df.loc[train_pairs], df.loc[val_pairs], df.loc[test_pairs]


def load_data_dgraph(x_omic, x_drug, y_pairs, split_seed, 
                     split_type='c_blind', batch_size=128):
    
    import torch_geometric.data as tgd
    from torch_geometric.loader import DataLoader as DataLoaderGeo


    pairs_path = f'{tt_path}train_test_pairs/{split_type}/seed_'
    train_pairs = pd.read_csv(
        f'{pairs_path}{split_seed}_train', header=None)[0].values
    val_pairs = pd.read_csv(
        f'{pairs_path}{split_seed}_val', header=None)[0].values
    test_pairs = pd.read_csv(
        f'{pairs_path}{split_seed}_test', header=None)[0].values
    
    xo_train, xo_val, xo_test = load_tt_split(
        x_omic, train_pairs, val_pairs, test_pairs)
    
    xd_train, xd_val, xd_test = load_tt_split(
        x_drug, train_pairs, val_pairs, test_pairs)
    
    y_train, y_val, y_test = load_tt_split(
        y_pairs, train_pairs, val_pairs, test_pairs)

    train_graph_dls = into_dls([xo_train.values,
                                tgd.Batch().from_data_list(xd_train.values), 
                                np.expand_dims(y_train, 1)],
                                batch_size=batch_size,
                                shuffle_per_epoch=True)

    val_graph_dls = into_dls([xo_val.values,
                              tgd.Batch().from_data_list(xd_val.values),
                              np.expand_dims(y_val, 1)], 
                              batch_size=batch_size)

    test_graph_dls = into_dls([xo_test.values, 
                               tgd.Batch().from_data_list(xd_test.values),
                               np.expand_dims(y_test, 1)],
                               batch_size=batch_size)
    
    return train_graph_dls, val_graph_dls, test_graph_dls, y_test
    
    

def load_data_omic_df(x_omic, x_drug, y_pairs, split_seed, mm_data=False,
                      split_type='c_blind', expand_xo_dims=True, batch_size=128):
    
    pairs_path = f'{tt_path}train_test_pairs/{split_type}/seed_'
    train_pairs = pd.read_csv(
        f'{pairs_path}{split_seed}_train', header=None)[0].values
    val_pairs = pd.read_csv(
        f'{pairs_path}{split_seed}_val', header=None)[0].values
    test_pairs = pd.read_csv(
        f'{pairs_path}{split_seed}_test', header=None)[0].values

    xo_train, xo_val, xo_test = load_tt_split(
        x_omic, train_pairs, val_pairs, test_pairs)
    
    xd_train, xd_val, xd_test = load_tt_split(
        x_drug, train_pairs, val_pairs, test_pairs)
    
    y_train, y_val, y_test = load_tt_split(
        y_pairs, train_pairs, val_pairs, test_pairs)

    if expand_xo_dims:
        xo_train = np.expand_dims(xo_train, 1)
        xo_val = np.expand_dims(xo_val, 1)
        xo_test = np.expand_dims(xo_test, 1)

    train_graph_dls = into_dls([xo_train.values,
                                xd_train['drug_encoding'].values, 
                                np.expand_dims(y_train, 1)],
                                batch_size=batch_size,
                                shuffle_per_epoch=True)

    val_graph_dls = into_dls([xo_val.values,
                              xd_val['drug_encoding'].values,
                              np.expand_dims(y_val, 1)], 
                              batch_size=batch_size)

    test_graph_dls = into_dls([xo_test.values, 
                               xd_test['drug_encoding'].values,
                               np.expand_dims(y_test, 1)],
                               batch_size=batch_size)
    
    return train_graph_dls, val_graph_dls, test_graph_dls, y_test

def load_data_smile_dict(x_omic, pairs_to_smiles,  y_pairs, tokenizer, split_seed, m_data=False,
                      split_type='c_blind', expand_xo_dims=True, batch_size=128):

    '''load data in when encoding smiles with tokenizer and df omic rep'''
    
    pairs_path = f'{tt_path}train_test_pairs/{split_type}/seed_'
    train_pairs = pd.read_csv(
        f'{pairs_path}{split_seed}_train', header=None)[0].values
    val_pairs = pd.read_csv(
        f'{pairs_path}{split_seed}_val', header=None)[0].values
    test_pairs = pd.read_csv(
        f'{pairs_path}{split_seed}_test', header=None)[0].values


    xo_train, xo_val, xo_test = load_tt_split(
        x_omic, train_pairs, val_pairs, test_pairs)

    y_train, y_val, y_test = load_tt_split(
        y_pairs, train_pairs, val_pairs, test_pairs)

    xd_train = SmileTokDataset(train_pairs, pairs_to_smiles, tokenizer)
    xd_val = SmileTokDataset(val_pairs, pairs_to_smiles, tokenizer)
    xd_test = SmileTokDataset(test_pairs, pairs_to_smiles, tokenizer)

    if expand_xo_dims:
        xo_train = np.expand_dims(xo_train, 1)
        xo_val = np.expand_dims(xo_val, 1)
        xo_test = np.expand_dims(xo_test, 1)

    train_graph_dls = into_dls([xo_train,
                                xd_train, 
                                np.expand_dims(y_train, 1)],
                                batch_size=batch_size)

    val_graph_dls = into_dls([xo_val,
                              xd_val,
                              np.expand_dims(y_val, 1)], 
                              batch_size=batch_size)

    test_graph_dls = into_dls([xo_test, 
                               xd_test,
                               np.expand_dims(y_test, 1)],
                               batch_size=batch_size)
    
    return train_graph_dls, val_graph_dls, test_graph_dls, y_test


def load_data_smile(x_omic, pairs_to_smiles,  y_pairs, tokenizer, split_seed, m_data=False,
                      split_type='c_blind', expand_xo_dims=True, batch_size=128):

    '''load data in when encoding smiles with tokenizer and df omic rep'''

    pairs_path = f'{tt_path}train_test_pairs/{split_type}/seed_'
    train_pairs = pd.read_csv(
        f'{pairs_path}{split_seed}_train', header=None)[0].values
    val_pairs = pd.read_csv(
        f'{pairs_path}{split_seed}_val', header=None)[0].values
    test_pairs = pd.read_csv(
        f'{pairs_path}{split_seed}_test', header=None)[0].values


    y_train = y_pairs[train_pairs]
    y_val = y_pairs[val_pairs]
    y_test = y_pairs[test_pairs]

    xo_train = x_omic.loc[train_pairs].values
    xo_val = x_omic.loc[val_pairs].values
    xo_test = x_omic.loc[test_pairs].values

    xd_train = SmileTokDataset(train_pairs, pairs_to_smiles, tokenizer)
    xd_val = SmileTokDataset(val_pairs, pairs_to_smiles, tokenizer)
    xd_test = SmileTokDataset(test_pairs, pairs_to_smiles, tokenizer)

    if expand_xo_dims:
        xo_train = np.expand_dims(xo_train, 1)
        xo_val = np.expand_dims(xo_val, 1)
        xo_test = np.expand_dims(xo_test, 1)

    train_graph_dls = into_dls([xo_train,
                                xd_train, 
                                np.expand_dims(y_train, 1)],
                                batch_size=batch_size,
                                shuffle_per_epoch=True)

    val_graph_dls = into_dls([xo_val,
                              xd_val,
                              np.expand_dims(y_val, 1)], 
                              batch_size=batch_size)

    test_graph_dls = into_dls([xo_test, 
                               xd_test,
                               np.expand_dims(y_test, 1)],
                               batch_size=batch_size)
    
    return train_graph_dls, val_graph_dls, test_graph_dls, y_test


def into_dls(x: list, batch_size=512, shuffle_per_epoch=False):
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

    if shuffle_per_epoch:
        g1 = torch.Generator().manual_seed(1)
        g2 = torch.Generator().manual_seed(1)
        g3 = torch.Generator().manual_seed(1)
    else:
        g1, g2, g3 = None, None, None
        
    if isinstance(x[0], tgd.Batch): #or isinstance(x[0], OmicGraphDataset):
        print('Graph omic data')
        x[0] = DataLoaderGeo(x[0], batch_size=batch_size, generator=g1, shuffle=shuffle_per_epoch)
    else:
        x[0] = DataLoader(x[0], batch_size=batch_size, generator=g1, shuffle=shuffle_per_epoch)

    if isinstance(x[1], tgd.Batch):
        print('Graph drug data')
        x[1] = DataLoaderGeo(x[1], batch_size=batch_size, generator=g2, shuffle=shuffle_per_epoch)
    else:
        x[1] = DataLoader(x[1], batch_size=batch_size, generator=g2, shuffle=shuffle_per_epoch)

    x[2] = DataLoader(x[2], batch_size=batch_size, generator=g3, shuffle=shuffle_per_epoch)    

    return x
