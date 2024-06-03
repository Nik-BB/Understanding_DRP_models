'''Functions to load and clean data'''

import numpy as np
import pandas as pd 

def load_omics_drugs_target(omic_dir_path, gdsc2_target_path, 
                            pubchem_ids_path, save=False):
    '''main function that loads omics, drug and IC50 data'''

    gdsc2_ic50 = read_gdsc2(gdsc2_target_path)
    rna = read_rna_gdsc(omic_dir_path)
    #drop RH-18 and HCC202 as have over 95% missing values. (next most is 60%)
    rna = rna.drop(['RH-18', 'HCC202'])
    overlapping_cls = list(set(rna.index).intersection(gdsc2_ic50.columns))
    rna = rna.loc[overlapping_cls]
    gdsc2_ic50 = gdsc2_ic50[overlapping_cls]
    if save:
        dts = create_drug_to_smiles_mapping_gdsc2(pub_ids_path=pubchem_ids_path, save=save)
    else:
        dts = pd.read_csv('data/drugs_to_smiles_gdsc2.csv', index_col=0)['0']
    drugs_with_smiles = dts.index
    gdsc2_ic50 = gdsc2_ic50.loc[drugs_with_smiles] 
    gdsc2_ic50 = gdsc2_ic50.T

    rna = rna.astype(np.float32)
    gdsc2_ic50 = gdsc2_ic50.astype(np.float32)

    return rna, gdsc2_ic50, dts


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

def read_genomics(gen_path, gdsc_dir_path):
    #genomics data imports (mut (mutation) and cna (copy number alteration)) used in paper
    gen_raw = pd.read_csv(gen_path, delimiter='\t', index_col=0)
    gdsc_path = '/data/home/wpw035/GDSC'
    cell_names = pd.read_csv(
        f'{gdsc_dir_path}/gdsc_cell_names.csv', skiprows=1, skipfooter=1)
    cell_names.index = cell_names['COSMIC identifier']

    #only want mut and cna data not crh
    features = [i for i in gen_raw.index if i.split(':')[0][: 3] != 'chr']
    gen = gen_raw.loc[features].T
    assert len(gen.columns) == 735 #num features from paper
    #add cl names as index rather than COMSMI i.d
    new_index = []
    for idx in gen.index.astype('float'):
        cl = cell_names.loc[idx]['Sample Name']
        new_index.append(cl)
    assert len(new_index) == len(set(new_index)) #no duplicates
    gen.index = new_index

    #check for duplications and missing value in cols and index
    assert sum(gen.index.duplicated()) == 0
    assert sum(gen.columns.duplicated()) == 0
    assert sum(gen.index.isna()) == 0
    assert sum(gen.columns.isna()) == 0
    return gen.astype(np.float32)



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
