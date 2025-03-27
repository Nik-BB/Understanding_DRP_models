
import os
import pandas as pd
import numpy as np

cwd = os.getcwd()
if cwd[0].upper() == 'C':
    pass
    #mc_path = r'C:/Users/Nik/Documents/PhD_code/year3_code/ProtGraphRo/data/GDSC2_Wed Aug GDSC2_30_15_49_31_2023.csv'
    mc_path = 'data_et/GDSC2_Wed Aug GDSC2_30_15_49_31_2023.csv'
else:
    #mc_path = r'/data/home/wpw035/drp_omic_graph/Prot_rna_graph/data/GDSC2_Wed Aug GDSC2_30_15_49_31_2023.csv'
    mc_path = 'data_et/GDSC2_Wed Aug GDSC2_30_15_49_31_2023.csv'

def find_drug_to_max_con_gdsc2():
    '''Find max concentration of each drug used also find drugs with mutiple 
    max concentrations, only a handfull of these ~4 with smiles'''
    max_concentration = pd.read_csv(mc_path)

    #find mapping from drug to max concentration used
    #find drugs that have multiple max concentrations
    multi_con_drugs = []
    drug_to_max_con = {}
    for d in set(max_concentration['Drug Name']):
        d_subset = max_concentration[max_concentration['Drug Name'] == d]
        num_cons = len(set(d_subset['Max Conc']))
        if num_cons > 1:
            multi_con_drugs.append(d)
        else:
            #log Conc as have ln(ic50)
            drug_to_max_con[d] = np.log(d_subset['Max Conc'].iloc[0])           
    return drug_to_max_con, multi_con_drugs


def binarise_ic50(ic50_df, drug_to_max_con_dict):
    '''create binary drug effctive using max concentration as threshold
    ic50 lower than max con to be effective: 1, else not effective 0
    '''
    binary_ic50 = []
    for d, col in ic50_df.items():
        mask = col.dropna() < drug_to_max_con_dict[d]
        mask = mask.replace({True: int(1), False: int(0)})
        nans = col[col.isna()]
        binary_ic50.append(pd.concat([mask, nans]).loc[mask.index])
    return pd.DataFrame(binary_ic50, dtype=np.float32).T



def unit_test_binary_ic50(binary_ic50_df, ic50_cont, drug_to_max_con, drug, cl,verb=1):
    '''checks binary ic50 vals correct '''
    val = ic50_cont.loc[cl, drug]
    mc = drug_to_max_con[drug]
    if verb:
        print(f'testing {drug} for {cl}, ic50: {val} ')
        print(f'{drug} has max con of {mc}')
        print(f'thus {drug} efftive? {val < mc} ')
        print(f'binary value is {binary_ic50_df.loc[cl, drug]}')
    return (val < mc) == bool(binary_ic50_df.loc[cl, drug])