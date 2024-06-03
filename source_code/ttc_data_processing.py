'''Processing specific to the DeepTTC model'''
import sys
import codecs
import pandas as pd
import numpy as np 
import os
sys.path.insert(0, 'source_code') 
#sys.path.insert(0, '../') 
import bpe #bpe is just the needed funciton from subword-nmt github 
#but can pip install subword-nmt instead
#https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/apply_bpe.py


def smile_encoder(smile):
    '''Encodes smiles as ESPF's https://github.com/kexinhuang12345/ESPF'''
    epsf_dir = 'data/epsf'
    vocab_path = f'{epsf_dir}/drug_codes_chembl_freq_1500.txt'
    sub_csv = pd.read_csv(f'{epsf_dir}/subword_units_map_chembl_freq_1500.csv')

    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = bpe.BPE(bpe_codes_drug, merges=-1, separator='')

    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    max_d = 50
    #splits on smile substructures from EPSF github
    smile_substructures = dbpe.process_line(smile).split()
    encoded_substructures = np.asarray(
        [words2idx_d[sub] for sub in smile_substructures]) 
    # except:
    #     encoded_substructures = np.array([0])
    #     print('except')

    #made encoded length max_d by padding or just taking max_d charaters 
    l = len(encoded_substructures)
    if l < max_d:
        encoded_substructures = np.pad(
            encoded_substructures, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        encoded_substructures = encoded_substructures[:max_d]
        input_mask = [1] * max_d

    return encoded_substructures, np.asarray(input_mask)
