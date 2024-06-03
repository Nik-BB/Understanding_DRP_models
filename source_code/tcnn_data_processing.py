import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def pad_smiles(drugs_to_smiles_raw, max_smile_len=188, pad_character='!'):
    '''pads smile strings'''

    #checks 
    smile_len = [len(smile) for smile in drugs_to_smiles_raw]
    smile_len = np.array(smile_len)
    num_short = len(smile_len[smile_len > max_smile_len])
    mean_pad = np.mean(smile_len[smile_len < max_smile_len])
    print(f'Num smiles that will be shortened {num_short}')
    print(f'Mean number of padding characters needed {mean_pad}')
    
    #do padding and shortening
    new_smiles = []
    for smile in drugs_to_smiles_raw:
    #subset smiles that are too long
        if len(smile) > max_smile_len:
            new_smile = smile[: max_smile_len]
        #add padding to shorter smiles
        elif len(smile) < max_smile_len:
            new_smile = smile + pad_character * (max_smile_len - len(smile))
        #leave smiles that are the right length    
        else: 
            new_smile = smile
        new_smiles.append(new_smile)

    drugs_to_smiles = pd.Series(
        new_smiles, index=drugs_to_smiles_raw.index)
    #check above is done correctly. 
    for smile in drugs_to_smiles:
        assert len(smile) == max_smile_len
    
    return drugs_to_smiles

def hot_encode_smiles(drugs_to_smiles_pad, max_smile_len=188):
    label_encoder = LabelEncoder()
    int_encoded = label_encoder.fit_transform(list(''.join(drugs_to_smiles_pad)))
    int_encoded = int_encoded.reshape(len(drugs_to_smiles_pad), max_smile_len)

    one_hot_smiles  = np.zeros(
        shape = (len(drugs_to_smiles_pad), max_smile_len,max(int_encoded.flatten()) + 1))
    one_hot_smiles = one_hot_smiles.astype(np.float32)
    for i, sample in enumerate(int_encoded):
        for j, char_numb in enumerate(sample):
            one_hot_smiles[i, j, char_numb] = 1
            
    #dict to map drug to one hot enconded vec
    drug_to_hot_smile = {}
    for i, drug in enumerate(drugs_to_smiles_pad.index):
        drug_to_hot_smile[drug] = np.swapaxes(one_hot_smiles[i], 0, 1)
    drug_to_hot_smile = pd.Series(drug_to_hot_smile)

    return drug_to_hot_smile