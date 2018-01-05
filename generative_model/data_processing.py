from rdkit import Chem
from numpy import random
import numpy as np
'''
loading chembl into lines of smiles
'''
def load_chembl():
    f = open('data/chembl_smiles.txt')
    lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    f.close()
    return lines

chembl = load_chembl()
data_set = []
dictionary = {}
'''
create a character dictionary
'''
def added_to_dictionary(smile):
    for char in smile:
        if char not in dictionary:
            dictionary[char] = True

print('processing smiles !!!')

'''
! is the start token and ? is the end token
'''
for i in range(len(chembl)):
    if len(chembl[i])<=100:
        smile = '!'+chembl[i]+' '
        data_set.append(smile)
        added_to_dictionary(smile)

print('saving smiles')
vocabs = [ele for ele in dictionary]
print(len(vocabs))

'''
save the smiles string and vocabs
'''
np.savez_compressed('data/smiles_data.npz', data_set=np.array(data_set, dtype=object),vocabs=np.array(vocabs), dtype=object)