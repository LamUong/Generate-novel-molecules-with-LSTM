import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time, math
from random import randint

def load_data():
    f = np.load('data/smiles_data.npz',allow_pickle=True)
    return f['data_set'],f['vocabs']

def tensor_from_chars_list(chars_list,vocabs,cuda):
    tensor = torch.zeros(len(chars_list)).long()
    for c in range(len(chars_list)):
        tensor[c] = vocabs.index(chars_list[c])
    return tensor.view(1,-1)
'''
create batches of batch_size
'''
def process_batch(sequences,batch_size,vocabs,cuda):
    chunk_len = len(sequences[0])-1
    end_of_i = int(len(sequences)/batch_size)*batch_size
    batches=[]
    for i in range(0,len(sequences),batch_size):
        input_list = []
        output_list = []
        for j in range(i,i+batch_size,1):
            if j <len(sequences):
                input_list.append(tensor_from_chars_list(sequences[j][:-1],vocabs,cuda))
                output_list.append(tensor_from_chars_list(sequences[j][1:],vocabs,cuda))
        inp = Variable(torch.cat(input_list, 0))
        target = Variable(torch.cat(output_list, 0))
        if cuda:
            inp = inp.cuda()
            target = target.cuda()
        batches.append((inp,target))
    train_split = int(0.9*len(batches))
    return batches[:train_split],batches[train_split:]
'''
group all the same length smiles together to process them in batch
'''
def process_data_to_batches(data,batch_size,vocabs,cuda):
    hash_length_data = {}
    for ele in data:
        l = len(ele)
        if l>=3:
            if l not in hash_length_data:
                hash_length_data[l] = []
            hash_length_data[l].append(ele)
    train_batches = []
    val_batches = []
    for length in hash_length_data:
        train,val = process_batch(hash_length_data[length],batch_size,vocabs,cuda)
        train_batches.extend(train)
        val_batches.extend(val)
    return train_batches,val_batches

def get_random_batch(train_batches):
    r_n = randint(0,len(train_batches)-1)
    return train_batches[r_n]

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



