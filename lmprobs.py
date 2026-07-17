import pickle
import os
import sys
import json
from operator import itemgetter
from datetime import datetime


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model_extra import Bert
from dataset_ACLM_new import MaskedDataset, CausalDataset, ValidationDataset
#from train_10m import get_batch

import numpy as np
from sklearn.neighbors import KDTree
from skimage.transform import resize
from nltk.util import trigrams
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from tqdm import tqdm

#from config import default_args

class AbstractSurprisalSpace:
    def __init__(self, dims):
        self.dims = dims
        self.filtered_to_original = []
        
    def fit(self, sequences):
        self.sequences = sequences

        print("Training language model. ")
        vocab = self.train(sequences)
        print("Building surprisal space. ")
        self.surprisalvecs = []
        for x in tqdm(sequences):
            self.surprisalvecs.append(self.surprisalizer_(x))
            
        self.currentsurprisalvecs = self.surprisalvecs.copy()
        self.filtered_to_original = list(range(len(self.surprisalvecs)))
        
        print("Building KD Tree. ")
        self.nnfinder = KDTree(self.surprisalvecs)

    def reset_dims(self, newdims):
        self.dims = newdims
        
        self.surprisalvecs = []
        for x in tqdm(self.sequences):
            self.surprisalvecs.append(self.surprisalizer_(x))
            
        self.currentsurprisalvecs = self.surprisalvecs.copy()
        self.filtered_to_original = list(range(len(self.surprisalvecs)))
        # If we reset the dimensionality, we reset the whole space back to the full pool.
        self.nnfinder = KDTree(self.surprisalvecs)
        
    def find_index(self, vec_index, k=5):
        size = self.nnfinder.data.shape[0]
        print("size:" ,size)
        if k > size:
            return [], [], tuple()

        query_vec = self.surprisalvecs[vec_index].reshape(1, -1)
        dists, indices = self.nnfinder.query(query_vec, k=k)

        dists = list(dists[0])
        indices = list(indices[0])

        mapped_indices = [self.filtered_to_original[i] for i in indices]
        mapped_vectors = itemgetter(*mapped_indices)(self.surprisalvecs)

        return dists, mapped_indices, mapped_vectors
        #dists, indices = self.nnfinder.query(self.surprisalvecs[vec_index].reshape(1,-1),
        #                              k=k)

        #return list(dists[0]), list(indices[0]), itemgetter(*list(indices[0]))(self.surprisalvecs)
    
    # Remove from the stored vectors
    def remove_from_space(self, to_remove):
        #print(f'length of surprisal space {len(self.currentsurprisalvecs)}')
        reverse_map = {v: i for i, v in enumerate(self.filtered_to_original)}
        filtered_indices = [reverse_map[i] for i in to_remove if i in reverse_map]
        for index in sorted(filtered_indices, reverse=True):
            # print(index)
            del self.currentsurprisalvecs[index] #make sure this behaves as a reference
            del self.filtered_to_original[index]
        print(f'length of surprisal space {len(self.currentsurprisalvecs)}')
        self.nnfinder = KDTree(self.currentsurprisalvecs)
        print("kdtree rebuilt")
        
class TrigramSurprisalSpace(AbstractSurprisalSpace):
    def __init__(self, dims):
        super(TrigramSurprisalSpace, self).__init__(dims)

    def train(self, sequences):
        # sequences = [[str(x) for x in list(y)] for y in sequences]
        # print('sequences', sequences[0])
        trainingdata, vocab = padded_everygram_pipeline(3, sequences)
        # print('trainingdata', trainingdata)
        # for i, g in enumerate(trainingdata):
        #     for v in g:
        #         print(v)
        #     if i > 50: break
        # print('vocab', vocab)
        # for i, v in enumerate(vocab):
        #     print(v)
        #     if i > 50: break
        self.lm = MLE(3)
        self.lm.fit(trainingdata, vocab)
        return vocab

    def surprisalizer_(self, sentence):
        if len(sentence) == 2:
            sentence.append('.')
        if len(sentence) == 1:
            sentence.append('.')
            sentence.append('.')
        if len(sentence) == 0:
            sentence.append('.')
            sentence.append('.')
            sentence.append('.')
        trisent = list(trigrams(sentence))
        
        surps = np.array([-self.lm.logscore(x[2], [x[0], x[1]]) for x in trisent if len(x)>=3])
        try:
            resized = np.nan_to_num(resize(surps, (self.dims,)))
        except ValueError:
            resized = np.array(['O'])
            print("sentence {} trisent {} surps {}".format(sentence, trisent, surps))
        return resized


class GPTBertSurprisalSpace(AbstractSurprisalSpace):
    def __init__(self,dims, dataloader,model):
        super().__init__(dims)
        self.dataloader = dataloader
        self.model=model
    def train(self, sequences):
        pass

    def surprisalizer_(self, dataloader):
        surprisals = []
        dataloader = iter(self.dataloader)
        self.model.eval()
        with torch.no_grad():
            for local_step in tqdm(range(len(dataloader)),desc="ACLM num steps"):
                input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(dataloader, args.device, 0)
                print(len(dataloader))
                with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
                    prediction,_, _, _, num_tokens = model(input_ids_, attention_mask_, target_ids_,return_all=True)
                    print(prediction.size())
                    prediction = prediction.flatten(0, 1)
                    print(prediction.size())
                    target_ids= target_ids_.flatten()

                    loss = F.cross_entropy(
                            Prediction, target_ids, reduction='none')
                    print(loss.size())
                    loss_np = loss.detach().cpu().numpy()
                    surprisals.append(loss_np)
                    print(len(surprisals))
        return surprisals
                     
if __name__ == "__main__":
    tss = TrigramSurprisalSpace(8)
    
    itemfile = open("/mimer/NOBACKUP/groups/naiss2024-6-297/gpt-bert/data/train_10M.txt", "r")
    tokens = [x.split(" ") for x in itemfile]
    print(f'orig tokens {len(tokens)}')
    print(tokens[3999])
    #print(tokens[:5000])

    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Fit Starting Time =", current_time)
    tss.fit(tokens)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Fit Stopping Time =", current_time)
    
    distances, indices, vectors = tss.find_index(3999)

    print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))
    
    distances, indices, vectors = tss.find_index(2124)

    print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))
    
    pickle.dump(tss, open("surprisals_8.pkl", "wb"))

    #loadtss = pickle.load("surprisals_8.pkl", "rb")

    with open("surprisals_8.pkl", "rb") as file:
        loadtss = pickle.load(file)

    distances, indices, vectors = loadtss.find_index(2124)

    print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))

    distances, indices, vectors = loadtss.find_index(1111)

    print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))


    # loadtss.reset_dims(7)
    
    # distances, indices, vectors = loadtss.find_index(1111)

    # print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))
        
    # pickle.dump(tss, open('tss1.pkl', "wb"))
    
    loadtss.remove_from_space([20,500,550,1024,2048,3333])
    # distances, indices, vectors = loadtss.find_index(1024)
    # print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))
    
'''
import pandas as pd

if __name__ == "__main__":
    tss = TrigramSurprisalSpace(8)

    # Load CSV file
    df = pd.read_csv("/mimer/NOBACKUP/groups/naiss2024-6-297/gpt-bert/data/train_10M_4k.csv")

    # Extract Tokens column and split into lists of tokens
    tokens = df['Tokens'].apply(lambda x: x.split()).tolist()
    
    print(f'orig tokens {len(tokens)}')
    print(tokens[3999])

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Fit Starting Time =", current_time)

    tss.fit(tokens)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Fit Stopping Time =", current_time)

    distances, indices, vectors = tss.find_index(3999)
    print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))

    distances, indices, vectors = tss.find_index(2124)
    print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))

    # Save the surprisal space
    pickle.dump(tss, open("surprisals_8_4k.pkl", "wb"))

    # Load and test
    with open("surprisals_8_6k.pkl", "rb") as file:
        loadtss = pickle.load(file)

    distances, indices, vectors = loadtss.find_index(2124)
    print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))

    distances, indices, vectors = loadtss.find_index(1111)
    print("We get distances {} at indices {}.\nThe vectors are:\n{}".format(distances, indices, vectors))

    # Remove specific indices from space
    loadtss.remove_from_space([20, 500, 550, 1024, 2048, 3333])

'''
