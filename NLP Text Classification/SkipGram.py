# coding: utf-8

# In[ ]:


#From the Exercise
from __future__ import division
import argparse
import pandas as pd


# useful stuff
import numpy as np
# from scipy.special import expit
# from sklearn.preprocessing import normalize


__authors__ = ['re:']
__emails__  = ['re:']
#From the Exercise

#Other Imports
import string
import pickle # to save and load the model
import nltk
import sys
import re
import io
from nltk.tokenize import word_tokenize #for tokenization
from collections import Counter #for word count

import os 
import random 
import time 
import matplotlib 
import matplotlib.pyplot as plt 

#PART 1: Tokenizing: Removing punctuation, tokenization and conversion to lowercase 
def text2sentences(path):

    sentences = [] #list for sentences
    # removing punctuations. maketrans replaces the declared punctuations with blank spaces
    puncs = str.maketrans('', '', '''!"#$%&()*+,./:;<=>?@[\]^_`{|}~''')

    # read file. convert to lowercase. remove puntuation and append to the list of sentences
    with open(path, encoding="utf8") as f:
        for l in f:
            sent = l.lower().split()
            rem_puncs = [word.translate(puncs) for word in sent]
            sentences.append(rem_puncs)

    return sentences

#PART 2: (not changed: from the Question) Creating word pairs and similarity index for later testing 
def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs

#Previously used sigmoid but now using log of sigmoid because produces outputs in scale of (-∞, 0] 
#so the derivative would be in simpler form
def sigmoid(x):
    return 1.0 /(1 + np.exp(-x))


#class that defines skipgram
class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5):
        
        self.minCount = max(minCount, 1) #to handle the minimum occurence of a word and making sure of one occurence
        self.sentences = sentences #use self for context #sentences as sentences
        self.negativeRate = negativeRate #for noise words extraction
        self.nEmbed = int(nEmbed) #represents feature ventor's dimensionality making sure it returns int values
        self.winSize = winSize #window Size
        self.generate_vocabulary(minCount) #generating vocabulary
    
        
        self.word2idx = None
        self.unigram = None

#generating vocabulary and dictionary by merging the two functions used previously
#uses the tokenized sentences to generate a vocabulary and word occurence 
   
    def generate_vocabulary_dictionary(self, unigram_power=0.75):
        # to calculate frequency of occurence of a word
        word_freq = {}
        # loop for parsing through sentences
        for sent in self.sentences:
            # inner loop for parsing through words in a sentence
            for word in sent:
                # populating the dictionary with word_freq
                word_freq[word] = word_freq.get(word, 0) + 1

        
        #for val, word  in self.original_vocabulary.items():
        #    if val > minCount:
        #self.cleaned_vocabulary[word] = val
        if self.minCount > 1:
            word_freq = {word:freq for word, freq in word_freq.items() if freq >= self.minCount}
            
        # word2idx where word is the key and idx is the value
        self.word2idx = {w: idx for (idx, w) in enumerate(word_freq.keys())}

        # Initializing array where n=1 => unigram
        unigram = np.zeros(len(self.word2idx))
        # loop for calulating probablity of each word 
        for word, frequency in word_freq.items():
            # frquency to the power of 0.75
            f = frequency ** unigram_power
            # Updating
            unigram[self.word2idx[word]] = f
        
        # end Normalization
        self.unigram = unigram / np.sum(unigram)

#Generating training data that takes in a tokenized sentence list and outputs a 2d list of positive samples 
    def compute_positives(self):

        # list initialization
        #Pindexes positive sampling of tokens
        P = [] 
        #self.new_vocab_list = []
        V = len(self.word2idx)
        number_of_sentences = len(self.sentences)
        #self.generate_Dictionary() # create a Data Set of (word, context) pairs
        #self.neg_sampling()
        # This step was unhandled the case of rare or unseen words so here we set non existant new word as index -1
        
        sentences_index_form = [None]*number_of_sentences
        #self.new_vocab_len = len(self.new_vocab_list) #length of the new list
        for idx, sent in enumerate(self.sentences):
            sentences_index_form[idx] = [self.word2idx.get(w, -1) for w in sent]

        # calulating P(t)
        number_of_positives = np.zeros(V, dtype=int)
        for idx, sent_word_indices in enumerate(sentences_index_form):
            for i, word_idx in enumerate(sent_word_indices):
                if word_idx < 0:
                    continue
                first = max(0, i-self.winSize)
                last = min(i+self.winSize+1, len(sent_word_indices))
                number_of_positives[word_idx] += (last - first - 1)
        #self.W_int = np.random.rand(self.new_vocab_len, self.nEmbed)
        # Now we can allocate the memory for P in advance
        P = [None]*V
        #self.W1 = np.random.rand(self.new_vocab_len, self.nEmbed) 
        for word_idx in range(V):
            P[word_idx] = np.zeros(number_of_positives[word_idx], dtype=int)
        #self.W2 = np.random.rand(self.new_vocab_len, self.nEmbed) 
        # To polulate P we RECORD last position filled in P[t] using P_next_position[t]
        #self.W2[w, :] = self.W_int[w, :]
        P_next_position = [0]*V

        # parse each sentence to create target and context word pairs
        for idx, sent_word_indices in enumerate(sentences_index_form):
            if (idx + 1) % 100 == 0:
                print('Processing sentence', idx + 1, '/', number_of_sentences)
            # parse each word and increment it to Positive list
            for i, word_idx in enumerate(sent_word_indices):
                if word_idx < 0:
                    continue
                first = max(0, i-self.winSize)
                last = min(i+self.winSize+1, len(sent_word_indices))
                number_of_words = (last - first - 1)
                position = P_next_position[word_idx]
               
                P[word_idx][position:position+number_of_words] = np.asarray(sent_word_indices[first:i] + sent_word_indices[i+1:last])
                P_next_position[word_idx] += number_of_words

        # If minCount > 1 then a lot of -1 in P. We will remove them
        #for num_ep in range(epochs):

            #print("Epoch:  {}".format(str(ep  och)))
        if self.minCount > 1:
            for word_idx in range(V):
                P[word_idx] = np.delete(P[word_idx], np.where(P[word_idx] < 0))
               #for idxw, word in enumerate(self.new_vocab_list): 
        # Remove duplicates
        for word_idx in range(V):
            P[word_idx] = np.unique(P[word_idx])
        return P


"""negative sampling: we create false (Word, context) pairs by randomly picking words from the dictionary we just created"""
    def neg_sampling(self, trgt, Ptrgt):
        #word_frequency_list = []  
        #self.neg_Dictionary= {} 
        # Remove indices of t and Pt as they cannot be negative wrt to t
        neg_idc = Ptrgt.tolist() + [trgt]
        probs = np.copy(self.unigram)
        #freq_arr = np.asarray(word_frequency_list)  # converting to np.array
        #power = 0.75 
        probs[neg_idc] = 0
        probs /= np.sum(probs)
        neg_samples = np.random.choice(len(self.unigram), size=self.negativeRate, p=probs)

        return neg_samples


#Training the model with i/p as path to store model and o/p as saved path to model and trained embeddings
    def train(self, stepsize, epochs, patience=3, save_model_path=None):
        
        print('Generating vocabulary & dictionary')
        #start = time.time()
        self.generate_vocabulary_dictionary()
        
     
        P = self.compute_positives()

        if save_model_path is not None:
            save_model_path = os.path.expanduser(save_model_path)
            save_dir = os.path.dirname(save_model_path)
            if save_dir != '' and not os.path.exists(save_dir):
                os.makedirs(save_dir)

        V = len(self.word2idx)
        
        # Init
        W = np.random.rand(self.nEmbed, V)
        C = np.random.rand(V, self.nEmbed)

        losses = []
        loss_best = 1e100
        epochs_no_improvement = 0

        start = time.time()
        for epoch_idx in range(epochs):
            print('Epoch', epoch_idx + 1)
            
            # iterating through the index of each word directly,
            loss_epoch = 0.0 
            #dummy variable to accumulate loss in
            for t in range(V): # target word's index
                # extracting the embedding vectors
                wt = W[:, t]
                positive_samples = P[t]
                for p in positive_samples:
                    neg_samples = self.neg_sampling(t, positive_samples)

                    # postive and negative sample's context vector
                    cp = C[p, :]
                    C_neg = C[neg_samples, :]
                    sp = sigmoid(-np.dot(cp, wt))
                    s_neg = sigmoid(np.dot(C_neg, wt))
                   #self.Dictionary = {} #declaring an empty
                   #div_winSize = winSize/2
                    dwt = -sp*cp + np.dot(s_neg, C_neg)
                    dcp = -sp*wt
                    dC_neg = np.outer(s_neg, wt)

                    # Gradient descent update
                    #self.Dictionary[word].append(context)
                    wt -= stepsize*dwt
                    cp -= stepsize*dcp
                    C_neg -= stepsize*dC_neg
                    #reusing the previously declared sigmoid and calculating loss
                    
                    loss = -np.log(sigmoid(np.dot(cp, wt)))                             + np.sum(-np.log(sigmoid(-np.dot(C_neg, wt))))
                    loss_epoch += loss
                  

            losses.append(loss_epoch)
            #comparing outputs of the loss 
            if loss_epoch < loss_best:
                loss_best = loss_epoch
                epochs_no_improvement = 0
                # now we save the best paramaeters
                self.W = W
                self.C = C
                if save_model_path is not None:
                    self.save(save_model_path)
            else:
                epochs_no_improvement += 1
    
            fname = 'losses' + '_nEmbed' + str(self.nEmbed)                     + '_negativeRate' + str(self.negativeRate)                     + '_winSize' + str(self.winSize)                     + '_minCount' + str(self.minCount)                     + '_stepsize' + str(stepsize)

            np.save(fname + '.npy', losses)
            

            if epochs_no_improvement >= patience:
                print('EARLY STOPPING.')
                break
            

    def save(self, path):
       #declaring the columns to save the data in
        data = {'word2idx': self.word2idx,
                'W': self.W,
                'C': self.C,
                'negativeRate': self.negativeRate,
                'nEmbed': self.nEmbed,
                'winSize': self.winSize,
                'minCount': self.minCount}

        with open(path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


    def similarity(self, word1, word2):
        """
         Revising the formula from the referred research paper: 
         The cosine similarity formula vector1.vector2/(||vector1||*||vector2||)
         returns a score netween [0,1] that gives the similarity 0 being least similar and 1 bring most
        """
        # using get to get index instead of index 
        #word_index1 = self.vocabulary_list_n.index(word1)
        idx1 = self.word2idx.get(word1, 0)
        #word_index2 = self.vocabulary_list_n.index(word2)
        idx2 = self.word2idx.get(word2, 0)
        
        # using self.W instead of np.dot() used previously
        w1 = self.W[:, idx1]
        #a=np.dot(self.W_2, self.W_1[word_index1, :])
        w2 = self.W[:, idx2]
        #b=np.dot(self.W_2, self.W_1[word_index2, :])
        
        # Calculate cosine similarity score
        norm1 = np.linalg.norm(w1)
        norm2 = np.linalg.norm(w2)
        score = np.dot(w1, w2)/ (norm1 * norm2)
        #return np.sum(np.multiply(self.sigmoid(a), self.sigmoid(b))) / (np.linalg.norm(self.sigmoid(a)) * np.linalg.norm(self.sigmoid(b)))
        #return ((np.multiply(self.sigmoid(a), self.sigmoid(b))) / (np.linalg.norm(self.sigmoid(a)) * np.linalg.norm(self.sigmoid(b))))

        return score

    @staticmethod
    def load(path):
        #improving load handling compared to previous version
        #to read a binary file
        #load is for file opened for reading
        with open(path, "rb") as f: 
            data = pickle.load(f)
            #specifying the format to load the data in to maintain uniformity
        sg = SkipGram(sentences=None,
                      nEmbed=data['nEmbed'],
                      negativeRate=data['negativeRate'],
                      winSize=data['winSize'],
                      minCount=data['minCount'])
        sg.W = data['W']
        sg.C = data['C']
        sg.word2idx = data['word2idx']
        return sg


#From the Question=
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode (for submission to Prof.)', action='store_true')
    #added more detailed arguments for error handling
    parser.add_argument('--validate', help='enters validation mode (compute cross-validation with ground-truth)', action='store_true')
    parser.add_argument('--epochs', help='number of training epochs', type=int, default=100)
    parser.add_argument('--nEmbed', help='embedding dimension', type=int, default=100)
    parser.add_argument('--winSize', help='context window size', type=int, default=5)
    parser.add_argument('--stepsize', help='stepsize for gradient descent', type=float, default=0.0001)
    parser.add_argument('--negativeRate', help='ratio of negative sampling', type=int, default=5)
    parser.add_argument('--minCount', help='take into account only words with more appearances than this number', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5, help='number of epochs for early stopping: stop training if the loss has not been improved over the last N epochs')

    opts = parser.parse_args()

    if not opts.test:
        if not opts.validate:
            print('Read text to sentences')
            start = time.time()
            sentences = text2sentences(opts.text)
            print('Took', time.time() - start, '(s). Total', len(sentences), 'sentences.')

            sg = SkipGram(sentences, opts.nEmbed, opts.negativeRate, opts.winSize, opts.minCount)
            print('Start training')
            start = time.time()
            sg.train(stepsize=opts.stepsize, epochs=opts.epochs, patience=opts.patience, save_model_path=opts.model)
            print('Total training time:', time.time() - start, '(s)')
            
        else:
            print('Validation mode')
            data = pd.read_csv(opts.text, delimiter='\t')
            pairs = zip(data['word1'], data['word2'])
            sim_gt = data['similarity'].values

            sg = SkipGram.load(opts.model)
            sim_predicted = np.zeros(sim_gt.shape)
            for idx, (a,b) in enumerate(pairs):
                if (idx+1)%100 == 0:
                    print(idx + 1, '/', len(sim_predicted))
                sim_predicted[idx] = sg.similarity(a,b)
            # Compute cross-correlation
            corr = np.corrcoef(sim_gt, sim_predicted)
            print('correlation:', corr)
    else:
        pairs = loadPairs(opts.text)
        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
            print(sg.similarity(a,b))


if __name__ == '__main__':
    main()
    
#=From the Question
