import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk
import re
import numpy as np
import os

class wordGRU(nn.Module):
    def __init__(self,  embd_dim, hidden_dim):
        super(wordGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.embd_dim = embd_dim
        self.wordgru = torch.nn.GRU(self.embd_dim, self.hidden_dim, num_layers=1, bidirectional=True)
        self.word_attention = torch.nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()
    
    #inputs: list of dim=3; dim1: list of words, dim2: batch_size, dim3: word embed
    def forward(self, inputs):
        word_inp, word_hid =  self.wordgru(inputs)
        word_inp = word_inp*self.softmax(self.tanh(self.word_attention(word_inp)))
        sent_inp = torch.sum(word_inp,dim=0)
        return sent_inp