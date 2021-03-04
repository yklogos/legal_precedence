import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk
import re
import numpy as np
import os

class Sent_WordGRU(nn.Module):
    def __init__(self, embd_dim, hidden_dim):
        super(Sent_WordGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.embd_dim = embd_dim
        self.wordgru = torch.nn.GRU(self.embd_dim, self.hidden_dim, num_layers=1, bidirectional=True)
        self.word_attention = torch.nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()

    def forward(self, inputs):
        word_inp, word_hidden = self.wordgru(inputs)
        word_inp = word_inp*self.softmax(self.tanh(self.word_attention(word_inp)))
        sent_hid = torch.sum(word_inp,dim=0)
        # sent_hid (dim: batch_size, 2*hidden_dim)
        return sent_hid

    
class sentGRU(nn.Module):
    def __init__(self,  embd_dim, hidden_dim, batch_size):
        super(sentGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        #self.max_para_len = max_para_len
        self.embd_dim = embd_dim
        self.wordgru = Sent_WordGRU(self.embd_dim, self.hidden_dim)
        self.sentgru = torch.nn.GRU(2*self.hidden_dim, self.hidden_dim, num_layers=1, bidirectional=True)
        self.sent_attention = torch.nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()
    
    #inputs: list of dim=4; dim=1: list of sentences, dim2: list of words, dim3: batch_size, dim4: word embed
    def forward(self, inputs):
        tsent_hid=[]
        for sent in inputs:
            sent_hid_indv = self.wordgru(sent)
            sent_hid_indv = sent_hid_indv.unsqueeze(0)
            tsent_hid.append(sent_hid_indv)
        #sent_hid = torch.tensor(sent_hid)
        sent_hid = torch.Tensor(len(tsent_hid), self.batch_size, 2*self.hidden_dim)
        sent_hid = torch.cat(tsent_hid, dim=0)
        para_inp, para_hidden = self.sentgru(sent_hid)
        para_inp = para_inp*self.softmax(self.tanh(self.sent_attention(para_inp)))
        para_inp = torch.sum(para_inp,dim=0) 
        return para_inp