import torch 
import utils
import numpy as np
import roberta 
import pickle

import model
import ParaGRU 
import SentGRU 
import WordGRU 
import train

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import cudnn

use_cuda = torch.cuda.is_available()
print("cuda: ",use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.empty_cache()
#cudnn.benchmark = True

num_rows = 20

print("\nnum_rows: \n",num_rows)

df = utils.load(num_rows,"data/csv/100_train.csv","data/text/100_train.txt")

utils.get_pickle(df,num_rows)

with open('data/pickle/processed_embed_a_'+str(num_rows)+'.pkl', 'rb') as f:
    apembd_list = pickle.load(f)
apembd_list = [[p] for p in apembd_list]

with open('data/pickle/processed_embed_b_'+str(num_rows)+'.pkl', 'rb') as f:
    bpembd_list = pickle.load(f)
bpembd_list = [[p] for p in bpembd_list]

#apembd_list, bpembd_list = apembd_list.to(device), bpembd_list.to(device)

with open('data/pickle/max_lens_'+str(num_rows)+'.pkl', 'rb') as f:
    max_lens = pickle.load(f)

#with open('processed_embed_2.pkl', 'rb') as f:
#    pembd_list = pickle.load(f)
#pembd_list = [[p] for p in pembd_list]

# dim of pembd_list: (num of rows, num of paras(1 always), num of sentences, num of words, word embeddings))

print("max_lens: ",max_lens)

max_doc_len = 1
max_para_len =  max_lens[0]
max_sent_len =  max_lens[1]

embd_dim=768
hidden_dim=100
batch_size=2
num_classes=2
num_epochs=1

print("\nbatch_size: ",batch_size)

p_model = ParaGRU.paraGRU(embd_dim, hidden_dim, batch_size)
s_model = SentGRU.sentGRU(embd_dim, hidden_dim, batch_size)
w_model = WordGRU.wordGRU(embd_dim, hidden_dim)

model = model.StackingModel(embd_dim, hidden_dim, w_model, s_model, p_model, max_doc_len, max_para_len, max_sent_len, batch_size, num_classes)
model = model.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
#inputsA, inputsB, labels, model, loss_function, optimizer, num_epochs, batch_size, max_doc_len, max_para_len, max_sent_len
model = train.traindf(device, apembd_list, bpembd_list, df['label'].tolist(), model, loss_function, optimizer, num_epochs, batch_size, embd_dim, max_doc_len, max_para_len, max_sent_len)
#model = train.traindf(pembd_list[0], pembd_list[1], [1], model, loss_function, optimizer, num_epochs, batch_size, embd_dim, max_doc_len, max_para_len, max_sent_len)
torch.save(model.state_dict(), 'data/model_'+str(num_rows))
print("train over")


