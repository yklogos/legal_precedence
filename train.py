import os
import torch
import pickle

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm_notebook


def traindf(device, inputsA, inputsB, labels, model, loss_function, optimizer, num_epochs, batch_size, embd_dim, max_doc_len, max_para_len, max_sent_len):
    model.train()
    total_loss=0
    log_file = open("logs/trainer_log.txt","w")
    log_file.write("trainer file:\n\n")
    log_file.close()
    for epoch in range(int(num_epochs)):
        for i in range(int(len(inputsA)/batch_size)):
            model.zero_grad()
            minpA = torch.tensor(inputsA[i*batch_size:(i+1)*batch_size]).reshape(max_doc_len, max_para_len, max_sent_len, batch_size, embd_dim)
            minpB = torch.tensor(inputsB[i*batch_size:(i+1)*batch_size]).reshape(max_doc_len, max_para_len, max_sent_len, batch_size, embd_dim)
            # dim pred: batch_size, num_classes
            minpA, minpB = minpA.to(device), minpB.to(device)
            pred = model(minpA, minpB)
            log_file = open("logs/trainer_log.txt","a")
            log_file.write("pred : "+str(pred)+"\n")
            log_file.close()            
            loss = loss_function(pred, torch.tensor(labels[i*batch_size:(i+1)*batch_size]))
            total_loss+=loss
            log_file = open("logs/trainer_log.txt","a")
            log_file.write("average loss after "+str(epoch)+" epochs and "+str(i)+" batches is "+str(float(loss/batch_size))+"\n\n")
            log_file.close()
            loss.backward(retain_graph=True)
            optimizer.step()
    #print("total loss after "+str(i)+" batches is "+str(total_loss))
    return model