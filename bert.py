import re
import numpy as np
import os
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


def get_bert_embd(linput,rinput):
    
    sents = []
    sents.append(linput)
    sents.append(rinput)

    marked_text = ["[CLS] " + sent + " [SEP]" for sent in sents]

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    tokenized_text = []
    for text in marked_text:
        tokenized_text.append(tokenizer.tokenize(text))
        

    indexed_tokens_temp = []
    for text in tokenized_text:
        indexed_tokens_temp.append(tokenizer.convert_tokens_to_ids(text))

    indexed_tokens= [a for b in indexed_tokens_temp for a in b]
    
    segments_ids=[]
    for i in range(len(indexed_tokens_temp)):
        for _ in range(len(indexed_tokens_temp[i])):
            segments_ids.append(i)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
   
    #Load pre-trained model (weights)
    modelt = BertModel.from_pretrained('bert-base-uncased')

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    modelt.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = modelt(tokens_tensor, segments_tensors)

    #This object has four dimensions, in the following order:

    #The layer number (12 layers)
    #The batch number (1 sentence)
    #The word / token number (22 tokens in our sentence)
    #The hidden unit / feature number (768 features)

    linput = encoded_layers[0][0][1:len(tokenized_text[0])-1]

    rinput = encoded_layers[0][0][len(tokenized_text[0])+1:len(tokenized_text[0])+len(tokenized_text[1])-1]

    rinput=torch.Tensor(rinput)

    linput=torch.Tensor(linput)
    
    return(linput,rinput)

