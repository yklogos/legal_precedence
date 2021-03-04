import os
import nltk
import pickle
nltk.download('popular')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
import re
import string
import numpy as np
#from nltk.stem import PorterStemmer 
#from nltk.stem import WordNetLemmatizer 

import roberta

stop_words = set(stopwords.words('english'))

#ls = WordNetLemmatizer() 

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def load(num_rows, fp_csv, fp_txt):
    if os.path.exists(fp_csv):
        df = pd.read_csv(fp_csv).drop('Unnamed: 0', axis = 1)
        df = df.iloc[:num_rows,:]
        return df
    
    else:
        train_text = open(fp_txt,"r")
        text_list = train_text.readlines()
        num_rows=num_rows ##########
        df = pd.DataFrame(index=np.arange(num_rows),columns=['label','doc1_title','doc2_title','doc1_body','doc2_body'])
        i=0
        for text in text_list:
            split_text = text.split('|')
            df['label'][i] = split_text[0]
            df['doc1_title'][i] = split_text[1]
            df['doc2_title'][i] = split_text[2]
            df['doc1_body'][i] = split_text[3]
            df['doc2_body'][i] = split_text[4]
            i+=1
            if i==num_rows:
                break
        return df

# function to make pickle lists of embedings and labels
def get_pickle(df,num_rows):
    
    if os.path.exists('data/pickle/processed_embed_a_'+str(num_rows)+'.pkl') and os.path.exists('data/pickle/processed_embed_a_'+str(num_rows)+'.pkl') and os.path.exists('data/pickle/max_lens_'+str(num_rows)+'.pkl'):
        print("pickled files already exist")
        return
        
    else:
        ex_tokens = [str(n)+"." for n in np.arange(50)] # tokens to be excluded
        ex_tokens+= [str(n)+". " for n in np.arange(50)]
        ex_tokens+= ['S.','S. ']

        split_sentsa = split_into_sentences(df['doc1_body'])
        split_sentsb = split_into_sentences(df['doc2_body'])


        embd_lista = roberta.roberta_doc(split_sentsa)
        embd_listb = roberta.roberta_doc(split_sentsb)

        max_para_len = max(max_len(embd_lista), max_len(embd_listb))     

        print("max_para_len: ",max_para_len)

        sent_lista=[]
        for doc in embd_lista:
            sent_lista+=doc

        sent_listb=[]
        for doc in embd_listb:
            sent_listb+=doc

        max_sent_len = max(max_len(sent_lista),max_len(sent_listb))

        print("max_sent_len: ",max_sent_len)

        for i in range(len(embd_lista)):
            for j in range(len(embd_lista[i])):
                if isinstance(embd_lista[i][j],list):
                    #print("before: ",len(embd_list[i][j]))
                    #print("append part len: ",len((max_sent_len-temp_sentlen)*[[0]*768]))
                    temp_sentlen = len(embd_lista[i][j])
                    embd_lista[i][j] +=(max_sent_len-temp_sentlen)*[[0]*768]
                    #print("after: ",len(embd_list[i][j]))
                else:
                    #print("before: ",len(embd_list[i][j]))
                    temp_sentlen = len(embd_lista[i][j])
                    #print("append part len: ",len((max_sent_len-temp_sentlen)*[[0]*768]))
                    embd_lista[i][j] = embd_lista[i][j].tolist()
                    embd_lista[i][j]+=(max_sent_len-temp_sentlen)*[[0]*768]
                    #print("after: ",len(embd_list[i][j]))
            temp_paralen = len(embd_lista[i])
            embd_lista[i] +=(max_para_len-temp_paralen)*[[[0]*768]*max_sent_len]

        for i in range(len(embd_listb)):
            for j in range(len(embd_listb[i])):
                if isinstance(embd_listb[i][j],list):
                    #print("before: ",len(embd_list[i][j]))
                    #print("append part len: ",len((max_sent_len-temp_sentlen)*[[0]*768]))
                    temp_sentlen = len(embd_listb[i][j])
                    embd_listb[i][j] +=(max_sent_len-temp_sentlen)*[[0]*768]
                    #print("after: ",len(embd_list[i][j]))
                else:
                    #print("before: ",len(embd_list[i][j]))
                    temp_sentlen = len(embd_listb[i][j])
                    #print("append part len: ",len((max_sent_len-temp_sentlen)*[[0]*768]))
                    embd_listb[i][j] = embd_listb[i][j].tolist()
                    embd_listb[i][j]+=(max_sent_len-temp_sentlen)*[[0]*768]
                    #print("after: ",len(embd_list[i][j]))
            temp_paralen = len(embd_listb[i])
            embd_listb[i] +=(max_para_len-temp_paralen)*[[[0]*768]*max_sent_len]

        max_lens = [max_para_len,max_sent_len]

        print("pickling list")
        with open('data/pickle/processed_embed_a_'+str(num_rows)+'.pkl', 'wb') as f:
            pickle.dump(embd_lista, f)

        with open('data/pickle/processed_embed_b_'+str(num_rows)+'.pkl', 'wb') as f:
            pickle.dump(embd_listb, f)

        with open('data/pickle/max_lens_'+str(num_rows)+'.pkl', 'wb') as f:
            pickle.dump(max_lens, f)

        print("pickling done")


#function to split text to sentences
#input: text(str), list of tokens to bei=0 excluded
#output: list of sentences

def split_into_sentences(df):
    ex_tokens = [str(n)+"." for n in np.arange(50)] # tokens to be excluded
    ex_tokens+= [str(n)+". " for n in np.arange(50)]
    ex_tokens+= ['S.','S. ']
    split_sentences=[]
    for text in df:
        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = re.sub(prefixes,"\\1<prd>",text)
        text = re.sub(websites,"<prd>\\1",text)
        if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
        if "No." in text: text = text.replace("No.","No<prd>")
        if "Rs." in text: text = text.replace("Rs.","Rs<prd>")
        if "s." in text: text = text.replace("s.","s<prd>")
        if "S." in text: text = text.replace("S.","S<prd>")
        if "cl." in text: text = text.replace("cl.","cl<prd>")
        if "Will." in text: text = text.replace("Will.","Will<prd>")
        text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
        text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
        text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
        #if "”" in text: text = text.replace(".”","”.")
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        split_sentences.append([s for s in sentences if s not in ex_tokens])
    return split_sentences

#input: sentence(str)
def split_into_words(s):
    return s.split(" ")

#input: list of vec(not split into words)
def max_len(doc):
    max_len = -1
    for s in doc:
        if max_len<len(s):
            max_len = len(s)   
    return max_len
    
# output: tokenized sentence
def filter_tokens(tokens):
    return [w for w in tokens if w not in stopwords]
    

    
