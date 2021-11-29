import numpy as np
import data_prep.util as util
import os 
import torch  

def processLine(line):
    tokens = line.split()
    word = tokens[0]
    embedding = []
    for i in range(1,len(tokens)):
        embedding.append(float(tokens[i]))
    return word, torch.tensor(embedding)
    

def load_vocab():
    # first try loading from pre-built files. This is much faster
    try:
        embeddings = util.load_tensor("embeddings.pt")
        vocab = util.load_obj("vocab.pkl")
        word_list = util.load_obj("word_list.pkl")
        
    except:
        embeddings = []
        vocab = dict()
        word_list = []
        with open("resources"+os.sep+"vecs_RabbinicTexts_f.txt", 'r', encoding='utf-8') as infile:
            for index,line in enumerate(infile):
                if index==0:
                    continue
                word, embedding = processLine(line) 
                vocab[word] = index-1
                word_list.append(word)
                embeddings.append(embedding)
        embeddings = torch.stack(embeddings)
        
        # add <UNK> token
        vocab["<UNK>"] = len(word_list)
        word_list.append("<UNK>")
        
        # add <COMMA> token
        vocab["<COMMA>"] = len(word_list)
        word_list.append("<COMMA>")

        # add <STOP> token
        vocab["<STOP>"] = len(word_list)
        word_list.append("<STOP>")     

        # add <START> token
        vocab["<START>"] = len(word_list)
        word_list.append("<START>")             
        
        # add <QUES> token, which is <STOP>, but a question (ie ?)
        vocab["<QUES>"] = len(word_list)
        word_list.append("<QUES>")
        
        util.save_tensor(embeddings, "embeddings.pt")
        util.save_obj(vocab,"vocab.pkl")    
        util.save_obj(word_list,"word_list.pkl")
        
    return embeddings, vocab, word_list
