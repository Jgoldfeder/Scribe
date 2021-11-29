import torch
import torch.nn as nn  
import models.partially_frozen_embedding as partially_frozen_embedding
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.optim as optim  
from time import time 
import torch.nn.functional as F
import os, sys
import math
from collections import defaultdict

# constants
word_vector_length = 100

pos_vector_length = 60
num_pos_tags = 32

gender_vector_length = 15
num_gender = 4

num_vector_length = 15
num_num = 6

person_vector_length = 15
num_person = 6

status_vector_length = 30
num_status = 4

tense_vector_length = 30
num_tenses = 8

prefix_vector_length = 50
num_prefixes = 8


suffix_vector_length = 10
num_suffixes = 4

suffix_gen_vector_length = 10
num_suffix_gen = 4

suffix_num_vector_length = 10
num_suffix_num = 6

suffix_person_vector_length = 10
num_suffix_person = 5

is_aramaic_vector_length = 5
num_is_aramaic = 2

binyan_vector_length = 40
num_binyanim = 8

dist_vector_length = 100
MAX_DIST = 40


#conj_type_vector_length = 30
#num_conj_type = 4


class BLSTM(nn.Module):
    #input of shape (batch,seq_len , input_size)
    def __init__(self,num_classes,weights, lr = 0.001, weight_decay = 0.00001, dropout = 0.5):
        super(BLSTM,self).__init__()
        
        # init word embeddings
        self.word_embedding = partially_frozen_embedding.PartiallyFrozenEmbedding(weights,5) # five special tokens in vocab that we need to train
        # init POS embeddings
        self.pos_embedding = nn.Embedding(num_pos_tags, pos_vector_length)
        # init gender embeddings        
        self.gender_embedding = nn.Embedding(num_gender, gender_vector_length)
        # init binyan embeddings
        self.binyan_embedding = nn.Embedding(num_binyanim, binyan_vector_length)
        # init num embedding
        self.num_embedding = nn.Embedding(num_num, num_vector_length)
        # init person embedding
        self.person_embedding = nn.Embedding(num_person, person_vector_length)
        # init tense embedding
        self.tense_embedding = nn.Embedding(num_tenses, tense_vector_length)
        # init is_aramaic embedding
        self.is_aramaic_embedding = nn.Embedding(num_is_aramaic, is_aramaic_vector_length)
        # init status embedding
        self.status_embedding = nn.Embedding(num_status, status_vector_length)
        # init prefix embedding
        self.prefix_embedding = nn.Embedding(num_prefixes, prefix_vector_length)
        # init suffix embeddings        
        self.suffix_embedding = nn.Embedding(num_suffixes, suffix_vector_length)
        self.suffix_person_embedding = nn.Embedding(num_suffix_person, suffix_person_vector_length)
        self.suffix_gender_embedding = nn.Embedding(num_suffix_gen, suffix_gen_vector_length)
        self.suffix_num_embedding = nn.Embedding(num_suffix_num, suffix_num_vector_length)
 
        self.embeddings = [
            self.gender_embedding,
            self.pos_embedding,
            self.num_embedding ,
            self.person_embedding,
            self.status_embedding,
            self.tense_embedding,
            self.prefix_embedding,
            self.suffix_embedding,
            self.suffix_gender_embedding,
            self.suffix_num_embedding,
            self.suffix_person_embedding,
            self.is_aramaic_embedding,
            self.binyan_embedding,
            self.word_embedding,
        ]
        # init dist embeddings
        self.dist_embedding = nn.Embedding(MAX_DIST, dist_vector_length)

        self.sum_length = word_vector_length + pos_vector_length+gender_vector_length+binyan_vector_length+num_vector_length+person_vector_length+tense_vector_length+is_aramaic_vector_length+status_vector_length+prefix_vector_length+suffix_vector_length+suffix_person_vector_length+suffix_gen_vector_length+suffix_num_vector_length

        
        # init LSTM layer
        self.lstm = nn.LSTM(self.sum_length,self.sum_length,num_layers=2,batch_first = True,bidirectional = True,dropout = 0.5)
        
        # init fully connected layers
        self.fc1 = nn.Linear((self.sum_length)*2 + dist_vector_length,self.sum_length*2)
        self.fc2 = nn.Linear(self.sum_length*2,self.sum_length*2)     
        self.fc3 = nn.Linear(self.sum_length*2,num_classes)        
        
        # init training parameters
        self.optimizer = optim.Adam(self.parameters(),lr=lr,weight_decay = weight_decay)
        self.dropout = nn.Dropout(p=dropout)
        
        self.num_classes = num_classes
        
        # set default device to cpu
        self.device = torch.device("cpu")

    def _compute_epoch_stats(self, p, o,classes):
        arr=[]
        for i in range(classes):
            arr.append(i)
        p = p.argmax(2)

        prediction = p.flatten()
        truth = o.flatten()

        cf = confusion_matrix(prediction.cpu(),truth.cpu(),arr)
        acc = accuracy_score(prediction.cpu(),truth.cpu())

        return cf,acc   
      
    def to(self, device):
        # override this method so that the custom embedding layer is included
        super().to(device)
        self.word_embedding.to(device)
        self.device = device
        
    def forward(self,x,hidden,cell):
        #word_idx, pos_idx,gender_idx,binyan_idx,number_idx,person_idx,tense_idx,conj_type_idx,status_idx,prefix_idx,suffix_idx,suffix_gen_idx,suffix_num_idx,suffix_person_idx, dist = torch.split(x, [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dim=2)

        # split input into gender, pos, number, person, status, tense, prefix, suffix_function, suffix_gender, suffix_number, suffix_person, is_aramaic, binyan, word, dist        
        indices = torch.split(x, [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dim=2)
        dist = indices[-1]
        indices = indices[:-1]
        dist = torch.clamp(dist, min= 0, max=MAX_DIST-1)

        # get embeddings
        #word_embedding = self.word_embedding.embed(word_idx).squeeze(2)
        #pos_embedding = self.pos_embedding(pos_idx).squeeze(2)        
        #binyan_embedding = self.binyan_embedding(binyan_idx).squeeze(2)        
        #prefix_embedding = self.prefix_embedding(prefix_idx).squeeze(2)        
        #suffix_embedding = self.suffix_embedding(suffix_idx).squeeze(2)        
        #gender_embedding = self.gender_embedding(gender_idx).squeeze(2)        
        #suffix_person_embedding = self.suffix_person_embedding(suffix_person_idx).squeeze(2)        
        #suffix_num_embedding = self.suffix_num_embedding(suffix_num_idx).squeeze(2)        
        #suffix_gender_embedding = self.suffix_gender_embedding(suffix_gen_idx).squeeze(2)        

        # combine POS and word embeddings
        #x = torch.cat((word_embedding,pos_embedding,binyan_embedding,prefix_embedding,suffix_embedding,suffix_person_embedding, #suffix_num_embedding,suffix_gender_embedding,gender_embedding), dim=2)        
        
        embeddings = []
        for i in range(len(indices)):
            idx = indices[i]
            embedding = self.embeddings[i]
            if i == 13:
                embeddings.append(embedding.embed(idx).squeeze(2))                    
            else:
                embeddings.append(embedding(idx).squeeze(2))    
        
        x = torch.cat(embeddings, dim=2)        
        
        # run theough BLSTM
        x , _ = self.lstm(x,(hidden,cell))
               
        # get dist embedding    
        dist_embedding = self.dist_embedding(dist).squeeze(2)
        
        # combine BLSTM output with dist embedding        
        x = torch.cat((x,dist_embedding), dim=2)

        # run through feedforward network
        x = self.dropout(F.relu(self.fc1(x.squeeze(0))))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def _log(self, x):
        # log function which returns -inf when x=0
        if x == 0:
            return float('-inf')
        return math.log(x)
    
    def decode(self, x, known_tags = None, coefficient = 1):
        # known_tags is a dict mapping position in input to a known tag for that position (either 0 or 1)
        # coefficient is what we use to increment the position. 1 leads to inference based on training data, 
        # a number less than one leads to larger sentences, and more than one to smaller sentences
        #
        # when decode is called, we are in using the model for inference
        # this means we take in for each word just the word and pos, and must estimate dist
        # we return for each possible dist, P(sentence ends | word, pos, dist)       
        # Thus, if decode is true, this means we are given a single sequence with no batch
        # simply replace the batch with MAX_DIST, and have each row in the batch be a dist
        self.eval()
        with torch.no_grad():
            
            if known_tags is None:
                known_tags = {}
                
            x = x.to(self.device)
            
            batches = []
            for dist in range(MAX_DIST):
                d = torch.zeros(x.shape[0],1).long().to(self.device)
                d += int(round(dist * coefficient))
                c= torch.cat((x,d),dim=1)
                batches.append(c)
            x = torch.stack(batches)
            
            print(x.shape)

            hidden,cell = self.init_hidden(MAX_DIST)
            
            # gives the probability of transitioning to state 0
            m = nn.Sigmoid()
            probabilities = m(self.forward(x, hidden, cell))
            probabilities=torch.transpose(probabilities,0,1)

            print(probabilities.shape)
            
            # run viterbi algorithm
            table = defaultdict(lambda: defaultdict(lambda:float('-inf')))
            backtrack = defaultdict(lambda: defaultdict(lambda:float('-inf')))
            
            table[0][0] = 1
            for timestep in range(1,x.shape[1]):
                for new_state in range(MAX_DIST):
                
                    # incorporate knowledge of known tags
                    if (timestep-1) in known_tags:
                        if known_tags[timestep-1] == 0: # we know this timestep should not be a period
                            if new_state == 0:
                                continue
                        if known_tags[timestep-1] == 1: # we know this timestep should be a period
                            if new_state != 0:
                                continue 
                            
                    if new_state == 0:
                        transitions = []
                        for prev_state in range(MAX_DIST):
                            transitions.append( self._log(probabilities[timestep-1][prev_state][1]) + table[timestep-1][prev_state])
                        table[timestep][0] = max(transitions)
                        backtrack[timestep][0] = np.argmax(transitions)
                    else:
                        table[timestep][new_state] = self._log(probabilities[timestep-1][new_state-1][0]) + table[timestep-1][new_state-1]
                        backtrack[timestep][new_state] = new_state-1
            
            sequence = []
            # find max prob sequence
            values = []
            for state in range(MAX_DIST):
                values.append(table[x.shape[1]-1][state])
            max_state = np.argmax(values)
            sequence.append(max_state)
            for timestep in range(x.shape[1]-1,0,-1):
                max_state = backtrack[timestep][max_state]
                sequence.append(max_state)
            sequence.reverse()
            return sequence
        
    def init_hidden(self,batch_size):
        h0 = torch.zeros(4 , batch_size, self.sum_length).to(self.device)
        c0 = torch.zeros(4 , batch_size, self.sum_length).to(self.device)
        return h0,c0

    def load(self,file):
        try:
            self.load_state_dict(torch.load("trained_models"+os.sep+file))
            print("loaded state from ",file)
        except:
            print(sys.exc_info()[0])
            print("Can't find dict file ",file)
     
    def save(self,file):
        if not os.path.exists('trained_models'):
            os.makedirs('trained_models')
        torch.save(self.state_dict(),"trained_models"+os.sep+file)
    
    def validate(self,X,Y,batch_size, class_weights = None):
        self.eval()
        with torch.no_grad():
            epoch_loss = 0

            if class_weights is None:
                criterion = nn.CrossEntropyLoss()  
            else:
                criterion = nn.CrossEntropyLoss(class_weights.to(self.device).float())    

            seq_len = X.shape[1]
            
            batch_count = 0    
            
            cf = np.zeros((self.num_classes,self.num_classes))
            acc = 0

            permutation = torch.randperm(X.shape[0])   
            
            # run through batches
            for index in range(0,X.shape[0], batch_size):
                self.zero_grad()

                indices = permutation[index:index+batch_size]
                i, o = torch.tensor(X[indices]), torch.tensor(Y[indices])        

                h0,c0 = self.init_hidden(i.size(0)) 

                out = self(i.long().to(self.device),h0,c0)

                target = o.long().to(self.device)       
                loss = criterion(out.view(seq_len*out.shape[0],self.num_classes),target.view(seq_len*target.shape[0]))
                epoch_loss +=loss.item()

                
                batch_count+=1
                cf_b, acc_b = self._compute_epoch_stats(out,o,self.num_classes)
                
                cf += cf_b
                acc += acc_b
            self.train()
            return acc/batch_count,  epoch_loss/batch_count, cf/cf.sum()               
            

    
    def fit(self,EPOCHS,batch_size,X,Y, X_test = None, Y_test = None, class_weights = None):
        self.train()

        if class_weights is None:
            criterion = nn.CrossEntropyLoss()  
        else:
            criterion = nn.CrossEntropyLoss(class_weights.to(self.device).float())    

        print("training")
        print("epochs:",EPOCHS)
        print("Batch:",batch_size) 

        seq_len = X.shape[1]
        

        for epoch in range(EPOCHS):
            epoch_loss = 0
            batch_count = 0    
            
            # initialize confusion matrix and accuracy score for the epoch
            cf = np.zeros((self.num_classes,self.num_classes))
            acc = 0

            start_epoch = int(time() * 1000)
            
            permutation = torch.randperm(X.shape[0])
            
            # run through batches
            for index in range(0,X.shape[0], batch_size):
                self.zero_grad()

                indices = permutation[index:index+batch_size]
                i, o = torch.tensor(X[indices]), torch.tensor(Y[indices])        

                h0,c0 = self.init_hidden(i.size(0)) 

                out = self(i.long().to(self.device),h0,c0)

                target = o.long().to(self.device)

                loss = criterion(out.view(seq_len*out.shape[0],self.num_classes),target.view(seq_len*target.shape[0]))
                loss.backward()
                #nn.utils.clip_grad_value_(self.parameters(), clip_value=1.0)

                self.optimizer.step()

                batch_count+=1
                epoch_loss +=loss.item()
                cf_b, acc_b = self._compute_epoch_stats(out,o,self.num_classes)
                
                cf += cf_b
                acc += acc_b
            
            acc = acc/batch_count
            cf = cf/cf.sum()
            # if validation data is provided, validate
            if X_test is not None and Y_test is not None:
                v_acc, v_loss, v_cf = self.validate(X_test, Y_test, batch_size, class_weights)
            end_epoch = int(time() * 1000)
            print("********************************")    
            print("epoch:",epoch,":loss:",epoch_loss/batch_count, ":acc:", acc)
            print("confusion:")
            print(cf) 
            if X_test is not None and Y_test is not None:
                print("validation loss:",v_loss,":validation acc:", v_acc)
                print("validation confusion")
                print(v_cf)
            print("time:",end_epoch-start_epoch)
                    










class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class Transformer(nn.Module):
    #input of shape (batch,seq_len , input_size)
    def __init__(self,num_classes,weights, lr = 0.001, weight_decay = 0.00001, dropout = 0.5):
        super(Transformer,self).__init__()
        
        # init word embeddings
        self.word_embedding = partially_frozen_embedding.PartiallyFrozenEmbedding(weights,5) # five special tokens in vocab that we need to train
        # init POS embeddings
        self.pos_embedding = nn.Embedding(num_pos_tags, pos_vector_length)
        # init gender embeddings        
        self.gender_embedding = nn.Embedding(num_gender, gender_vector_length)
        # init binyan embeddings
        self.binyan_embedding = nn.Embedding(num_binyanim, binyan_vector_length)
        # init num embedding
        self.num_embedding = nn.Embedding(num_num, num_vector_length)
        # init person embedding
        self.person_embedding = nn.Embedding(num_person, person_vector_length)
        # init tense embedding
        self.tense_embedding = nn.Embedding(num_tenses, tense_vector_length)
        # init is_aramaic embedding
        self.is_aramaic_embedding = nn.Embedding(num_is_aramaic, is_aramaic_vector_length)
        # init status embedding
        self.status_embedding = nn.Embedding(num_status, status_vector_length)
        # init prefix embedding
        self.prefix_embedding = nn.Embedding(num_prefixes, prefix_vector_length)
        # init suffix embeddings        
        self.suffix_embedding = nn.Embedding(num_suffixes, suffix_vector_length)
        self.suffix_person_embedding = nn.Embedding(num_suffix_person, suffix_person_vector_length)
        self.suffix_gender_embedding = nn.Embedding(num_suffix_gen, suffix_gen_vector_length)
        self.suffix_num_embedding = nn.Embedding(num_suffix_num, suffix_num_vector_length)
 
        self.embeddings = [
            self.gender_embedding,
            self.pos_embedding,
            self.num_embedding ,
            self.person_embedding,
            self.status_embedding,
            self.tense_embedding,
            self.prefix_embedding,
            self.suffix_embedding,
            self.suffix_gender_embedding,
            self.suffix_num_embedding,
            self.suffix_person_embedding,
            self.is_aramaic_embedding,
            self.binyan_embedding,
            self.word_embedding,
        ]
        # init dist embeddings
        self.dist_embedding = nn.Embedding(MAX_DIST, dist_vector_length)

        self.sum_length = word_vector_length + pos_vector_length+gender_vector_length+binyan_vector_length+num_vector_length+person_vector_length+tense_vector_length+is_aramaic_vector_length+status_vector_length+prefix_vector_length+suffix_vector_length+suffix_person_vector_length+suffix_gen_vector_length+suffix_num_vector_length

        
        # init Transformer layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.sum_length, nhead=4,dropout = 0.1)
        encoder_norm = nn.LayerNorm(self.sum_length)
        num_layers = 1
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers,encoder_norm)
        self.positional_encoder = PositionalEncoding(d_model=self.sum_length)
        # init fully connected layers
        self.fc1 = nn.Linear((self.sum_length) + dist_vector_length,self.sum_length)
        self.fc2 = nn.Linear(self.sum_length,self.sum_length)     
        self.fc3 = nn.Linear(self.sum_length,num_classes)        
        
        # init training parameters
        self.optimizer = optim.Adam(self.parameters(),lr=lr,weight_decay = weight_decay)
        self.dropout = nn.Dropout(p=dropout)
        
        self.num_classes = num_classes
        
        # set default device to cpu
        self.device = torch.device("cpu")

    def _compute_epoch_stats(self, p, o,classes):
        arr=[]
        for i in range(classes):
            arr.append(i)
        p = p.argmax(2)

        prediction = p.flatten()
        truth = o.flatten()

        cf = confusion_matrix(prediction.cpu(),truth.cpu(),arr)
        acc = accuracy_score(prediction.cpu(),truth.cpu())

        return cf,acc   
      
    def to(self, device):
        # override this method so that the custom embedding layer is included
        super().to(device)
        self.word_embedding.to(device)
        self.device = device
        
    def forward(self,x,hidden,cell):
        # split input into gender, pos, number, person, status, tense, prefix, suffix_function, suffix_gender, suffix_number, suffix_person, is_aramaic, binyan, word, dist        
        indices = torch.split(x, [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], dim=2)
        dist = indices[-1]
        indices = indices[:-1]
        dist = torch.clamp(dist, min= 0, max=MAX_DIST-1)

        # get embeddings

        embeddings = []
        for i in range(len(indices)):
            idx = indices[i]
            embedding = self.embeddings[i]
            if i == 13:
                embeddings.append(embedding.embed(idx).squeeze(2))                    
            else:
                embeddings.append(embedding(idx).squeeze(2))    
        
        x = torch.cat(embeddings, dim=2)        

        x = x.permute(1, 0, 2)

        # add pos encodings
        x = self.positional_encoder(x)
        # run through transformer_encoder
        x = self.transformer_encoder(x)
        
        x = x.permute(1, 0, 2)
        # get dist embedding    
        dist_embedding = self.dist_embedding(dist).squeeze(2)
        
        # combine transformer_encoder output with dist embedding        
        x = torch.cat((x,dist_embedding), dim=2)
        
        # run through feedforward network
        x = self.dropout(F.relu(self.fc1(x.squeeze(0))))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
    
    def _log(self, x):
        # log function which returns -inf when x=0
        if x == 0:
            return float('-inf')
        return math.log(x)
    
    def decode(self, x, known_tags = None, coefficient = 1):
        # known_tags is a dict mapping position in input to a known tag for that position (either 0 or 1)
        # coefficient is what we use to increment the position. 1 leads to inference based on training data, 
        # a number less than one leads to larger sentences, and more than one to smaller sentences
        #
        # when decode is called, we are in using the model for inference
        # this means we take in for each word just the word and pos, and must estimate dist
        # we return for each possible dist, P(sentence ends | word, pos, dist)       
        # Thus, if decode is true, this means we are given a single sequence with no batch
        # simply replace the batch with MAX_DIST, and have each row in the batch be a dist
        self.eval()
        with torch.no_grad():
            
            if known_tags is None:
                known_tags = {}
                
            x = x.to(self.device)
            
            batches = []
            for dist in range(MAX_DIST):
                d = torch.zeros(x.shape[0],1).long().to(self.device)
                d += int(round(dist * coefficient))
                c= torch.cat((x,d),dim=1)
                batches.append(c)
            x = torch.stack(batches)
            
            print(x.shape)

            hidden,cell = self.init_hidden(MAX_DIST)
            
            # gives the probability of transitioning to state 0
            m = nn.Sigmoid()
            probabilities = m(self.forward(x, hidden, cell))
            probabilities=torch.transpose(probabilities,0,1)

            print(probabilities.shape)
            
            # run viterbi algorithm
            table = defaultdict(lambda: defaultdict(lambda:float('-inf')))
            backtrack = defaultdict(lambda: defaultdict(lambda:float('-inf')))
            
            table[0][0] = 1
            for timestep in range(1,x.shape[1]):
                for new_state in range(MAX_DIST):
                
                    # incorporate knowledge of known tags
                    if (timestep-1) in known_tags:
                        if known_tags[timestep-1] == 0: # we know this timestep should not be a period
                            if new_state == 0:
                                continue
                        if known_tags[timestep-1] == 1: # we know this timestep should be a period
                            if new_state != 0:
                                continue 
                            
                    if new_state == 0:
                        transitions = []
                        for prev_state in range(MAX_DIST):
                            transitions.append( self._log(probabilities[timestep-1][prev_state][1]) + table[timestep-1][prev_state])
                        table[timestep][0] = max(transitions)
                        backtrack[timestep][0] = np.argmax(transitions)
                    else:
                        table[timestep][new_state] = self._log(probabilities[timestep-1][new_state-1][0]) + table[timestep-1][new_state-1]
                        backtrack[timestep][new_state] = new_state-1
            
            sequence = []
            # find max prob sequence
            values = []
            for state in range(MAX_DIST):
                values.append(table[x.shape[1]-1][state])
            max_state = np.argmax(values)
            sequence.append(max_state)
            for timestep in range(x.shape[1]-1,0,-1):
                max_state = backtrack[timestep][max_state]
                sequence.append(max_state)
            sequence.reverse()
            return sequence
        
    def init_hidden(self,batch_size):
        h0 = torch.zeros(4 , batch_size, self.sum_length).to(self.device)
        c0 = torch.zeros(4 , batch_size, self.sum_length).to(self.device)
        return h0,c0

    def load(self,file):
        try:
            self.load_state_dict(torch.load("trained_models"+os.sep+file))
            print("loaded state from ",file)
        except:
            print(sys.exc_info()[0])
            print("Can't find dict file ",file)
     
    def save(self,file):
        if not os.path.exists('trained_models'):
            os.makedirs('trained_models')
        torch.save(self.state_dict(),"trained_models"+os.sep+file)
    
    def validate(self,X,Y,batch_size, class_weights = None):
        self.eval()
        with torch.no_grad():
            epoch_loss = 0

            if class_weights is None:
                criterion = nn.CrossEntropyLoss()  
            else:
                criterion = nn.CrossEntropyLoss(class_weights.to(self.device).float())    

            seq_len = X.shape[1]
            
            batch_count = 0    
            
            cf = np.zeros((self.num_classes,self.num_classes))
            acc = 0

            permutation = torch.randperm(X.shape[0])   
            
            # run through batches
            for index in range(0,X.shape[0], batch_size):
                self.zero_grad()

                indices = permutation[index:index+batch_size]
                i, o = torch.tensor(X[indices]), torch.tensor(Y[indices])        

                h0,c0 = self.init_hidden(i.size(0)) 

                out = self(i.long().to(self.device),h0,c0)

                target = o.long().to(self.device)       
                loss = criterion(out.view(seq_len*out.shape[0],self.num_classes),target.view(seq_len*target.shape[0]))
                epoch_loss +=loss.item()

                
                batch_count+=1
                cf_b, acc_b = self._compute_epoch_stats(out,o,self.num_classes)
                
                cf += cf_b
                acc += acc_b
            self.train()
            return acc/batch_count,  epoch_loss/batch_count, cf/cf.sum()               
            

    
    def fit(self,EPOCHS,batch_size,X,Y, X_test = None, Y_test = None, class_weights = None):
        self.train()

        if class_weights is None:
            criterion = nn.CrossEntropyLoss()  
        else:
            criterion = nn.CrossEntropyLoss(class_weights.to(self.device).float())    

        print("training")
        print("epochs:",EPOCHS)
        print("Batch:",batch_size) 

        seq_len = X.shape[1]
        

        for epoch in range(EPOCHS):
            epoch_loss = 0
            batch_count = 0    
            
            # initialize confusion matrix and accuracy score for the epoch
            cf = np.zeros((self.num_classes,self.num_classes))
            acc = 0

            start_epoch = int(time() * 1000)
            
            permutation = torch.randperm(X.shape[0])
            
            # run through batches
            for index in range(0,X.shape[0], batch_size):
                self.zero_grad()

                indices = permutation[index:index+batch_size]
                i, o = torch.tensor(X[indices]), torch.tensor(Y[indices])        

                h0,c0 = self.init_hidden(i.size(0)) 

                out = self(i.long().to(self.device),h0,c0)

                target = o.long().to(self.device)

                loss = criterion(out.view(seq_len*out.shape[0],self.num_classes),target.view(seq_len*target.shape[0]))
                loss.backward()
                #nn.utils.clip_grad_value_(self.parameters(), clip_value=1.0)

                self.optimizer.step()

                batch_count+=1
                epoch_loss +=loss.item()
                cf_b, acc_b = self._compute_epoch_stats(out,o,self.num_classes)
                
                cf += cf_b
                acc += acc_b
            
            acc = acc/batch_count
            cf = cf/cf.sum()
            # if validation data is provided, validate
            if X_test is not None and Y_test is not None:
                v_acc, v_loss, v_cf = self.validate(X_test, Y_test, batch_size, class_weights)
            end_epoch = int(time() * 1000)
            print("********************************")    
            print("epoch:",epoch,":loss:",epoch_loss/batch_count, ":acc:", acc)
            print("confusion:")
            print(cf) 
            if X_test is not None and Y_test is not None:
                print("validation loss:",v_loss,":validation acc:", v_acc)
                print("validation confusion")
                print(v_cf)
            print("time:",end_epoch-start_epoch)
                    
  