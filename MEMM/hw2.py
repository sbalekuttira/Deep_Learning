#Author: Somaiah Thimmaiah Balekuttira


import torch
from torch import autograd, nn
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import gzip
import cPickle
import subprocess
import argparse
import sys




#Embedding funtion is used to concatenate the word vector and the previous tag vector and prepare a input for the neural net
#The output of this function is the input embedding and the final tag embedding 
def embedding(train_lex,train_y,word_vectors,tag_vectors):
    
    
  
    input_embed=[] 
    input_embedding_labels=[]
    
    for i in range(len(train_lex)):
        for j in range (len(train_lex[i])):
            current_word_id=train_lex[i][j]
            current_word_label=train_y[i][j]
            input_embedding_labels.append(tag_vectors[current_word_label]) 
            if(j==0):
                    input_tag_embd=start_tag
                    input_word_embd=word_vectors[current_word_id]
                    input_tag_embd=input_tag_embd.reshape(1,tag_vector_size)
                    input_word_embd=input_word_embd.reshape(1,word_vector_size)
                    input_embed.append(np.concatenate([input_word_embd,input_tag_embd],axis=1))
                
                
            elif (j > 0) :
                input_tag_embd=tag_vectors[train_y[i][j-1]]
                input_word_embd=word_vectors[current_word_id]
                input_word_embd=input_word_embd.reshape(1,word_vector_size)
                input_tag_embd=input_tag_embd.reshape(1,tag_vector_size)
                input_embed.append(np.concatenate([input_word_embd,input_tag_embd],axis=1))
    
    input_embed=np.asarray(input_embed,dtype=np.float32)
    input_embedding_labels=np.asarray(input_embedding_labels)
    print input_embed.shape
    print input_embedding_labels.shape
    input_embedding=np.zeros(shape=(input_embed.shape[0],input_embed.shape[2]))
    print input_embedding.shape
    for i in range (input_embedding.shape[0]):
        for j in range (input_embedding.shape[1]):
            input_embedding[i][j]=input_embed[i][0][j]
    
    #print input_embedding.shape
    return input_embedding , input_embedding_labels    



#The neural_net function below takes as input the input embeddings and the target label embedding
#The output is a trained neural network model

def neural_net(input_embedding,input_embedding_labels):
    input_neurons=input_embedding.shape[1]
    hidden_neurons=400
    output_neurons=input_embedding_labels.shape[1]
    learning_r=0.001
    input_num=torch.from_numpy(np.array(input_embedding,dtype=np.float32))
    input=autograd.Variable(input_num,requires_grad=False)
   
    target_num=torch.from_numpy(np.array(input_embedding_labels,dtype=np.float32))
    target=autograd.Variable((target_num))
   
    class Net(nn.Module):
        def __init__(self, input_neurons, hidden_neurons, output_neurons):
            super(Net, self).__init__()
            self.layer_1 = nn.Linear(input_neurons, hidden_neurons) 
            self.layer_2 = nn.Linear(hidden_neurons, output_neurons)  
    
        def forward(self, x):
            x = self.layer_1(x)
            x = F.tanh(x)
            x = self.layer_2(x)
            x = F.softmax(x)
            return x
    
    net = Net(input_neurons, hidden_neurons, output_neurons)
    opt=torch.optim.Adam(params=net.parameters(),lr=learning_r)

    for epoch in range(10000):
       # print epoch
        output=net(input)
       
        loss = nn.MSELoss()
        loss_is=loss(output,target)
      #  print loss_is
        
        net.zero_grad()
        loss_is.backward()
        opt.step()
    
    
    return net




class Net(nn.Module):
    
        def __init__(self, input_neurons, hidden_neurons, output_neurons):
            super(Net, self).__init__()
            self.layer_1 = nn.Linear(input_neurons, hidden_neurons) 
            #self.relu = nn.ReLU()
            self.layer_2 = nn.Linear(hidden_neurons, output_neurons)  
    
        def forward(self, x):
            
            x = self.layer_1(x)
            x = F.tanh(x)
            x = self.layer_2(x)
            x = F.softmax(x)
            return x




def conlleval(p, g, w, filename='tempfile.txt'):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, ww in zip(sl, sp, sw):
            out += ww + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename)

def get_perf(filename):
    ''' run conlleval.pl perl script to obtain precision/recall and F1 score '''
    _conlleval = 'conlleval.pl'

    proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall    = float(out[8][:-2])
    f1score   = float(out[10])

    return (precision, recall, f1score)




#The below is the viterbi inferene function , it takes as input the trained model , test sentences, word embeddings and tag embeddings
#The output of this function is a list of the predicted labels for every sentence
def viterbi(model,test_lex,word_vectors,tag_vectors):
    
    predicted_tags=[]
    
    for sent in range (len(test_lex)):
       
        length_sent=len(test_lex[sent])+1
        num_labels = 129
        pos_start_tag = num_labels - 2
        viterbi_table = np.zeros((num_labels,length_sent))
        viterbi_table[pos_start_tag][0] = 1
        backtrack_table = np.zeros((num_labels,length_sent))
        
        
        for word in range (length_sent-1):
            
            current_word_id=test_lex[sent][word]
            probs=np.zeros(shape=(num_labels,num_labels))
            
            for k in range (num_labels):
                
                    input_tag_embed=tag_vectors[k]
                    input_word_embed=word_vectors[current_word_id]
                    input_embedding=np.concatenate([input_word_embed,input_tag_embed],axis=0)
                    input_embedding=torch.from_numpy(np.array(input_embedding,dtype=np.float32))
                    input=autograd.Variable(input_embedding,requires_grad=False)
                    output=model(input)
                    output[num_labels-2]=0
                    output[num_labels-1]=0
                
                    for h in range(num_labels):
                        
                        probs[k][h]=output.data[h]
        
                    if (word!=0):
                      
                        probs[num_labels-2]=0
                
                    probs[num_labels-1]=0
            
        
            max_probs = []
            backtrack_index = []
            word_tag_probs_table=probs 
 #Filing up viterbi table and the backtracking indexing table              
            for columns in range(num_labels):
                    
                     word_tag_probs_table[:,columns] = np.multiply(word_tag_probs_table[:,columns],viterbi_table[:,word])
                     tab_check = word_tag_probs_table[:,columns].tolist()
                     max_index = max(tab_check)
                     max_probs.append(max_index)
                     backtrack_index.append(tab_check.index(max_index))
        
            for columns in range(num_labels):
                
                    viterbi_table[columns][word+1] = max_probs[columns]
                    backtrack_table[columns][word+1] = backtrack_index[columns]
        
            

        #backtracking starts from here
        
        start_value = max(viterbi_table[:,-1])
        column_back = viterbi_table[:,-1]
        column_back = column_back.tolist()
        start_index = column_back.index(start_value)
        iterator = length_sent-1
        final_tags = []
        final_tags[:]=[]
        final_tags.append(int(start_index))
    
        while(iterator > 1):
            
            check = backtrack_table[:,iterator]
            final_tags.append(int(check[start_index]))
            start_index = int(check[start_index])
            iterator = iterator-1

        final_tags.reverse()
        #print final_tags 
        predicted_tags.append(final_tags)
    
    
    return predicted_tags




#Entry point of code starts here



with gzip.open('atis.small.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set, dicts = cPickle.load(f)

    train_lex, _, train_y = train_set
    valid_lex, _, valid_y = valid_set
    test_lex,  _,  test_y  = test_set

    idx2label = dict((k,v) for v,k in dicts['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dicts['words2idx'].iteritems())




word_vectors = np.load('word_vectors.npy')

tag_vectors=np.zeros(shape=(len(idx2label)+2,len(idx2label)+2))
np.fill_diagonal(tag_vectors, 1)
start_tag=tag_vectors[127]
end_tag=tag_vectors[128]
total_tags=tag_vectors.shape[0]

# My model is already trained and hence the below lines are commented 
#input_embedding,input_embedding_labels=embedding(train_lex,train_y,word_vectors,tag_vectors)
#net=neural_net(input_embedding,input_embedding_labels)
#torch.save(net.state_dict(), 'trained_model')

input_neurons=word_vectors.shape[1]+total_tags
hidden_neurons=400
output_neurons=total_tags
the_model=Net(input_neurons,hidden_neurons,output_neurons)
the_model.load_state_dict(torch.load('trained_model'))




predicted=viterbi(the_model,test_lex,word_vectors,tag_vectors)



predictions_test = [ map(lambda t: idx2label[t], y) for y in predicted]
groundtruth_test = [ map(lambda t: idx2label[t], y) for y in test_y]
words_test = [ map(lambda t: idx2word[t], w) for w in test_lex ]
test_precision, test_recall, test_f1score = conlleval(predictions_test, groundtruth_test, words_test)

print test_precision, test_recall, test_f1score

