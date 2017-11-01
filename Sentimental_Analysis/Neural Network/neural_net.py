
#Author : Somaiah Thimmaiah Balekuttira (4-oct-2017)


import numpy as np
from nltk.corpus import stopwords
import re






#feature_extraction is used to extract the features. Using a unigram model with bag of words
def feature_extraction():
    
    #shuffle_data is used to shuffle the data after splitting the data into test and train 
    
    
    
    #parse is used to parse the sentences into individual words
    def parse(sentence):
        return re.compile('\w+').findall(sentence)
   
    train_sentences=[]
    labels=[]
    bagofwords=[]
    sentences=[]
    
    #add the file name of the training data below
    train_sen=open('sentences.txt','r')
    
    #add the file name of the training labels data below

    train_label=open('labels.txt','r')	    

    #add the file name of the testing data below
    test_sen=open('test_sentences.txt','r')

    
    #add the file name of the testing labels below
    lab=open('test_labels.txt','r')
    
    for sentence in test_sen:
        sentences.append(parse(sentence))
    
    for sentence in train_sen:
        train_sentences.append(parse(sentence))
     
    for label in lab:
        labels.append(label.strip())
        
    labels=np.matrix(labels).astype(int)
    labels=np.matrix.transpose(labels)
    

    #making bag of words from the training data (sentences.txt)
    for i in range(len(train_sentences)):
        for word in train_sentences[i]:
            if word not in stopwords.words("english"):
                bagofwords.append(word)
    bagofwords=set(bagofwords)

    word2int={}
    for i,word in enumerate(bagofwords):
           word2int[word] = i

   #making feature vector matrix     
    w, h = len(bagofwords), len(sentences)

    Matrix = [[0 for x in range(w)] for y in range(h)] 

    for i in range(len(sentences)): 
        for word in bagofwords: 
            if word in sentences[i]: 
                    Matrix[i][word2int[word]]=sentences[i].count(word)
    
    test_features=np.array(Matrix)
    test_labels=np.array(labels) 
    
    return test_features,test_labels
    
    
    






#This function shuffles and splits data into testing and training 
def split_data():
    def shuffle_data(matrix, labels):
        assert len(matrix) == len(labels)
        q = np.random.permutation(len(matrix))
        return matrix[q], labels[q]
    
    features,labels=shuffle_data(feat,labels)
    
    #split 70% train 30% test
    index=int(len(features)*0.7)
    training_features, test_features = features[:index,:], features[index:,:]
    training_labels, test_labels=labels[:index,:], labels[index:,:]





#evalution function 
def evaluate(preds, golds):
    tp, pp, cp = 0.0, 0.0, 0.0
    for pred, gold in zip(preds, golds):
        if pred == 1:
            pp += 1
        if gold == 1:
            cp += 1
        if pred == 1 and gold == 1:
            tp += 1
    precision = tp / pp
    recall = tp / cp
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)






#This function trains the model and returns the input and output layer weights
def neural_net_trainweights(training_features,training_labels):
    #myfucntion is the sigmoid activation function
    def myfunction( x ):
        return (1/(1+np.exp(-x)))

    #derivative is the derivation of the sigmoid activation function
    def derivative(x):
        return (x*(1-x))
    
    #hidden will have 4 neurons
    firstlayer_neurons=4
    
    #output layer will have 1 neuron
    output_neurons=1
    
    
    #initilising weights to random between -1 and 1 for input_layer weights
    input_weights=2*np.random.random((training_features.shape[1],firstlayer_neurons))-1
    
    #initilising weights to random between -1 and 1 for output_layer weights
    output_weights=2*np.random.random((l1_weights.shape[1],output_neurons))-1
    
    epoch=0
    while epoch!=200:
     #forward propogation
    #input layer
        input_layer = training_features
    
    #first layer
        z2 = np.dot(input_layer,input_weights)
    
    #second layer after activation 
        a2 = myfunction(z2)
    
    #third layer - output layer
        z3=np.dot(a2,output_weights)
    
    #output_layer after activation    
        output_layer=myfunction(z3)

    
        output_error= training_labels - output_layer
    
    #convert to array
        output_layer=np.array(output_layer)
        output_error=np.array(output_error)
    
    
    
       #back propogation code
        output_delta=output_error*derivative(output_layer)
    
        input_error=output_delta.dot(output_weights.T)
    
        a2=np.array(a2)
        input_error=np.array(input_error)
    
        input_delta=input_error*derivative(a2)
   
    
        output_weights += a2.T.dot(output_delta)
        input_weights += training_features.T.dot(input_delta)
        
    
    
        epoch=epoch+1
    #saving the input layer (hidden layer) weights in the file l1_weights.npy    
    np.save("l1_weights.npy",input_weights)     
    
    #saving the output layer weights in the file output_weights.npy
    np.save("output_weights.npy",output_weights)     
    
    
    #return input layer (hidden layer) and output layer weights for the trained model
    return input_weights,output_weights





#predict_neural does forward propogation to predict labels of unseen testing data

def predict_neural(test_features,input_weights,output_weights):
    predicted=[]
   #myfucntion is the sigmoid activation function
    def myfunction( x ):
        return (1/(1+np.exp(-x)))

    #derivative is the derivation of the sigmoid activation function
    def derivative(x):
        return (x*(1-x))
    
    #input layer
    input_layer = test_features
    
    #first layer
    z2 = np.dot(input_layer,input_weights)
    
    #second layer after activation 
    a2 = myfunction(z2)
    
    #third layer - output layer
    z3=np.dot(a2,output_weights)
    
    #output_layer after activation    
    output_layer=myfunction(z3)
    
    final_predictions=output_layer
    hyper_parameter=0.575     #hard coded paramter after performing cross validation of training data (threshold) 
    final_predictions[final_predictions <= hyper_parameter] = 0
    final_predictions[final_predictions >hyper_parameter] = 1
    #returns the final prediction labels
    return final_predictions





# Entry POINT MAIN CODE starts from here

#feature extration from unseen testing data
test_features,test_labels=feature_extraction()





#loading the input or layer 1 weights from the file l1_weights.npy which was found after training the model on the training data on sentences.txt 

input_layer_weights = np.load('l1_weights.npy')

#loading the output layer weights from the file output_weights.npy which was found after training the model on the training data on sentences.txt

output_weights = np.load('output_weights.npy')

#neural_net_predicted has the predicted labels for the unseen test data
neural_net_predicted=predict_neural(test_features,input_layer_weights,output_weights)
precision_neural , recall_neural , f1_neural = evaluate(neural_net_predicted, test_labels)
print "Neural_net results", precision_neural, recall_neural, f1_neural 

