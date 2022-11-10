# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:28:31 2022

@author: Haitam

"""
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
import os

file = open("projectdata.txt", "r", encoding = "utf8")
lines = []

for i in file:
    lines.append(i)
    
data = ""

for i in lines:
    data=''.join(lines)
    
    data=data.replace('\n', '').replace('\r', '').replace('\ufeff', '')
    import string
    
    translator=str.maketrans(string.punctuation, ' '*len(string.punctuation))
     
    new_data = data.translate(translator)
    
    a=[]
    
    for i in data.split():
        if i not in a:
            a.append(i)
            
            data= ''.join(a)


tokenizer=Tokenizer()
tokenizer.fit_on_texts([data])
pickle.dump(tokenizer, open('tokenizer1.pkl', 'wb'))
data_sequence=tokenizer.texts_to_sequences([data])[0]

vocab_size = len(tokenizer.word_index) + 1

sequence=[]

for i in range(1,len(data_sequence)):
    words=data_sequence[i-1:i+1]
    sequence.append(words)
    
    sequence=np.array(sequence)
    
    #input
    X=[]
    
    #output(prediction)
    Y=[]
    
    
    for i in sequence:
        X.append(i[0])
        Y.append(i[1])
        
        
        
    X=np.array(X)
    Y=np.array(Y)
    
    #Converts a class vector (integers) to binary class matrix
    Y=tf.keras.utils.to_categorical(Y,num_classes=vocab_size)
    
    model=Sequential()
    model.add(Embedding(vocab_size,10,input_length=1)) #inputlenghth 1, the predicted word is based on 1 word 
    model.add(LSTM(1000, return_sequences=True)) #to pass it through another LSTM layer
    model.add(LSTM) #false by default
    model.add(Dense(1000, activation="relu"))
model.add(tf.keras.Dense(vocab_size, activation="softmax"))
from tensorflow import keras
from keras.utils.vis_utils import plot_model

keras.utils.plot_model(model, to_file='model.png', show_layer_names=True)

from tensorflow.keras.callbacks import ModelCheckpoint #storing the weights of our model after training
from tensorflow.keras.callbacks import ReduceLROnPlateau #reducing the learning rate
from tensorflow.keras.callbacks import TensorBoard #visualization of the graphs

checkpoint1 = ModelCheckpoint("nextword1.h5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto')
reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)
logdir='logsnextword1'
tensorboard_Visualization = TensorBoard(log_dir=logdir)


model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001))
model.fit(X, Y, epochs=150, batch_size=64, callbacks=[checkpoint1, reduce, tensorboard_Visualization])


    