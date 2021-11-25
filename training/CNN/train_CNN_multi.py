#coding=utf-8
#!/usr/bin/env python3

#python train_CNN.py h5/axion1_40-250GeV_100k_mass0p5GeV.h5

import sys
import os
import random
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import random
import time
import importlib
import logging
from tqdm import tqdm

import ROOT 

from h5py import File as HDF5File

import tensorflow as tf
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten , Convolution2D, MaxPooling2D , Lambda, Conv2D, Activation,Concatenate, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam , SGD , Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers , initializers
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
from joblib import dump, load


from keras.layers import Lambda, Input
from keras.layers import Dropout, Flatten, Dense 
import keras.backend as K
from keras.models import Sequential, Model 
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate


import enum
print(enum.__file__)  

def usage():
	print ('test usage')
	sys.stdout.write('''
			SYNOPSIS
			./train_CNN.py process 
			\n''')


# for particle, datafile in s.iteritems():
def _load_data(particle, datafile):

    import h5py
    print("load_data from datafile", datafile)
    d = h5py.File(datafile, 'r')
    first = np.expand_dims(d['layer_0'][:], -1)
    second = np.expand_dims(d['layer_1'][:], -1)
    third = np.expand_dims(d['layer_2'][:], -1)
    four = np.expand_dims(d['layer_3'][:], -1)
    energy = d['energy'][:].reshape(-1, 1) * 1000  # convert to MeV
    sizes = [first.shape[1], first.shape[2], second.shape[
        1], second.shape[2], third.shape[1], third.shape[2], four.shape[1], four.shape[2]]
    y = [particle] * first.shape[0]

    return first, second, third, four, y, energy, sizes




def main():

    args = sys.argv[1:]
    #if len(args) < 1:
    #    return usage()

    print ('part1')   

    #s = pd.DataFrame({1000 : "h5/axion1_40-250GeV_100k_mass0p5GeV.h5", 1000 : "h5/gamma_40-250GeV_100k_mass0p5GeV.h5"}, index=["sig","bkg"]);
    s = ["h5/axion1_40-250GeV_100k.h5", "h5/pi0_40-250GeV_100k.h5", "h5/gamma_40-250GeV_100k.h5"];
    events = [100000, 100000, 100000]

    #<HDF5 dataset "energy": shape (100000, 1), type "<f8">
    #<HDF5 dataset "layer_0": shape (100000, 4, 16), type "<f8">
    #<HDF5 dataset "layer_1": shape (100000, 4, 128), type "<f8">
    #<HDF5 dataset "layer_2": shape (100000, 16, 16), type "<f8">
    #<HDF5 dataset "layer_3": shape (100000, 16, 8), type "<f8">
    first, second, third, four, y, energy, sizes = [
        np.concatenate(t) for t in [
            a for a in zip(*[_load_data(events[0], file) for file in s])
        ]
    ]

    first_sig, second_sig, third_sig, four_sig, y_sig, energy_sig, sizes_sig = [
        np.concatenate(t) for t in [
            a for a in zip(*[_load_data(events[0], s[0])])
        ]
    ]
    first_bkg0, second_bkg0, third_bkg0, four_bkg0, y_bkg0, energy_bkg0, sizes_bkg0 = [
        np.concatenate(t) for t in [
            a for a in zip(*[_load_data(events[1], s[1])])
        ]
    ]
    first_bkg1, second_bkg1, third_bkg1, four_bkg1, y_bkg1, energy_bkg1, sizes_bkg1 = [
        np.concatenate(t) for t in [
            a for a in zip(*[_load_data(events[2], s[2])])
        ]
    ]
    
    print("first", first.shape)
    print("first_sig", first_sig.shape)
    print("second_sig", second_sig.shape)

    #Print sizes
    #print("sizes_sig = ", sizes_sig)
    #print("sizes_bkg0 = ", sizes_bkg0)
    #print("sizes_bkg1 = ", sizes_bkg1)

    # Use GeV for energy
    first_sig, second_sig, third_sig, four_sig, energy_sig = [
        (X.astype(np.float32) / 1000)[:100000]
        for X in [first_sig, second_sig, third_sig, four_sig, energy_sig]
    ]
    first_bkg0, second_bkg0, third_bkg0, four_bkg0, energy_bkg0 = [
        (X.astype(np.float32) / 1000)[:100000]
        for X in [first_bkg0, second_bkg0, third_bkg0, four_bkg0, energy_bkg0]
    ]
    first_bkg1, second_bkg1, third_bkg1, four_bkg1, energy_bkg1 = [
        (X.astype(np.float32) / 1000)[:100000]
       for X in [first_bkg1, second_bkg1, third_bkg1, four_bkg1, energy_bkg1]
    ]
#    y_sig = y_sig[:100000]
#    y_bkg0 = y_bkg0[:100000]
#    y_bkg1 = y_bkg1[:100000]
  

    inputs = ([(first, second, third, four)]) # Input Simulation
    
    labels = np.concatenate((np.ones(first_sig.shape[0]), np.zeros(first_bkg0.shape[0]), np.ones(first_bkg1.shape[0])+1))
    print(labels.shape)

    ## Taken from https://tutorials.one/how-to-use-the-keras-functional-api-for-deep-learning/
    # first input model
    visible1 = Input(shape=(4,16,1), name="visible1")
    conv11 = Conv2D(32, kernel_size=4, activation='relu', padding = "same", name="conv11")(visible1)
    pool11 = MaxPooling2D(pool_size=(2, 2), name="pool11")(conv11)
    conv12 = Conv2D(16, kernel_size=4, activation='relu', padding = "same", name="conv12")(pool11)
    pool12 = MaxPooling2D(pool_size=(2, 2), name="pool12")(conv12)
    logging.info("cov11.shape {}".format( conv11.shape))
    logging.info("pool1.shape {}".format( pool11.shape))
    logging.info("cov12.shape {}".format( conv12.shape))
    logging.info("pool2.shape {}".format( pool12.shape))
    flat1 = Flatten(name="flat1")(pool12)
    
    
    # second input model
    visible2 = Input(shape=(4,128,1),name="visible2")
    conv21 = Conv2D(32, kernel_size=4, activation='relu', padding = "same",name="conv21")(visible2)
    pool21 = MaxPooling2D(pool_size=(2, 2),name="pool21")(conv21)
    conv22 = Conv2D(16, kernel_size=4, activation='relu', padding = "same",name="conv22")(pool21)
    pool22 = MaxPooling2D(pool_size=(2, 2),name="pool22")(conv22)
    flat2 = Flatten(name="flat2")(pool22)
    logging.info("cov21.shape {}".format( conv21.shape))
    logging.info("poo21.shape {}".format( pool21.shape)) 
    logging.info("cov22.shape {}".format( conv22.shape)) 
    logging.info("poo22.shape {}".format( pool22.shape)) 
   

    # third input model
    visible3 = Input(shape=(16,16,1),name="visible3")
    conv31 = Conv2D(32, kernel_size=4, activation='relu', padding = "same",name="conv31")(visible3)
    pool31 = MaxPooling2D(pool_size=(2, 2),name="pool31")(conv31)
    conv32 = Conv2D(16, kernel_size=4, activation='relu', padding = "same",name="conv32")(pool31)
    pool32 = MaxPooling2D(pool_size=(2, 2),name="pool32")(conv32)
    flat3 = Flatten(name="flat3")(pool32)
    logging.info("cov31.shape {}".format( conv31.shape))
    logging.info("poo31.shape {}".format( pool31.shape))
    logging.info("cov32.shape {}".format( conv32.shape))
    logging.info("poo32.shape {}".format( pool32.shape))

    # forth input model
    visible4 = Input(shape=(16,8,1),name="visible4")
    conv41 = Conv2D(32, kernel_size=4, activation='relu', padding = "same",name="conv41")(visible4)
    pool41 = MaxPooling2D(pool_size=(2, 2),name="pool41")(conv41)
    conv42 = Conv2D(16, kernel_size=4, activation='relu', padding = "same",name="conv42")(pool41)
    pool42 = MaxPooling2D(pool_size=(2, 2),name="pool42")(conv42)
    flat4 = Flatten(name="flat4")(pool42)
    logging.info("cov41.shape {}".format( conv41.shape))
    logging.info("poo41.shape {}".format( pool41.shape))
    logging.info("cov42.shape {}".format( conv42.shape))
    logging.info("poo42.shape {}".format( pool42.shape))


    # merge input models
    merge = Concatenate(name="concatenate")([flat1, flat2, flat3, flat4])
    
    
    # interpretation model
    hidden1 = Dense(10, activation='relu',name="hidden1")(merge)
    hidden2 = Dense(10, activation='relu',name="hidden2")(hidden1)
    output = Dense(1, activation='softmax',name="output")(hidden2)
    cnn = Model(inputs=[visible1, visible2, visible3, visible4], outputs=output)

    cnn.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'], )
    # summarize layers
    print(cnn.summary())
    # plot graph
    #plot_model(model, to_file='multiple_inputs.png')
   
    #epoch = 5?
    history = cnn.fit(inputs, labels,  epochs=2, batch_size=100, validation_split =0.3)

    acc = history.history['acc']
    loss = history.history['loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.title('Training and validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.title('Training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


    return


if __name__ == '__main__':
	print('start')
	main()
