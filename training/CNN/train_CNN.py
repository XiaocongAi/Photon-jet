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

import ROOT 

from h5py import File as HDF5File

import tensorflow as tf
from tensorflow import keras

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
    if len(args) < 1:
        return usage()

    print ('part1')   

    #s = pd.DataFrame({1000 : "h5/axion1_40-250GeV_100k_mass0p5GeV.h5", 1000 : "h5/gamma_40-250GeV_100k_mass0p5GeV.h5"}, index=["sig","bkg"]);
    s = ["h5/axion1_40-250GeV_100k_mass0p5GeV.h5", "h5/gamma_40-250GeV_100k_mass0p5GeV.h5", "h5/pi0_40-250GeV_100k_mass0p5GeV.h5"];
    events = [1000, 1000, 1000]

#<HDF5 dataset "energy": shape (100000, 1), type "<f8">
#<HDF5 dataset "layer_0": shape (100000, 4, 16), type "<f8">
#<HDF5 dataset "layer_1": shape (100000, 4, 128), type "<f8">
#<HDF5 dataset "layer_2": shape (100000, 16, 16), type "<f8">
#<HDF5 dataset "layer_3": shape (100000, 16, 8), type "<f8">
    first, second, third, four, y, energy, sizes = [
        np.concatenate(t) for t in [
            a for a in zip(*[_load_data(1000, file) for file in s])
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
  
#    calorimeter = [Input(shape=sizes[:2] + [1]),
#                   Input(shape=sizes[2:4] + [1]),
#                   Input(shape=sizes[4:6] + [1]),
#                   Input(shape=sizes[8:] + [1])
#                   ]
#    input_energy = Input(shape=(1, )) 
#    print("calorimeter.shape", calorimeter[0].shape)

    #first = first.reshape(first.shape[0], 16, 16, 1)
    #print("first shape", first.shape)

    inputs = ([(first, second)]) # Input Simulation
    labels = np.concatenate((np.ones(first_sig.shape[0]), np.zeros(first_bkg0.shape[0]), np.ones(first_bkg1.shape[0])+1))
    print(labels.shape)

    ## Taken from https://tutorials.one/how-to-use-the-keras-functional-api-for-deep-learning/
    # first input model
    visible1 = Input(shape=(4,16,1))
    conv11 = Conv2D(32, kernel_size=4, activation='relu', padding = "same")(visible1)
    pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
    conv12 = Conv2D(16, kernel_size=4, activation='relu', padding = "same")(pool11)
    pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
    print("cov11.shape", conv11.shape) 
    print("pool1.shape", pool11.shape) 
    print("cov12.shape", conv12.shape) 
    print("pool2.shape", pool12.shape) 
    flat1 = Flatten()(pool12)
    # second input model
    visible2 = Input(shape=(4,128,1))
    conv21 = Conv2D(32, kernel_size=4, activation='relu', padding = "same")(visible2)
    pool21 = MaxPooling2D(pool_size=(2, 2))(conv21)
    conv22 = Conv2D(16, kernel_size=4, activation='relu', padding = "same")(pool21)
    pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)
    flat2 = Flatten()(pool22)
    print("cov21.shape", conv21.shape) 
    print("poo21.shape", pool21.shape) 
    print("cov22.shape", conv22.shape) 
    print("poo22.shape", pool22.shape) 

    # merge input models
    merge = concatenate([flat1, flat2])
    # interpretation model
    hidden1 = Dense(10, activation='relu')(merge)
    hidden2 = Dense(10, activation='relu')(hidden1)
    output = Dense(1, activation='sigmoid')(hidden2)
    cnn = Model(inputs=[visible1, visible2], outputs=output)

    cnn.compile(loss="binary_crossentropy", optimizer='adam', metrics=['acc'])
    # summarize layers
    print(cnn.summary())
    # plot graph
    #plot_model(model, to_file='multiple_inputs.png')
   
    #epoch = 5?
    history = cnn.fit(inputs, labels,  epochs=2, batch_size=100)

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





def single_layer_energy(x):
    #shape = K.get_variable_shape(x)
    return K.reshape(K.sum(x, axis=range(1, len(x))), (-1, 1))


def single_layer_energy_output_shape(input_shape):
    shape = list(input_shape)
    # assert len(shape) == 3
    return (shape[0], 1)


def calculate_energy(x):
    return Lambda(single_layer_energy, single_layer_energy_output_shape)(x)


if __name__ == '__main__':
	print('start')
	main()
