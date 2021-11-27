#coding=utf-8
#!/usr/bin/env python3

#python train_CNN.py h5/axion1_40-250GeV_100k_mass0p5GeV.h5

import sys
import os
#import random
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

random.seed(10)

from h5py import File as HDF5File

import tensorflow as tf
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten , Convolution2D, MaxPooling2D , Lambda, Conv2D, Activation,Concatenate, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam , SGD , Adagrad, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import regularizers , initializers
import tensorflow.keras.backend as K
from joblib import dump, load

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

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
def _load_data(label, datafile):

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
    y = [label] * first.shape[0]

    return first, second, third, four, y, energy, sizes




def main():

    args = sys.argv[1:]
    #if len(args) < 1:
    #    return usage()

    print ('part1')   

    s = ["h5/axion1_40-250GeV_100k.h5",  "h5/pi0_40-250GeV_100k.h5", "h5/gamma_40-250GeV_100k.h5"];
    labels_ = {"h5/axion1_40-250GeV_100k.h5":0, "h5/pi0_40-250GeV_100k.h5":1, "h5/gamma_40-250GeV_100k.h5":2}
    test_split=0.3
    n_classes=3

    #<HDF5 dataset "energy": shape (100000, 1), type "<f8">
    #<HDF5 dataset "layer_0": shape (100000, 4, 16), type "<f8">
    #<HDF5 dataset "layer_1": shape (100000, 4, 128), type "<f8">
    #<HDF5 dataset "layer_2": shape (100000, 16, 16), type "<f8">
    #<HDF5 dataset "layer_3": shape (100000, 16, 8), type "<f8">
    first_all_, second_all_, third_all_, four_all_, y_all_, energy_all_, sizes_all_ = [
        np.concatenate(t) for t in [
            #This assumes different inputs have the same size
           a for a in zip(*[_load_data(labels_[file], file) for file in s])
        ]
    ]
  
    shuffler = np.random.permutation(len(first_all_))
    first_all = first_all_[shuffler]
    second_all = second_all_[shuffler]
    third_all = third_all_[shuffler]
    four_all = four_all_[shuffler]
    y_all = y_all_[shuffler]
    print("Before shuffel: y_all[0, 100000, 299999]'", y_all_[0], " ", y_all_[100000], "", y_all_[299999])
    print("After shuffel: y_all[0, 100000, 299999]'", y_all[0], " ", y_all[100000], "", y_all[299999])

    nall=first_all.shape[0] 
    #nall=100000 
    ntrain=int(nall*(1-test_split)) 

    #Print sizes
    #print("sizes_sig = ", sizes_sig)
    #print("sizes_bkg0 = ", sizes_bkg0)
    #print("sizes_bkg1 = ", sizes_bkg1)

    # Use GeV for energy
#    first_sig, second_sig, third_sig, four_sig, energy_sig = [
#        (X.astype(np.float32) / 1000)[:100000]
#        for X in [first_sig, second_sig, third_sig, four_sig, energy_sig]
#    ]
 
    #tuple object
    #inputs = [(first_all, second_all, third_all, four_all)] # Input Simulation
    X_train = [(first_all[:ntrain], second_all[:ntrain], third_all[:ntrain], four_all[:ntrain])] # Input Simulation
    X_test = [(first_all[ntrain:nall], second_all[ntrain:nall], third_all[ntrain:nall], four_all[ntrain:nall])] # Input Simulation
    
    y_train = y_all[:ntrain] 
    y_test = y_all[ntrain:nall] 

    print("train size", X_train[0][0].shape)
    print("test size", X_test[0][0].shape)
    
    y_train = to_categorical(y_train, n_classes) 
    y_test = to_categorical(y_test, n_classes) 
    print(y_train.shape)

    ## Taken from https://tutorials.one/how-to-use-the-keras-functional-api-for-deep-learning/
    # first input model
    visible1 = Input(shape=(4,16,1), name="visible1")
    conv11 = Conv2D(32, kernel_size=5, activation='relu', padding = "same", name="conv11")(visible1)
    pool11 = MaxPooling2D(pool_size=(2, 2), name="pool11")(conv11)
    conv12 = Conv2D(64, kernel_size=5, activation='relu', padding = "same", name="conv12")(pool11)
    pool12 = MaxPooling2D(pool_size=(2, 2), name="pool12")(conv12)
    logging.info("cov11.shape {}".format( conv11.shape))
    logging.info("pool1.shape {}".format( pool11.shape))
    logging.info("cov12.shape {}".format( conv12.shape))
    logging.info("pool2.shape {}".format( pool12.shape))
    flat1 = Flatten(name="flat1")(pool12)
    
    
    # second input model
    visible2 = Input(shape=(4,128,1),name="visible2")
    conv21 = Conv2D(32, kernel_size=5, activation='relu', padding = "same",name="conv21")(visible2)
    pool21 = MaxPooling2D(pool_size=(2, 2),name="pool21")(conv21)
    conv22 = Conv2D(64, kernel_size=5, activation='relu', padding = "same",name="conv22")(pool21)
    pool22 = MaxPooling2D(pool_size=(2, 2),name="pool22")(conv22)
    flat2 = Flatten(name="flat2")(pool22)
    logging.info("cov21.shape {}".format( conv21.shape))
    logging.info("poo21.shape {}".format( pool21.shape)) 
    logging.info("cov22.shape {}".format( conv22.shape)) 
    logging.info("poo22.shape {}".format( pool22.shape)) 
   

    # third input model
    visible3 = Input(shape=(16,16,1),name="visible3")
    conv31 = Conv2D(32, kernel_size=5, activation='relu', padding = "same",name="conv31")(visible3)
    pool31 = MaxPooling2D(pool_size=(2, 2),name="pool31")(conv31)
    conv32 = Conv2D(64, kernel_size=5, activation='relu', padding = "same",name="conv32")(pool31)
    pool32 = MaxPooling2D(pool_size=(2, 2),name="pool32")(conv32)
    flat3 = Flatten(name="flat3")(pool32)
    logging.info("cov31.shape {}".format( conv31.shape))
    logging.info("poo31.shape {}".format( pool31.shape))
    logging.info("cov32.shape {}".format( conv32.shape))
    logging.info("poo32.shape {}".format( pool32.shape))

    # forth input model
    visible4 = Input(shape=(16,8,1),name="visible4")
    conv41 = Conv2D(32, kernel_size=5, activation='relu', padding = "same",name="conv41")(visible4)
    pool41 = MaxPooling2D(pool_size=(2, 2),name="pool41")(conv41)
    conv42 = Conv2D(64, kernel_size=5, activation='relu', padding = "same",name="conv42")(pool41)
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
    output = Dense(n_classes, activation='softmax',name="output")(hidden2)
    cnn = Model(inputs= [visible1, visible2, visible3, visible4], outputs=output)

    #cnn.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    cnn.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=1e-4), metrics=['acc'])
    # summarize layers
    print(cnn.summary())
    # plot graph
    #plot_model(cnn, to_file='cnn_multiple_inputs.png')
  
    print("start fit\n")
    #epoch = 5
    #history = cnn.fit(X_train, y_train,  epochs=5, batch_size=100, validation_split =test_split)
    history = cnn.fit(X_train, y_train,  epochs=5, batch_size=100, validation_data =(X_test, y_test))
    y_pred = cnn.predict(X_test, batch_size=100)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print("y_pred", y_pred.shape) 
    print("y_test", y_test.shape) 
    #log_confusion_matrix(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, normalize = 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                          display_labels=["Signal", "bg0", "bg1"])
    disp.plot(cmap="Blues", values_format='.2f')
    plt.yticks(rotation=90)
    plt.tight_layout()
    plt.title('Confusion matrix ')
    plt.savefig('./ConfusionMatrix.png')


    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    fig=plt.figure(figsize=(8,6))
    fig.patch.set_color('white')
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('./cnn_acc.png')
    
    fig=plt.figure(figsize=(8,6))
    fig.patch.set_color('white')
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('./cnn_loss.png')

    return


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def log_confusion_matrix(y_test, y_pred):

    # Use the model to predict the values from the test_images.

    # Calculate the confusion matrix using sklearn.metrics
    cm = confusion_matrix(y_test, y_pred)

    figure = plot_confusion_matrix(cm, class_names = ["Signal", "Bg0", "Bg1"])
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)


if __name__ == '__main__':
	print('start')
	main()
