from typing import Sequence
from tensorflow import keras
from keras.models import Model
from tensorflow.python.keras.layers.core import Flatten
from utils import load_mnist_data, scale_min_max_data
from tensorflow.keras.layers import Conv2D, Input, Dense, AveragePooling2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from metrics import accuracy_score
from tensorflow.keras.datasets import mnist
import numpy as np

EPOCHS = 6
BATCH_SIZE = 32

def mlp_architecture():
    model = keras.models.Sequential()
    model.add(Flatten())  
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    return model

def conv_architecture(pooling=None):
    model = keras.models.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1)))
    model.add(Dense(128, activation='relu'))
    if pooling is not None:
        model.add(pooling)
    model.add(Flatten())              
    model.add(Dense(10, activation='softmax'))
    
    return model

def research_1(X_train, Y_train, X_val, Y_val, verbose=False):
    model = mlp_architecture()
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=verbose)
    
    mlp_results = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=2)

def research_2(X_train, Y_train, X_val, Y_val, poolings, verbose=False):
    for pooling in poolings:
        model = conv_architecture(use_pooling=pooling)
        model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=verbose)
        cnn_pool_results = model.evaluate(X_val, Y_val, batch_size=BATCH_SIZE, verbose=2)

        print(f'CNN with pooling: {cnn_pool_results[1]}')

if __name__ == '__main__':
    X_train, Y_train = load_mnist_data('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', flatten=False)#('AND_bi_train_dset.csv')
    X_train = scale_min_max_data(X_train)

    X_test, Y_test = load_mnist_data('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', flatten=False)
    X_test = scale_min_max_data(X_test)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2],1)) 
    
    poolings = [AveragePooling2D(2, 2), MaxPooling2D(2, 2)]
    filters = [2**k for k in range(7)]
    kernel_sizes = [[k, k] for k in range(1, 5)]
    research_1(X_train, Y_train, X_test, Y_test, poolings)
