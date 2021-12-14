from typing import Sequence
from tensorflow import keras
from keras.models import Model
from tensorflow.python.keras.layers.core import Flatten
from utils import load_mnist_data, scale_min_max_data
from tensorflow.keras.layers import Conv2D, Input, Dense, AveragePooling2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from metrics import accuracy_score
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 25
BATCH_SIZE = 300
SAMPLES = 4

def mean(a):
    return sum(a) / len(a)


def mlp_architecture():
    model = keras.models.Sequential()
    model.add(Flatten())  
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    return model

def conv_architecture(filter=32, pooling=None, kernel=(3,3)):
    model = keras.models.Sequential()
    model.add(Conv2D(filter, kernel_size=kernel, activation='linear', input_shape=(28, 28, 1)))
    model.add(Dense(128, activation='relu'))
    if pooling is not None:
        model.add(pooling)
    model.add(Flatten())              
    model.add(Dense(10, activation='softmax'))
    
    return model

def research_1(X_train, Y_train, X_val, Y_val, verbose=False):
    accuracies_param = []
    losses_param = []
    for i in range(0, SAMPLES):
        model = mlp_architecture()
        model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
        history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=verbose)
        accuracy = model.evaluate(X_val, Y_val, verbose=2)
        accuracies_param.append(accuracy)
        losses_param.append(history.history['loss'])
    
    accuracy = np.mean(accuracies_param)
    losses = np.mean(np.array(losses_param), axis=0)
    plt.plot(losses)
    plt.xlabel(f'Numer epoki')
    plt.ylabel('Wartość funkcji straty')
    plt.title((f'Zależność wartości funkcji straty zbioru od epoki dla MLP'))
    with open('results/mlp_accuracies.txt', 'w') as f:
        accuracy = model.evaluate(X_val, Y_val, verbose=2)
        f.write(str(accuracy))
    plt.savefig(f'results/mlp_loss.png', bbox_inches='tight')
    plt.clf()
    plt.cla()

def research_2(X_train, Y_train, X_val, Y_val, poolings, x_labels, verbose=False):
    accuracies = []
    plt.xlabel(f'Numer epoki')
    plt.ylabel('Wartość funkcji straty')

    for index, pooling in enumerate(poolings):
        accuracies_param = []
        losses_param = []
        for i in range(0, SAMPLES):
            model = conv_architecture(pooling=pooling)
            model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
            history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=verbose)
            accuracy = model.evaluate(X_val, Y_val, verbose=2)
            accuracies_param.append(accuracy)
            losses_param.append(history.history['loss'])
        
        accuracy = np.mean(accuracies_param)
        accuracies.append(accuracy)
        losses = np.mean(np.array(losses_param), axis=0)
        
        accuracy = model.evaluate(X_val, Y_val, verbose=2)
        accuracies.append({pooling: accuracy })
        plt.plot(losses, label=x_labels[index])

        
    plt.title((f'Zależność wartości funkcji straty zbioru od metody poolingu na przestrzeni epok'))
    plt.legend()    
    with open('results/pooling_accuracies.txt', 'w') as f:
        f.write(str(accuracies))

    plt.savefig(f'results/pooling_loss.png', bbox_inches='tight')
    plt.clf()
    plt.cla()
    # plt.show()
        

def research_3(X_train, Y_train, X_val, Y_val, filters, verbose=False):
    accuracies = []
    plt.xlabel(f'Numer epoki')
    plt.ylabel('Wartość funkcji straty')

    for filter in filters:
        accuracies_param = []
        losses_param = []
        for i in range(0, SAMPLES):
            model = conv_architecture(filter=filter)
            model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
            history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=verbose)
            # cnn_pool_results = model.evaluate(X_val, Y_val, batch_size=BATCH_SIZE, verbose=2)

            accuracy = model.evaluate(X_val, Y_val, verbose=2)
            accuracies_param.append(accuracy)
            losses_param.append(history.history['loss'])
        
        accuracy = np.mean(accuracies_param)
        accuracies.append(accuracy)
        losses = np.mean(np.array(losses_param), axis=0)
        #print(losses_param)
        #losses = [np.mean([loss[index] for loss in losses_param] for index in range(len(losses_param[0])))]
        print(f'CNN with {filter} filter: {accuracy}')
        plt.plot(losses, label=filter)
        
    plt.title((f'Zależność wartości funkcji straty zbioru od rozmiaru filtra na przestrzeni epok'))
    
    print(accuracies)
    with open('results/filter_accuracies.txt', 'w') as f:
        f.write(str(accuracies))
    plt.legend()
    plt.savefig(f'results/filter_loss.png', bbox_inches='tight')
    # plt.show()
    plt.clf()
    plt.cla()

def research_4(X_train, Y_train, X_val, Y_val, kernels, verbose=False):
    accuracies = []
    plt.xlabel(f'Numer epoki')
    plt.ylabel('Wartość funkcji straty')

    for kernel in kernels:
        accuracies_param = []
        losses_param = []
        for i in range(0, SAMPLES):
            model = conv_architecture(kernel=kernel)
            model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
            history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=verbose)
            accuracy = model.evaluate(X_val, Y_val, verbose=2)
            accuracies_param.append(accuracy)
            losses_param.append(history.history['loss'])
                
        accuracy = np.mean(accuracies_param)
        accuracies.append(accuracy)
        print(losses_param)
        losses = np.mean(np.array(losses_param), axis=0)
            
        print(f'CNN with {kernel} kernel: {accuracy}')
        plt.plot(losses, label=kernel)
        
        plt.title((f'Zależność wartości funkcji straty zbioru od rozmiaru filtra na przestrzeni epok'))
    
    print(accuracies)
    with open('results/kernel_accuracies.txt', 'w') as f:
        f.write(str(accuracies))
    plt.legend()
    plt.savefig(f'results/kernel_loss.png', bbox_inches='tight')
    # plt.show()
    plt.clf()
    plt.cla()
    
        

if __name__ == '__main__':
    X_train, Y_train = load_mnist_data('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', flatten=False)#('AND_bi_train_dset.csv')
    X_train = scale_min_max_data(X_train)
    
    X_train = X_train[0:8500]
    Y_train = Y_train[0:8500]


    X_test, Y_test = load_mnist_data('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', flatten=False)
    X_test = scale_min_max_data(X_test)
    
    X_test = X_test[0:1500]
    Y_test = Y_test[0:1500]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2],1)) 
    
    poolings = [None, AveragePooling2D(2, 2), MaxPooling2D(2, 2)]
    filters = [2**k for k in range(7)]
    kernel_sizes = [[k, k] for k in range(1, 5)]
    research_1(X_train, Y_train, X_test, Y_test)
    research_2(X_train, Y_train, X_test, Y_test, poolings, x_labels=['Brak', 'Average pooling', 'Max pooling'])
    research_3(X_train, Y_train, X_test, Y_test, filters)
    research_4(X_train, Y_train, X_test, Y_test, kernel_sizes)
    research_2(X_train, Y_train, X_test, Y_test, poolings)
