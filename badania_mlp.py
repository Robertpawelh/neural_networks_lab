from os import error
import numpy as np
import matplotlib.pyplot as plt
from MLP import MLP
from utils import load_mnist_data, print_data, scale_min_max_data
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from metrics import accuracy_score
from pathlib import Path
import seaborn as sns
import time

sns.set()

REPETITIONS = 10
MAX_EPOCHS = 100
MAX_ACCEPTABLE_ERROR = 0.001
MAX_ACCEPTABLE_VAL_ERROR_DIFF = 0.1
MAX_TRAINING_TIME = 3600
DSET_SIZE = 10000

def get_datasets(val_dset_size):
    X_train, Y_train = load_mnist_data('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', flatten=True)
    X_test, Y_test = load_mnist_data('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', flatten=True)
    
    indices =  np.random.choice(X_train.shape[0], DSET_SIZE, replace=False)
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    
    X_train = scale_min_max_data(X_train)
    X_test = scale_min_max_data(X_test)

    n_outputs = len(np.unique(Y_train))
    Y_train = np.eye(n_outputs)[Y_train]
    
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_dset_size)
    
    return X_train, X_val, Y_train, Y_val, X_test, Y_test

def run_experiments(X_train, X_val, Y_train, Y_val,
                        architecture, 
                        learning_rate = 0.01,
                        max_epochs = MAX_EPOCHS,
                        sigma = 0.1,
                        max_acceptable_error = MAX_ACCEPTABLE_ERROR,
                        max_acceptable_val_error_diff = MAX_ACCEPTABLE_VAL_ERROR_DIFF,
                        max_training_time = MAX_TRAINING_TIME,
                        batch_size = 100,
                        weight_min = -1,
                        weight_max = 1,
                        debug = False,
                        verbose = True,
                        repetitions=REPETITIONS):
    
    model = MLP(learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    sigma=sigma,
                    max_acceptable_error=max_acceptable_error,
                    max_acceptable_val_error_diff=max_acceptable_val_error_diff,
                    max_training_time=max_training_time,
                    batch_size=batch_size,
                    weight_min=weight_min,
                    weight_max=weight_max,
                    debug=debug,
                    verbose=verbose)
    
    def run_exp():
        model.init_layers(architecture)
        epochs, training_time = model.fit(X_train, Y_train, X_val, Y_val)
        Y_pred = np.argmax(model.predict(X_val), axis=1)
        Y_val_real = np.argmax(Y_val, axis=1)
        accuracy = accuracy_score(Y_pred, Y_val_real)
        
        return epochs, accuracy, training_time

    results = Parallel(n_jobs=int(repetitions/2))(delayed(run_exp)() for i in range(repetitions))

    epochs, accuracies, exec_times = list(map(list, zip(*results)))
    #print(epochs)
    #print(accuracies)
    return epochs, accuracies, exec_times

def run_exp_group(params_values, run_exp_function, exp_label, filename, plot_title, x_label, x_log_scale=False, x_data=None):
    Path('results').mkdir(parents=True, exist_ok=True)
    epochs_list = []
    accuracy_list = []
    error_list = []
    time_list = []
    
    if x_data is None:
        x_data = params_values
    
    for param in params_values[::-1]:    
        epochs, accuracy, exec_time = run_exp_function(param)
        result = f'With {param} {exp_label}, model has reached {round(100 * np.mean(accuracy), 2)} +- {round(100 * np.std(accuracy), 2)} accuracy_score after {np.mean(epochs)} epochs. Exec time: {round(np.mean(exec_time), 2)}s\n'

        with open(f"results/{filename}.txt", "a") as f:
            f.write(result)
            
        epochs_list.append(float(np.mean(epochs)))
        accuracy_list.append(float(np.mean(accuracy)))
        error_list.append(float(np.std(accuracy)))
        time_list.append(float(np.mean(exec_time)))
    
    plt.title((f'Zależność trafności klasyfikacji od {plot_title}'))
    plt.xlabel(x_label)
    plt.ylabel('Trafność klasyfikacji')
    #print(len(x_data), len(accuracy_list), len(error_list))
    plt.errorbar(x_data, accuracy_list[::-1], error_list[::-1], fmt='o', markersize=4, capsize=4)
    if x_log_scale:
        plt.xscale('log')
    
    #plt.yticks(np.arange(0, 100, 20))
    #plt.ylim(0.9, 1)
    plt.savefig(f'results/{filename}_acc.png', bbox_inches='tight')
    plt.clf()
    plt.cla()
    
    plt.title(f'Zależność liczby epok od {plot_title}')
    plt.xlabel(x_label)
    plt.ylabel('Liczba epok potrzebna do wyuczenia modelu')
    plt.plot(x_data, epochs_list[::-1], 'o-')
    if x_log_scale:
        plt.xscale('log')
    plt.ylim(0, MAX_EPOCHS + 1)
    plt.savefig(f'results/{filename}_epochs.png', bbox_inches='tight')
    plt.clf()
    plt.cla()
    
    plt.title(f'Zależność czasu uczenia od {plot_title}')
    plt.xlabel(x_label)
    plt.ylabel('Czas (w sekundach) potrzebny do wyuczenia modelu')
    plt.plot(x_data, time_list[::-1], 'o-')
    if x_log_scale:
        plt.xscale('log')
    plt.savefig(f'results/{filename}_time.png', bbox_inches='tight')
    plt.clf()
    plt.cla()

def run_research_1(X_train, X_val, Y_train, Y_val, act_function, hidden_layer_sizes = list(range(25, 525, 25))):
    architectures = [[
        {'layer_dim': 784 },
        {'layer_dim': hidden_layer_size, 'activation': act_function},
        {'layer_dim': 10, 'activation': 'softmax'}
    ] for hidden_layer_size in hidden_layer_sizes]
    
    run_exp_function = lambda x: run_experiments(X_train, X_val, Y_train, Y_val, architecture=x)
    run_exp_group(architectures, 
                  run_exp_function, 
                  'architecture', 
                  f'architecture_scores_{act_function}', 
                  'rozmiaru warstwy ukrytej',
                  'Rozmiar warstwy ukrytej',
                  x_data = hidden_layer_sizes
                  )
    
def run_research_2(X_train, X_val, Y_train, Y_val):
    learning_rates = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0001, 0.00001, 0.000001]
    
    architecture = [
        {'layer_dim': 784 },
        {'layer_dim': 400, 'activation': 'relu'},
        {'layer_dim': 10, 'activation': 'softmax'}
    ]
    
    run_exp_function = lambda x: run_experiments(X_train, X_val, Y_train, Y_val, architecture, learning_rate = x)
    run_exp_group(learning_rates, 
                  run_exp_function, 
                  'learning_rate', 
                  'learning_rate', 
                  'wartości współczynnika uczenia',
                  'Wartość współczynnika uczenia',
                  x_log_scale=True
                  )
    
def run_research_3(X_train, X_val, Y_train, Y_val):
    batch_sizes = [1]
    batch_sizes += list(np.arange(50, 550, 50))
    
    architecture = [
        {'layer_dim': 784 },
        {'layer_dim': 400, 'activation': 'relu'},
        {'layer_dim': 10, 'activation': 'softmax'}
    ]
    
    run_exp_function = lambda x: run_experiments(X_train, X_val, Y_train, Y_val, architecture, batch_size = x, learning_rate = 0.02)
    run_exp_group(batch_sizes, 
                  run_exp_function, 
                  'batch_size', 
                  'batch_size', 
                  'rozmiaru paczki danych',
                  'Rozmiar paczki danych',
                  )
    
def run_research_4(X_train, X_val, Y_train, Y_val):
    sigma_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]

    architecture = [
        {'layer_dim': 784 },
        {'layer_dim': 400, 'activation': 'relu'},
        {'layer_dim': 10, 'activation': 'softmax'}
    ]
    
    run_exp_function = lambda x: run_experiments(X_train, X_val, Y_train, Y_val, architecture, sigma = x, learning_rate = 0.02)
    run_exp_group(sigma_values, 
                  run_exp_function, 
                  'sigma_value', 
                  'sigma_value', 
                  'wartości parametru sigma',
                  'Wartość parametru sigma',
                  )


if __name__ == '__main__':
    X_train, X_val, Y_train, Y_val, X_test, Y_test = get_datasets(0.15)
    
    """BADANIE 1 """
    #run_research_1(X_train, X_val, Y_train, Y_val, 'relu')
    
    """Badanie 1_2"""
    #hidden_layer_sizes = list(range(100, 500, 100))
    #run_research_1(X_train, X_val, Y_train, Y_val, 'tanh', hidden_layer_sizes)
    
    """ BADANIE 2 """
    # run_research_2(X_train, X_val, Y_train, Y_val)
    
    """ BADANIE 3 """
    run_research_3(X_train, X_val, Y_train, Y_val)
    
    """ BADANIE 4 """
    # run_research_4(X_train, X_val, Y_train, Y_val)