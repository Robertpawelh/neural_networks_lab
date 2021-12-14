from os import error
import numpy as np
import matplotlib.pyplot as plt
from MLP import MLP
from optimizers import AdamOptimizer
from utils import load_mnist_data, print_data, scale_min_max_data
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from metrics import accuracy_score
from pathlib import Path
from optimizers import GradientDescentOptimizer, MomentumOptimizer, NesterovMomentumOptimizer, AdagradOptimizer, AdadeltaOptimizer, AdamOptimizer
import seaborn as sns
import time

sns.set()

REPETITIONS = 4
MAX_EPOCHS = 50
MAX_ACCEPTABLE_ERROR = 0.0001
MAX_ACCEPTABLE_VAL_ERROR_DIFF = 0.15
MAX_TRAINING_TIME = 3600
MAX_ACCURACY = 0.96
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
                        max_epochs = MAX_EPOCHS,
                        sigma = 0.1,
                        max_acceptable_error = MAX_ACCEPTABLE_ERROR,
                        max_acceptable_val_error_diff = MAX_ACCEPTABLE_VAL_ERROR_DIFF,
                        max_training_time = MAX_TRAINING_TIME,
                        accuracy_to_achieve = MAX_ACCURACY,
                        batch_size = 300,
                        weight_min = -1,
                        weight_max = 1,
                        optimizer = None,
                        weight_init_method = '',
                        debug = False,
                        verbose = True,
                        repetitions=REPETITIONS):
    
    if optimizer is None:
        optimizer = AdamOptimizer()
        
    model = MLP( max_epochs = max_epochs,
                 sigma = sigma,
                 max_acceptable_error = max_acceptable_error,
                 max_acceptable_val_error_diff = max_acceptable_val_error_diff,
                 max_training_time = max_training_time,
                 accuracy_to_achieve = accuracy_to_achieve,
                 batch_size = batch_size,
                 weight_min = weight_min,
                 weight_max = weight_max,
                 optimizer = optimizer,
                 weight_init_method = weight_init_method,
                 debug = debug,
                 verbose = verbose)
    
    def run_exp():
        model.init_layers(architecture)
        epochs, losses, val_losses, training_time, accuracy_goal_epoch = model.fit(X_train, Y_train, X_val, Y_val)
        Y_pred = np.argmax(model.predict(X_val), axis=1)
        Y_val_real = np.argmax(Y_val, axis=1)
        accuracy = accuracy_score(Y_pred, Y_val_real)
        
        return epochs, accuracy, losses, val_losses, training_time, accuracy_goal_epoch

    results = Parallel(n_jobs=int(repetitions))(delayed(run_exp)() for i in range(repetitions))

    epochs, accuracies, losses_lists, val_losses_list, exec_times, accuracy_goal_epochs = list(map(list, zip(*results)))
    return epochs, accuracies, losses_lists, val_losses_list, exec_times, accuracy_goal_epochs

def run_exp_group(params_values, run_exp_function, exp_label, filename, plot_title, x_label, x_log_scale=False, x_data=None):
    Path('results').mkdir(parents=True, exist_ok=True)
    epochs_list = []
    accuracy_list = []
    losses_list = []
    val_losses_list = []
    error_list = []
    time_list = []
    accuracy_goal_epochs_list = []
    
    if x_data is None:
        x_data = params_values
    
    for param in params_values:
        epochs, accuracy, losses, val_losses, exec_time, accuracy_goal_epochs = run_exp_function(param)
        result = f'With {param} {exp_label}, model has reached {round(100 * np.mean(accuracy), 2)} +- {round(100 * np.std(accuracy), 2)} accuracy_score after {np.mean(epochs)} epochs. Exec time: {round(np.mean(exec_time), 2)}s. Epochs to achieve min accuracy: {int(np.mean(accuracy_goal_epochs))}\n'

        with open(f"results/{filename}.txt", "a") as f:
            f.write(result)
            
        with open(f"results/{filename}_losses.txt", "a") as f:
            f.write(f'{param}: {str(losses)}')
            
        with open(f"results/{filename}_val_losses.txt", "a") as f:
            f.write(f'{param}: {str(val_losses)}')
            
        epochs_list.append(float(np.mean(epochs)))
        accuracy_list.append(float(np.mean(accuracy)))
        accuracy_goal_epochs_list.append(int(np.mean(accuracy_goal_epochs)))

        max_epochs = max([len(loss) for loss in losses])
        losses_list.append([ float(np.mean( [loss[i] for loss in losses if i < len(loss)])) for i in range(max_epochs) ])
        val_losses_list.append([ float(np.mean( [loss[i] for loss in val_losses if i < len(loss)])) for i in range(max_epochs) ])
            
        error_list.append(float(np.std(accuracy)))
        time_list.append(float(np.mean(exec_time)))
    
    plt.title((f'Zależność wartości funkcji straty zbioru od {plot_title} na przestrzeni epok'))
    plt.xlabel(f'Numer epoki')
    plt.ylabel('Wartość funkcji straty')

    for index, param in enumerate(params_values):
        plt.plot(accuracy_goal_epochs_list[index], losses_list[index][accuracy_goal_epochs_list[index] - 1], marker='o', markersize=8, markeredgecolor='black', markerfacecolor="None")  
        plt.plot(range(1, 1 + len(losses_list[index])), losses_list[index], label=x_data[index], marker='o', markersize=2)  
    plt.yscale('log')

    plt.legend()
    plt.savefig(f'results/{filename}_loss.png', bbox_inches='tight')
    plt.clf()
    plt.cla()
    
    
    plt.title((f'Zależność wartości funkcji straty zbioru walidacyjnego od {plot_title} na przestrzeni epok'))
    plt.xlabel(f'Numer epoki')
    plt.ylabel('Wartość funkcji straty zbioru walidacyjnego')

    for index, param in enumerate(params_values):
        plt.plot(range(1, 1 + len(val_losses_list[index])), val_losses_list[index], label=x_data[index], marker='o', markersize=2)  
    plt.yscale('log')

    plt.legend()
    plt.savefig(f'results/{filename}_val_loss.png', bbox_inches='tight')
    plt.clf()
    plt.cla()
     
    
    plt.title((f'Zależność trafności klasyfikacji od {plot_title}'))
    plt.xlabel(x_label)
    plt.ylabel('Trafność klasyfikacji')
    plt.errorbar(x_data, accuracy_list, error_list, fmt='o', markersize=4, capsize=4)
    plt.xticks(rotation=30)
    if x_log_scale:
        plt.xscale('log')
    
    plt.savefig(f'results/{filename}_acc.png', bbox_inches='tight')
    plt.clf()
    plt.cla()
    
    plt.title(f'Zależność liczby epok od {plot_title}')
    plt.xlabel(x_label)
    plt.ylabel('Liczba epok potrzebna do wyuczenia modelu')
    plt.plot(x_data, epochs_list, 'o')
    plt.xticks(rotation=30)
    if x_log_scale:
        plt.xscale('log')
    plt.ylim(0, MAX_EPOCHS + 1)
    plt.savefig(f'results/{filename}_epochs.png', bbox_inches='tight')
    plt.clf()
    plt.cla()
    
    plt.title(f'Zależność czasu uczenia od {plot_title}')
    plt.xlabel(x_label)
    plt.ylabel('Czas (w sekundach) potrzebny do wyuczenia modelu')
    plt.plot(x_data, time_list, 'o')
    plt.xticks(rotation=30)
    if x_log_scale:
        plt.xscale('log')
    plt.savefig(f'results/{filename}_time.png', bbox_inches='tight')
    plt.clf()
    plt.cla()

def run_research_1(X_train, X_val, Y_train, Y_val, act_function):
    LEARNING_RATE = 0.02
    optimizers = [
            GradientDescentOptimizer(learning_rate=LEARNING_RATE),
            MomentumOptimizer(learning_rate=LEARNING_RATE, momentum_rate=0.9),
            NesterovMomentumOptimizer(learning_rate=LEARNING_RATE, momentum_rate=0.9),
            AdagradOptimizer(learning_rate=LEARNING_RATE, epsilon=1e-8),
            AdadeltaOptimizer(learning_rate=LEARNING_RATE, epsilon=1e-8),
            AdamOptimizer(learning_rate=LEARNING_RATE)
    ]
    
    architecture = [
        {'layer_dim': 784 },
        {'layer_dim': 350, 'activation': act_function},
        {'layer_dim': 10, 'activation': 'softmax'}
    ]
    
    run_exp_function = lambda x: run_experiments(X_train, X_val, Y_train, Y_val, architecture, optimizer = x)
    run_exp_group(optimizers, 
                  run_exp_function, 
                  f'optimizer_{act_function}', 
                  f'optimizer_{act_function}', 
                  f'optymalizatora {act_function}',
                  'Optymalizator',
                  x_log_scale=False,
                  x_data=['Mini batch SGD', 'Momentum', 'Nesterov Momentum', 'Adagrad', 'Adadelta', 'Adam']
                  )

def run_research_2(X_train, X_val, Y_train, Y_val, act_function):
    weight_init_methods = ['he']
    
    architecture = [
        {'layer_dim': 784 },
        {'layer_dim': 350, 'activation': act_function},
        {'layer_dim': 10, 'activation': 'softmax'}
    ]
    
    if act_function == 'sigmoid':
        optimizer = AdagradOptimizer(learning_rate=0.02, epsilon=1e-8)
    elif act_function == 'relu':
        optimizer = AdagradOptimizer(learning_rate=0.02, epsilon=1e-8)
        
    run_exp_function = lambda x: run_experiments(X_train, X_val, Y_train, Y_val, architecture, optimizer = optimizer, weight_init_method = x)
    run_exp_group(weight_init_methods, 
                  run_exp_function, 
                  f'weight_init_method_{act_function}', 
                  f'weight_init_method_{act_function}', 
                  f'metody inicjalizacji wag {act_function}',
                  'Metoda inicjalizacji wag',
                  x_log_scale=False,
                  x_data=['Losowo, rozkład normalny', 'Xavier', 'He']
                  )

if __name__ == '__main__':
    X_train, X_val, Y_train, Y_val, X_test, Y_test = get_datasets(0.15)

    """BADANIE 1 """
    # run_research_1(X_train, X_val, Y_train, Y_val, act_function='relu')
    # run_research_1(X_train, X_val, Y_train, Y_val, act_function='sigmoid')
    
    """BADANIE 2 """
    #run_research_2(X_train, X_val, Y_train, Y_val, act_function='relu')
    run_research_2(X_train, X_val, Y_train, Y_val, act_function='sigmoid')
