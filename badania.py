import numpy as np
from perceptron import Perceptron
from utils import load_data

if __name__ == '__main__':
    X_train_bipolar, Y_train_bipolar = load_data('AND_bi_train_dset.csv')
    X_train_unipolar, Y_train_unipolar = load_data('AND_train_dset.csv')

    activation_function = 'unipolar'
    if activation_function == 'unipolar':
        X_train, Y_train = X_train_unipolar, Y_train_unipolar
    elif activation_function == 'bipolar':
        X_train, Y_train = X_train_bipolar, Y_train_bipolar

    for threshold in np.arange(-1, 1, 0.1):
        model = Perceptron(n_inputs=2,
                           start_weight_min=-0.05,
                           start_weight_max=0.05,
                           threshold=threshold,
                           learning_rate=0.4,
                           activation_function=activation_function,
                           use_bias=False,
                           verbose=False)

        epochs = model.fit(X_train, Y_train)
        print(f'Trained after {epochs} epochs')
