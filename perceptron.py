import numpy as np
import pandas as pd
from utils import load_data, print_data


class Perceptron:
    def __init__(self,
                 n_inputs,
                 weight_min=-1,
                 weight_max=1,
                 start_weight_min=-0.1,
                 start_weight_max=0.1,
                 threshold=0.5,
                 learning_rate=0.1,
                 max_epochs=100,
                 use_bias=False,
                 activation_function='unipolar',
                 debug=False,
                 verbose=True):
        self.n_inputs = n_inputs
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.start_weight_min = start_weight_min
        self.start_weight_max = start_weight_max
        self.max_epochs = max_epochs
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.use_bias = use_bias
        self.debug = debug
        self.verbose = verbose

        self.weights = np.zeros(n_inputs + 1 if use_bias else n_inputs)

    def reset_weights(self):
        n_weights = self.n_inputs + 1 if self.use_bias else self.n_inputs
        self.weights = np.random.uniform(self.start_weight_min, self.start_weight_max, n_weights)

    def fix_weights_range(self):
        self.weights = self.weights.clip(self.weight_min, self.weight_max)

    def calculate_z(self, neuron_values):
        if self.debug:
            print(f'Z dla: {self.weights} i {neuron_values}: ')
            print(f'{neuron_values.T @ self.weights}')
        return neuron_values.T @ self.weights

    def calculate_unipolar_output(self, z):
        threshold = 0 if self.use_bias else self.threshold
        return 1 if z > threshold else 0

    def calculate_bipolar_output(self, z):
        threshold = 0 if self.use_bias else self.threshold
        return 1 if z > threshold else -1

    def calculate_output(self, z):
        if self.activation_function == 'bipolar':
            return self.calculate_bipolar_output(z)
        elif self.activation_function == 'unipolar':
            return self.calculate_unipolar_output(z)
        else:
            raise Exception('Activation function currently is not implemented')

    def calculate_loss(self, Y_pred, Y_train):
        return Y_train - Y_pred

    def calculate_weights_increase(self, loss, X_train):
        if self.debug:
            print(f'Weight updates for loss: {loss} and X_train: {X_train}')
        return loss * X_train

    def update_weights(self, loss, X_train):
        weights_increase = self.calculate_weights_increase(loss, X_train)
        self.weights = self.weights + self.learning_rate * weights_increase
        self.fix_weights_range()

    def fit(self, X_train, Y_train):
        self.reset_weights()
        if self.use_bias:
            X_train = np.insert(X_train, 0, 1, axis=1)

        data_len = len(Y_train)

        # loss = np.ones(len(Y_train))
        epochs = 0
        global_loss_counter = 1
        while global_loss_counter != 0 and epochs < self.max_epochs:  # np.any(loss):
            epochs += 1
            global_loss_counter = 0
            for i in range(0, data_len):
                X = X_train[i]
                Y = Y_train[i]
                z = self.calculate_z(X)
                Y_pred = self.calculate_output(z)
                loss = self.calculate_loss(Y_pred, Y)
                self.update_weights(loss, X)
                if np.any(loss):
                    global_loss_counter += 1

            if self.verbose:
                print(f'Global loss after {epochs} epochs: {global_loss_counter}')

        return epochs

    def predict(self, X):
        if self.use_bias:
            X = np.insert(X, 0, 1, axis=0)
        z = self.calculate_z(X)
        Y_pred = self.calculate_output(z)
        return Y_pred


if __name__ == '__main__':
    X_train, Y_train = load_data('AND_train_dset.csv')
    print_data(X_train, Y_train)
    model = Perceptron(n_inputs=2, threshold=0.5, learning_rate=0.1, activation_function='unipolar', use_bias=False)

    model.fit(X_train, Y_train)

    X_test, Y_test = load_data('AND_test_dset.csv')
    for index, X in enumerate(X_test):
        print(f'Prediction for {X}: {model.predict(X)}. Real label: {Y_test[index]}')
