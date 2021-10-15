import numpy as np
from utils import load_data, print_data
from model import Model

class Adaline(Model):
    def __init__(self,
                 n_inputs,
                 weight_min=-1,
                 weight_max=1,
                 start_weight_min=-0.1,
                 start_weight_max=0.1,
                 learning_rate=0.1,
                 max_acceptable_error=0.05,
                 max_epochs=1000,
                 debug=False,
                 verbose=True):
        Model.__init__(self, n_inputs, weight_min, weight_max, start_weight_min, start_weight_max, max_epochs, debug, verbose)
        self.max_acceptable_error = max_acceptable_error
        self.learning_rate = learning_rate

        self.weights = np.zeros(n_inputs + 1)

    def calculate_neuron_output(self, z):
        return self.calculate_bipolar_output(z)

    def calculate_loss(self, Y_train, z):
        return Y_train - z

    def calculate_loss_gradient(self, loss, X_train):
        return - 2 * loss * X_train

    def update_weights(self, X_train, Y_train):
        z = self.calculate_z(X_train)
        loss = self.calculate_loss(Y_train, z)
        self.weights = self.weights - self.learning_rate * self.calculate_loss_gradient(loss, X_train)
        self.fix_weights_to_keep_range()

        return loss**2

    def fit(self, X_train, Y_train):
        self.reset_weights()

        X_train = np.insert(X_train, 0, 1, axis=1)

        data_len = len(Y_train)
        epochs = 0
        global_error = np.inf

        while self.max_acceptable_error < global_error and epochs < self.max_epochs:
            epochs += 1
            loss_sum = 0
            for i in range(0, data_len):
                X = X_train[i]
                Y = Y_train[i]
                loss_sum += self.update_weights(X, Y)

            global_error = (1/data_len) * loss_sum
            if self.verbose:
                print(f'Global loss after {epochs} epochs: {global_error}')

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=0)
        z = self.calculate_z(X)
        Y_pred = self.calculate_neuron_output(z)
        return Y_pred


if __name__ == '__main__':
    X_train, Y_train = load_data('AND_bi_train_dset.csv')
    print_data(X_train, Y_train)
    model = Adaline(n_inputs=2,
                    learning_rate=0.01,
                    max_acceptable_error=0.15,
                    max_epochs=100,
                    start_weight_min=-0.1,
                    start_weight_max=0.1)
    model.fit(X_train, Y_train)

    X_test, Y_test = load_data('AND_bi_test_dset.csv')
    for index, X in enumerate(X_test):
        print(f'Prediction for {X}: {model.predict(X)}. Real label: {Y_test[index]}')