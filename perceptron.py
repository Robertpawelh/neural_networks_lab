import numpy as np

WEIGHT_MIN = -1
WEIGHT_MAX = 1
START_WEIGHT_MIN = -0.1
START_WEIGHT_MAX = 0.1
MAX_EPOCHS = 100

class Perceptron:
    def __init__(self, n_inputs, threshold=0.5, learning_rate=0.1, use_bias=False, type='unipolar', debug=False):
        self.n_inputs = n_inputs
        self.weights = np.zeros(n_inputs + 1 if use_bias else n_inputs)
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.type = type
        self.use_bias = use_bias
        self.debug = debug

    def reset_weights(self):
        n_weights = self.n_inputs + 1 if self.use_bias else self.n_inputs
        self.weights = np.random.uniform(START_WEIGHT_MIN, START_WEIGHT_MAX, n_weights)

    def calculate_z(self, neuron_values):
        if self.debug:
            print(f'Z dla: {self.weights} i {neuron_values}: ')
            print(f'{self.weights @ neuron_values}')
        return self.weights @ neuron_values

    def calculate_unipolar_output(self, z):
        return 1 if z > self.threshold else 0

    def calculate_bipolar_output(self, z):
        return 1 if z > self.threshold else -1

    def calculate_output_with_bias(self, z):
        return z > 0

    def calculate_output(self, z):
        if self.type == 'bipolar':
            return self.calculate_bipolar_output(z)
        elif self.type == 'unipolar':
            return self.calculate_unipolar_output(z)

    def calculate_loss(self, Y_pred, Y_train):
        return Y_train - Y_pred

    def calculate_weights_increase(self, loss, X_train):
        if self.debug:
            print(f'Weight updates for loss: {loss} and X_train: {X_train}')
        return loss * X_train
        
    def update_weights(self, loss, X_train):
        weights_increase = self.calculate_weights_increase(loss, X_train)
        self.weights = self.weights + self.learning_rate * weights_increase
    
    def train(self, X_train, Y_train):
        self.reset_weights()
        if self.use_bias:
            X_train = np.insert(X_train, 0, 1, axis=1)
        
        data_len = len(Y_train)

        # loss = np.ones(len(Y_train))
        epochs = 0
        global_loss_counter = 1
        while global_loss_counter != 0 and epochs < MAX_EPOCHS: #np.any(loss):
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
            print(f'Global loss after {epochs} epochs: {global_loss_counter}')

    def predict(self, X):
        if self.use_bias:
            X = np.insert(X, 0, 1, axis=0)
        z = self.calculate_z(X)
        Y_pred = self.calculate_output(z)
        return Y_pred




if __name__ == '__main__':
    training_datasets = [
        [ np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ]), np.array([0, 0, 0, 1]) ],
        [ np.array([ [0.07, 0.11], [1, 1], [0.05, 0.95], [0.96, 0.1], [0.97, 0.97], [0.91, 0.1] ]), np.array( [0, 1, 0, 0, 1, 0] ) ]
    ]
    X_train = training_datasets[0][0]
    Y_train = training_datasets[0][1]

    print(f'X_train: {X_train}')
    print(f'Y_train: {Y_train}')
    model = Perceptron(n_inputs=2, threshold=0.5, learning_rate=0.01, type='unipolar', use_bias=False)
    model.train(X_train, Y_train)

    X_test = np.array([ [1, 1], [0, 1], [1, 0], [0, 0 ] ] )
    for X in X_test:
        print(f'Predykcja dla {X}: {model.predict(X)}')