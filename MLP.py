import numpy as np 
from utils import load_mnist_data, print_data
from sklearn.metrics import accuracy_score

class MLP():
    def __init__(self,
                 learning_rate=0.1,
                 max_epochs=1000,
                 sigma = 0.1,
                 debug=False,
                 verbose=True):

        self.learning_rate = learning_rate
        self.sigma = sigma
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.debug = debug

    def init_layers(self, architecture):
        self.weights = []
        self.biases = []
        self.activations = []
        index = 1
        for layer_arch in architecture[1:]:
            n_inputs = architecture[index - 1]['layer_dim']
            n_outputs = architecture[index]['layer_dim']
            activation = layer_arch['activation']

            weights = np.random.randn(n_inputs, n_outputs) * self.sigma
            biases = np.random.randn(1, n_outputs) * self.sigma

            self.weights.append(weights)
            self.biases.append(biases)
            self.activations.append(activation)

            index += 1
        

    def calculate_z(self, neuron_values, weights, bias):
        if self.debug:
            print(f'Z dla: {weights} i {neuron_values} i {bias}: ')
            print(f'{neuron_values @ weights.T + bias.T}')
        return neuron_values @ weights + bias

    def tanh(self, z):
        tanh = (2 / (1 + np.exp(-2 * z))) -1
        return tanh

    def sigmoid(self, z):
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid

    def relu(self, z):
        relu = np.maximum(0, z)
        return relu

    def softmax(self, z):
        softmax = np.exp(z) / np.sum(np.exp(z), axis=0)
        return softmax

    def activation_function(self, z, activation_func):
        if self.debug:
            print(activation_func, 'activated')

        if activation_func == 'sigmoid':
            return self.sigmoid(z)
        elif activation_func == 'relu':
            return self.relu(z)
        elif activation_func == 'tanh':
            return self.tanh(z)
        elif activation_func == 'softmax':
            return self.softmax(z)
        else:
            raise Exception('Activation function currently is not implemented')

    def fit(self, X_train, Y_train):
        pass

    def predict(self, X):
        previous_a = X
        for index in range(len(self.weights)):
            z = self.calculate_z(previous_a, self.weights[index], self.biases[index])
            a = self.activation_function(z, self.activations[index])

            previous_a = a

        return a

    def save_model(self, filepath):
        with open(filepath, 'w') as f:
            pass

    def load_model(self, filepath):
        pass

if __name__ == '__main__':
    X_train, Y_train = load_mnist_data('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', flatten=True)#('AND_bi_train_dset.csv')
    n_outputs = len(np.unique(Y_train))
    Y_train = np.eye(n_outputs)[Y_train]
    # print_data(X_train, Y_train)
    model = MLP(learning_rate=0.01, max_epochs=100, sigma=0.1)

    architecture = [
        {'layer_dim': X_train.shape[1] },
        {'layer_dim': 5, 'activation': 'sigmoid'},
        {'layer_dim': 20, 'activation': 'relu'},
        {'layer_dim': 30, 'activation': 'tanh'},
        {'layer_dim': n_outputs, 'activation': 'softmax'}
    ]

    model.init_layers(architecture)
    #model.fit(X_train, Y_train)

    X_test, Y_test = load_mnist_data('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', flatten=True)
    Y_pred = []

    for index, X in enumerate(X_test):
        prediction = model.predict(X)
        Y_pred.append(np.argmax(prediction))

    print('Accuracy score: ', accuracy_score(Y_pred, Y_test))
