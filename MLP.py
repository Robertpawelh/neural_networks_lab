import numpy as np 
from utils import load_mnist_data, print_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class MLP():
    def __init__(self,
                 learning_rate = 0.1,
                 max_epochs = 10,
                 sigma = 0.1,
                 max_acceptable_error = 0.01,
                 batch_size = 1,
                 weight_min = -1,
                 weight_max = 1,
                 debug = False,
                 verbose = True):

        self.learning_rate = learning_rate
        self.sigma = sigma
        self.max_epochs = max_epochs
        self.max_acceptable_error = max_acceptable_error
        self.batch_size = batch_size

        self.weight_min = weight_min
        self.weight_max = weight_max

        self.verbose = verbose
        self.debug = debug

    def init_layers(self, architecture, parameters_values=None):
        self.weights = []
        self.biases = []
        self.activations = []
        index = 1
        for layer_arch in architecture[1:]:
            n_inputs = architecture[index - 1]['layer_dim']
            n_outputs = architecture[index]['layer_dim']
            activation = layer_arch['activation']

            if parameters_values is None:
                weights = np.random.randn(n_inputs, n_outputs) * self.sigma
                biases = np.random.randn(1, n_outputs) * self.sigma
            else:
                weights = parameters_values[index - 1]['w']
                biases = parameters_values[index - 1]['b']

            self.weights.append(weights)
            self.biases.append(biases)
            self.activations.append(activation)

            index += 1
    
    def fix_parameters_to_keep_range(self):
        for weights in self.weights:
            weights = weights.clip(self.weight_min, self.weight_max)
        
        for biases in self.biases:
            biases = biases.clip(self.weight_min, self.weight_max)

    def calculate_z(self, neuron_values, weights, bias):
        if self.debug:
            print('W: ', weights.shape)
            print('X: ', neuron_values.shape)
            print('B: ', bias.shape)

        return (neuron_values.T @ weights + bias).T 

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
        softmax = np.e**z / np.sum(np.e**z, axis = 0)

        return softmax

    def softmax_gradient(self, y, a):
        y = y.reshape((len(y), 1))
        return -(y - a)

    def activation_function(self, z, activation_func):
        z = np.clip(z, -188.72, 188.72) #np.clip(z, -709.78, 709.78)
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

    def sigmoid_derivative(self, neuron_values, index):
        sig = neuron_values[index]['a']

        return sig * (1 - sig)

    def softplus_derivative(self, neuron_values, index):
        sig = neuron_values[index]['a']
        return sig

    def relu_derivative(self, neuron_values, index):
        z = neuron_values[index]['z']
        return np.greater(z, 0).astype(int)
        
        #return self.softplus_derivative(neuron_values, index)

    def tanh_derivative(self, neuron_values, index):
        tanh = neuron_values[index]['a']
        return (1 - tanh ** 2)


    def derivative(self, neuron_values, index, activation_func):
        if self.debug:
            print(activation_func, 'derivative activated')

        if activation_func == 'sigmoid':
            return self.sigmoid_derivative(neuron_values, index)
        elif activation_func == 'relu':
            return self.relu_derivative(neuron_values, index)
        elif activation_func == 'tanh':
            return self.tanh_derivative(neuron_values, index)
        else:
            raise Exception('Activation function currently is not implemented')

    def loss_function(self, Y_pred, Y_train):
        return Y_train @ -np.log(Y_pred)

    def feed_forward(self, X):
        X = X.reshape((len(X), 1))
        neuron_values = [ {'a': X} ]

        for index in range(len(self.weights)):
            z = self.calculate_z(neuron_values[-1]['a'], self.weights[index], self.biases[index])
            a = self.activation_function(z, self.activations[index])
            neuron_values.append({'a': a, 'z': z})

        return neuron_values[-1]['a'], neuron_values

    def gradient_descent(self, n_samples, neuron_values, gradient_values):
        gradient_values = gradient_values[::-1]
        for index in range(len(self.weights) - 1, -1, -1):
            weight_grad = gradient_values[index]
            #print(self.weights[index].shape)
            #print(neuron_values[index]['a'].shape)
            #print(weight_grad.shape)
            self.weights[index] = self.weights[index] - (self.learning_rate/n_samples) * (neuron_values[index]['a'] * weight_grad.T)
            self.biases[index] = self.biases[index] - (self.learning_rate/n_samples)  * weight_grad.T

        self.fix_parameters_to_keep_range()

    def update_weights(self, n_samples, neuron_values, gradient_values):
        self.gradient_descent(n_samples, neuron_values, gradient_values)

    def single_layer_backpropagation(self, neuron_values, index, curr_activation, next_loss, next_weight):
        a = neuron_values[index - 1]['a'].T
        loss = ((next_weight @ next_loss) * (self.derivative(neuron_values, index, curr_activation)))

        return loss

    def backward_propagation(self, Y_pred, Y, neuron_values):
        loss = self.softmax_gradient(Y, Y_pred)
        gradient_values = [ loss ]

        for index in range(len(self.weights) - 2, -1, -1):
            loss = self.single_layer_backpropagation(neuron_values, index + 1, self.activations[index], loss, self.weights[index + 1])
            gradient_values.append(loss)

        return gradient_values

    def fit(self, X_train, Y_train, X_val=None, Y_val=None):
        full_data_len = len(Y_train)
        val_data_len = len(Y_val) if Y_val is not None else 0

        epochs = 0
        global_error = np.inf

        batch_start = 0

        previous_parameters = []

        while self.max_acceptable_error < global_error and epochs < self.max_epochs:
            epochs += 1
            loss_sum = 0
            
            batch_start = 0

            while batch_start < full_data_len:
                batch_end = min(batch_start + self.batch_size, full_data_len)
                for i in range(batch_start, batch_end):
                    X = X_train[i]
                    Y = Y_train[i]
                    Y_pred, neuron_values = self.feed_forward(X)
                    gradient_values = self.backward_propagation(Y_pred, Y, neuron_values)
                    loss_sum += self.loss_function(Y_pred, Y)

                    # czy to nie powinno byc po batchu
                    self.update_weights(batch_end - batch_start, neuron_values, gradient_values)
        
                batch_start = batch_end

            global_error = (1/full_data_len) * loss_sum
            if self.verbose:
                print(f'Global loss after {epochs} epochs: {global_error}')
            
            if X_val is not None and Y_val is not None:
                loss_sum = 0
                for i in range(val_data_len):
                    X = X_val[i]
                    Y = Y_val[i]
                    Y_pred, _ = self.feed_forward(X) 
                    loss_sum += self.loss_function(Y_pred, Y)

                val_error = (1/val_data_len) * loss_sum
                parameters = { 'val_error': val_error, 'weights': [weight.copy() for weight in self.weights], 'biases': [bias.copy() for bias in self.biases]}
                previous_parameters.append(parameters)

                if self.verbose:
                    print(f'Loss on validation dset after {epochs} epochs: {val_error}')
            

        
        return epochs

    def predict(self, X):
        result, _ = self.feed_forward(X)
        return result

    def save_model(self, filepath):
        if not self.weights:
            raise Exception('Model was not initialized')

        with open(filepath, 'w') as f:
            #f.write('n_layers\n')
            for weight in self.weights:
                f.write(f'{weight.shape[0]}\t')

            f.write(f'{self.weights[-1].shape[1]}\t')
            #f.write('activations\n')
            f.write('\n')
            for activation in self.activations:
                f.write(f'{activation}\t')
            #f.write('weights\n')
            f.write('\n')
            for weight in self.weights:
                for row in weight:
                    for value in row:
                        f.write(f'{value} ')
                    f.write('\t')
                f.write('\n')
                #f.write('\n')

            for bias in self.biases:
                for row in bias:
                    for value in row:
                        f.write(f'{value} ')
                    f.write('\t')
                f.write('\n')
                #f.write('\n')



    def load_model(self, filepath):
        with open(filepath, 'r') as f:
            parameters_values = []
            weights = []
            biases = []
            #lines = f.readlines()
            layers_dims = [int(val) for val in f.readline().strip().split('\t')]
            activations = f.readline().strip().split('\t')

            architecture = [ {'layer_dim': layers_dims[0] }]
            for index, layer_dim in enumerate(layers_dims[1:]):
                architecture.append( {'layer_dim': layer_dim, 'activation': activations[index]})

            for i in range(len(activations)):
                w = f.readline().strip().split('\t')
                for index, row in enumerate(w):
                    w[index] = list(map(float, row.strip().split(' ')))

                weights.append(np.asarray(w))

            for i in range(len(activations)):
                b = f.readline().strip().split('\t')
                for index, row in enumerate(b):
                    b[index] = list(map(float, row.strip().split(' ')))

                biases.append(np.asarray(b))
            
            for i in range(len(activations)):
                print(weights[i].shape)
                print(biases[i].shape)
                values = {'w': weights[i], 'b': biases[i]}
                parameters_values.append(values)

            self.init_layers(architecture, parameters_values)

        
    

if __name__ == '__main__':
    X_train, Y_train = load_mnist_data('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', flatten=True)#('AND_bi_train_dset.csv')
    n_outputs = len(np.unique(Y_train))
    Y_train = np.eye(n_outputs)[Y_train]
    # print_data(X_train, Y_train)
    model = MLP(learning_rate = 0.05,
                 max_epochs = 20,
                 sigma = 0.1,
                 max_acceptable_error = 0.15,
                 batch_size = 100,
                 debug = False,
                 verbose = True
    )

    architecture = [
        {'layer_dim': X_train.shape[1] },
        {'layer_dim': 30, 'activation': 'tanh'},
        {'layer_dim': 100, 'activation': 'sigmoid'},
        {'layer_dim': n_outputs, 'activation': 'softmax'}
    ]
    X_train = X_train[0:5000]
    Y_train = Y_train[0:5000]

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15)

    model.init_layers(architecture)
    model.fit(X_train, Y_train)#, X_val, Y_val)


    X_test, Y_test = load_mnist_data('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', flatten=True)
    Y_pred = []

    for index, X in enumerate(X_test):
        prediction = model.predict(X)
        #print(prediction)
        Y_pred.append(np.argmax(prediction))

    print('Accuracy score: ', accuracy_score(Y_pred, Y_test))
