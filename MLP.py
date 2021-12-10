import numpy as np
import time

from numpy.lib.arraysetops import isin
from utils import load_mnist_data, scale_min_max_data
from metrics import accuracy_score
from sklearn.model_selection import train_test_split
from optimizers import GradientDescentOptimizer, MomentumOptimizer, NesterovMomentumOptimizer, AdagradOptimizer, AdadeltaOptimizer, AdamOptimizer

class MLP():
    def __init__(self,
                 #learning_rate = 0.1,
                 max_epochs = 10,
                 sigma = 0.1,
                 max_acceptable_error = 0.01,
                 max_acceptable_val_error_diff = 0.1,
                 max_training_time = 600,
                 accuracy_to_achieve = 1,
                 batch_size = None,
                 weight_min = -1,
                 weight_max = 1,
                 optimizer = None,
                 weight_init_method = '',
                 debug = False,
                 verbose = True):

        #self.learning_rate = learning_rate
        self.sigma = sigma
        self.max_epochs = max_epochs
        self.max_acceptable_error = max_acceptable_error
        self.max_acceptable_val_error_diff = max_acceptable_val_error_diff
        self.max_training_time = max_training_time
        self.accuracy_to_achieve = accuracy_to_achieve
        self.batch_size = batch_size

        self.weight_min = weight_min
        self.weight_max = weight_max

        self.verbose = verbose
        self.debug = debug
        
        self.optimizer = GradientDescentOptimizer(0.01) if optimizer is None else optimizer
        self.weight_init_method = weight_init_method

    def init_layer(self, n_inputs, n_outputs):
        if self.weight_init_method == '':
            return np.random.randn(n_inputs, n_outputs) * self.sigma
        
        elif self.weight_init_method == 'xavier':
            return np.random.randn(n_inputs, n_outputs) * np.sqrt(2 / (n_inputs + n_outputs))
            
        elif self.weight_init_method == 'he':
            return np.random.randn(n_inputs, n_outputs) * np.sqrt(2 / n_inputs)
 
        raise Exception('Invalid weight init method.')
    
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
                weights = self.init_layer(n_inputs, n_outputs)
                biases = self.init_layer(1, n_outputs)
            else:
                weights = parameters_values[index - 1]['w']
                biases = parameters_values[index - 1]['b']

            self.weights.append(weights)
            self.biases.append(biases)
            self.activations.append(activation)

            index += 1
        
        self.fix_parameters_to_keep_range()
    
    def fix_parameters_to_keep_range(self):
        for index, weights in enumerate(self.weights):
            self.weights[index] = weights.clip(self.weight_min, self.weight_max)
        
        for index, biases in enumerate(self.biases):
            self.biases[index] = biases.clip(self.weight_min, self.weight_max)

    def calculate_z(self, neuron_values, weights, bias):
        if self.debug:
            print('X: ', neuron_values.shape)
            print('W: ', weights.shape)
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
        return -(y - a)

    def activation_function(self, z, activation_func):
        z = np.clip(z, -709.78, 709.78)
        
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
        Y_train = Y_train.reshape((Y_train.shape[0], 1, Y_train.shape[1]))
        Y_pred = Y_pred.reshape((Y_pred.shape[0], Y_pred.shape[1], 1))

        return (Y_train @ -np.log(Y_pred))

    def feed_forward(self, X):
        neuron_values = [ {'a': X.T} ]

        for index in range(len(self.weights)):
            z = self.calculate_z(neuron_values[-1]['a'], self.weights[index], self.biases[index])
            a = self.activation_function(z, self.activations[index])
            neuron_values.append({'a': a, 'z': z})

        return neuron_values[-1]['a'].T, neuron_values

    def update_weights(self, Y_pred, Y_batch, neuron_values):
        gradient_values = self.backward_propagation(Y_pred, Y_batch, neuron_values)
        
        self.weights, self.biases = self.optimizer.get_updated_parameters(self.weights, self.biases, neuron_values, gradient_values)
        self.fix_parameters_to_keep_range()

    def single_layer_backpropagation(self, neuron_values, index, curr_activation, next_loss, next_weight):
        derivative = self.derivative(neuron_values, index, curr_activation)
        derivative = np.expand_dims(derivative, axis=0).T

        loss = ((next_weight @ next_loss) * (derivative))
        return loss

    def backward_propagation(self, Y_pred, Y, neuron_values):
        loss = self.softmax_gradient(Y, Y_pred)
        loss = np.expand_dims(loss, axis=2)
        gradient_values = [ loss ]

        for index in range(len(self.weights) - 2, -1, -1):
            loss = self.single_layer_backpropagation(neuron_values, index + 1, self.activations[index], loss, self.weights[index + 1])
            gradient_values.append(loss)

        return gradient_values

    def shuffle_dset(self, X_train, Y_train):
        p = np.random.permutation(len(X_train))
        X_train = X_train[p]
        Y_train = Y_train[p]
        
        return X_train, Y_train
        
    def fit(self, X_train, Y_train, X_val=None, Y_val=None):
        full_data_len = len(Y_train)
        val_data_len = len(Y_val) if Y_val is not None else 0
        batch_size = self.batch_size if self.batch_size is not None else full_data_len

        val_losses = []
        losses = []
        epochs = 0
        training_start_time = time.time()
        current_training_time = 0
        global_error = np.inf
        batch_start = 0
        previous_parameters = []
        accuracy_goal_epoch = None

        while self.max_acceptable_error < global_error and epochs < self.max_epochs:
            epochs += 1
            loss_sum = 0
            
            X_train, Y_train = self.shuffle_dset(X_train, Y_train)
            batch_start = 0
            batch_end = min(batch_start + batch_size, full_data_len)
            
            while batch_start < full_data_len:
                batch_end = min(batch_start + batch_size, full_data_len)
                X_batch = X_train[batch_start:batch_end]
                Y_batch = Y_train[batch_start:batch_end]
                Y_pred, neuron_values = self.feed_forward(X_batch)

                loss_sum += np.sum(self.loss_function(Y_pred, Y_batch))#, axis=0)
                self.update_weights(Y_pred, Y_batch, neuron_values)
    
                batch_start = batch_end + 1

            self.optimizer.on_epoch_end()
            
            current_training_time = time.time() - training_start_time
            global_error = (1/full_data_len) * loss_sum
            
            losses.append(global_error)
            
            Y_pred = np.argmax(self.predict(X_train), axis=1)
            Y_test = np.argmax(Y_train, axis=1)
            accuracy = accuracy_score(Y_pred, Y_test)
            
            if self.verbose:
                print(f'Global accuracy score after {epochs}: {accuracy}')
                # print(f'Global classification loss after {epochs} epochs: {classification_error(Y_pred, Y_test)}')
                print(f'Global loss after {epochs} epochs: {global_error}. ', end='')
            
            if X_val is not None and Y_val is not None:
                loss_sum = 0
 
                Y_pred, _ = self.feed_forward(X_val) 
                loss_sum = np.sum(self.loss_function(Y_pred, Y_val))
                val_error = (1/val_data_len) * loss_sum
                val_losses.append(val_error)
                
                if self.verbose:
                    print(f'Loss on validation: {val_error}', end='')
                
                if previous_parameters:
                    if (val_error > previous_parameters[-1]['val_error'] and val_error - global_error > self.max_acceptable_val_error_diff):
                        min_error_params = sorted(previous_parameters, key = lambda x: x['val_error'])[0]

                        self.weights = min_error_params['weights']
                        self.biases = min_error_params['biases']
                        
                        accuracy_goal_epoch = epochs if accuracy_goal_epoch is None else accuracy_goal_epoch
                        
                        if self.verbose:
                            print()
                        return epochs, losses, val_losses, current_training_time, accuracy_goal_epoch
                
                parameters = { 'val_error': val_error, 'weights': [weight.copy() for weight in self.weights], 'biases': [bias.copy() for bias in self.biases]}
                previous_parameters.append(parameters)
                
            if accuracy >= self.accuracy_to_achieve and accuracy_goal_epoch is None:
                if self.verbose:
                    print('Success!')
                accuracy_goal_epoch = epochs
            
            if current_training_time > self.max_training_time:
                accuracy_goal_epoch = epochs if accuracy_goal_epoch is None else accuracy_goal_epoch
                if self.verbose:
                    print()
                return epochs, losses, val_losses, current_training_time, accuracy_goal_epoch
            
            if self.verbose:
                print()
        
        accuracy_goal_epoch = epochs if accuracy_goal_epoch is None else accuracy_goal_epoch
        return epochs, losses, val_losses, current_training_time, accuracy_goal_epoch

    def predict(self, X):
        result, _ = self.feed_forward(X)
        return result

    def save_model(self, filepath):
        if not self.weights:
            raise Exception('Model was not initialized')

        with open(filepath, 'w') as f:
            for weight in self.weights:
                f.write(f'{weight.shape[0]}\t')

            f.write(f'{self.weights[-1].shape[1]}\t')
            f.write('\n')
            for activation in self.activations:
                f.write(f'{activation}\t')
            f.write('\n')
            for weight in self.weights:
                for row in weight:
                    for value in row:
                        f.write(f'{value} ')
                    f.write('\t')
                f.write('\n')
                
            for bias in self.biases:
                for row in bias:
                    for value in row:
                        f.write(f'{value} ')
                    f.write('\t')
                f.write('\n')

    def load_model(self, filepath):
        with open(filepath, 'r') as f:
            parameters_values = []
            weights = []
            biases = []
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
    X_train = scale_min_max_data(X_train)
    
    X_train = X_train[0:6010]
    Y_train = Y_train[0:6010]
    
    print(X_train.shape, Y_train.shape)
    n_outputs = len(np.unique(Y_train))
    Y_train = np.eye(n_outputs)[Y_train]

    #optimizer = GradientDescentOptimizer(learning_rate=0.01)
    #optimizer = MomentumOptimizer(learning_rate=0.01, momentum_rate=0.9) #TODO: verify
    #optimizer = NesterovMomentumOptimizer(learning_rate=0.01, momentum_rate=0.9)
    #optimizer = AdagradOptimizer(epsilon=1e-8)#(epsilon=1e-8)
    #optimizer = AdadeltaOptimizer(epsilon=1e-8)#(epsilon=1e-8)
    optimizer = AdamOptimizer()#(epsilon=1e-8)
    
    model = MLP(#learning_rate = 0.01,
                 max_epochs = 50,#100,
                 sigma = 0.1,
                 max_acceptable_error = 0.001,
                 max_acceptable_val_error_diff = 0.1,
                 max_training_time = 3600,
                 accuracy_to_achieve = 1,
                 batch_size = 300,
                 optimizer = optimizer,
                 weight_init_method = '',
                 debug = False,
                 verbose = True
    )

    architecture = [
        {'layer_dim': X_train.shape[1] },
        {'layer_dim': 350, 'activation': 'relu'},
        {'layer_dim': n_outputs, 'activation': 'softmax'}
    ]

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15)

    model.init_layers(architecture)
    epochs, losses, val_losses, training_time = model.fit(X_train, Y_train, X_val, Y_val)
    print(f'Trained after {epochs} epochs and {training_time} time')

    X_test, Y_test = load_mnist_data('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', flatten=True)
    X_test = scale_min_max_data(X_test)

    Y_pred = np.argmax(model.predict(X_test), axis=1)

    print('Accuracy score: ', accuracy_score(Y_pred, Y_test))
