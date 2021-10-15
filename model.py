from abc import ABC, abstractmethod
import numpy as np 

class Model(ABC):
    def __init__(self, n_inputs,
                 weight_min,
                 weight_max,
                 start_weight_min,
                 start_weight_max,
                 max_epochs,
                 debug,
                 verbose):
        self.n_inputs = n_inputs
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.start_weight_min = start_weight_min
        self.start_weight_max = start_weight_max
        self.max_epochs = max_epochs
        
        self.debug = debug
        self.verbose = verbose

        self.weights = None
    
    def reset_weights(self):
        n_weights = self.n_inputs + 1
        self.weights = np.random.uniform(self.start_weight_min, self.start_weight_max, n_weights)

    def fix_weights_range(self):
        self.weights = self.weights.clip(self.weight_min, self.weight_max)

    def calculate_z(self, neuron_values):
        if self.debug:
            print(f'Z dla: {self.weights} i {neuron_values}: ')
            print(f'{neuron_values.T @ self.weights}')
        return neuron_values.T @ self.weights

    def calculate_bipolar_output(self, z):
        return 1 if z > 0 else -1

    def calculate_unipolar_output(self, z):
        return 1 if z > 0 else 0
        
    @abstractmethod
    def fit(self, X_train, Y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass