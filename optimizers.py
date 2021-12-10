import numpy as np
import abc

class Optimizer(abc.ABC):
    def __init__(self):
        super().__init__()
    
    @abc.abstractmethod
    def get_updated_parameters(self, weights, biases, neuron_values, gradient_values):
        pass
        
    def on_epoch_end(self):
        pass
    
    def calculate_update_from_grad(self, gradient_values, neuron_values):
        grad = gradient_values
        a = neuron_values['a']
        n_samples = a.shape[1]
        
        a = np.expand_dims(a.T, axis=1)

        weight_update = np.sum((grad @ a), axis=0).T
        bias_update = np.sum(grad, axis=0).T

        weight_update = (1/n_samples) * weight_update
        bias_update = (1/n_samples) * bias_update
        
        return weight_update, bias_update

class GradientDescentOptimizer(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def get_updated_parameters(self, weights, biases, neuron_values, gradient_values):
        gradient_values = gradient_values[::-1]
        
        updated_weights = [None] * len(weights)
        updated_biases = [None] * len(biases)
        for index in range(len(weights) - 1, -1, -1):
            weight_update, bias_update = self.calculate_update_from_grad(gradient_values[index], neuron_values[index])
            
            updated_weights[index] = weights[index] - (self.learning_rate * weight_update)
            updated_biases[index] = biases[index] - (self.learning_rate * bias_update)
        
        return updated_weights, updated_biases

class MomentumOptimizer(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.last_epoch_weights_update = None
        self.last_epoch_biases_update = None
        self.last_weights_update = None
        self.last_biases_update = None
        
        self.n_samples = 0
    
    def on_epoch_end(self):
        self.last_epoch_weights_update = [x/self.n_samples for x in self.last_weights_update]
        self.last_epoch_biases_update = [x/self.n_samples for x in self.last_biases_update]
        self.last_weights_update = None
        self.last_biases_update = None
        
        self.n_samples = 0
    
    def get_updated_parameters(self, weights, biases, neuron_values, gradient_values):
        gradient_values = gradient_values[::-1]
        
        updated_weights = [None] * len(weights)
        updated_biases = [None] * len(biases)
        
        if self.last_weights_update is None and self.last_biases_update is None:
            self.last_weights_update = [None] * len(weights)
            self.last_biases_update = [None] * len(biases)
            for index in range(len(weights) - 1, -1, -1):
                self.last_weights_update[index] = np.zeros(shape=weights[index].shape)
                self.last_biases_update[index] = np.zeros(shape=biases[index].shape)
        
        for index in range(len(weights) - 1, -1, -1):
            weight_update, bias_update = self.calculate_update_from_grad(gradient_values[index], neuron_values[index])

            if self.last_epoch_weights_update is not None and self.last_epoch_biases_update is not None:
                weight_update = (self.momentum_rate * self.last_epoch_weights_update[index]) - (self.learning_rate * weight_update)
                bias_update = (self.momentum_rate * self.last_epoch_biases_update[index]) - (self.learning_rate * bias_update)
            else:
                weight_update = -self.learning_rate * weight_update
                bias_update = -self.learning_rate * bias_update
                            
            updated_weights[index] = weights[index] + weight_update
            updated_biases[index] = biases[index] + bias_update
                
            self.last_weights_update[index] += weight_update
            self.last_biases_update[index] += bias_update
            
        self.n_samples += 1
        
        return updated_weights, updated_biases
    
class NesterovMomentumOptimizer(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.last_epoch_weights_update = None
        self.last_epoch_biases_update = None
        self.last_weights_update = None
        self.last_biases_update = None
        
        self.n_samples = 0
        
    def on_epoch_end(self):
        self.last_epoch_weights_update = [x/self.n_samples for x in self.last_weights_update]
        self.last_epoch_biases_update = [x/self.n_samples for x in self.last_biases_update]
        self.last_weights_update = None
        self.last_biases_update = None
        
        self.n_samples = 0
    
    
    def get_updated_parameters(self, weights, biases, neuron_values, gradient_values):
        gradient_values = gradient_values[::-1]
        
        updated_weights = [None] * len(weights)
        updated_biases = [None] * len(biases)
        
        if self.last_weights_update is None and self.last_biases_update is None:
            self.last_weights_update = [None] * len(weights)
            self.last_biases_update = [None] * len(biases)
            for index in range(len(weights) - 1, -1, -1):
                self.last_weights_update[index] = np.zeros(shape=weights[index].shape)
                self.last_biases_update[index] = np.zeros(shape=biases[index].shape)
        
        for index in range(len(weights) - 1, -1, -1):
            weight_update, bias_update = self.calculate_update_from_grad(gradient_values[index], neuron_values[index])

            if self.last_epoch_weights_update is not None and self.last_epoch_biases_update is not None:
                weight_update = (self.momentum_rate * self.last_epoch_weights_update[index]) - (self.learning_rate * weight_update)
                bias_update = (self.momentum_rate * self.last_epoch_biases_update[index]) - (self.learning_rate * bias_update)
                
                self.last_weights_update[index] += weight_update
                self.last_biases_update[index] += bias_update
                
                weight_update = -(self.momentum_rate) * self.last_epoch_weights_update[index] + (1 + self.momentum_rate) * weight_update
                bias_update = -(self.momentum_rate) * self.last_epoch_biases_update[index] + (1 + self.momentum_rate) * bias_update
            else:
                weight_update = -self.learning_rate * weight_update
                bias_update = -self.learning_rate * bias_update
                
                self.last_weights_update[index] += weight_update
                self.last_biases_update[index] += bias_update
                            
            updated_weights[index] = weights[index] + weight_update
            updated_biases[index] = biases[index] + bias_update
            
        self.n_samples += 1#gradient_values[len(weights) - 1].shape[0]
        
        return updated_weights, updated_biases
  
class AdagradOptimizer(Optimizer):
    def __init__(self, epsilon, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        self.last_epoch_weights_grad_cache = None
        self.last_epoch_biases_grad_cache = None
        self.last_weights_grad_cache = None
        self.last_biases_grad_cache = None
        
    def on_epoch_end(self):
        if self.last_epoch_weights_grad_cache is None or self.last_epoch_biases_grad_cache is None:
            self.last_epoch_weights_grad_cache = [None] * len(self.last_weights_grad_cache)
            self.last_epoch_biases_grad_cache = [None] * len(self.last_biases_grad_cache)
            for index, x in enumerate(self.last_weights_grad_cache):
                self.last_epoch_weights_grad_cache[index] = np.zeros(shape=self.last_weights_grad_cache[index].shape)
                self.last_epoch_biases_grad_cache[index] = np.zeros(shape=self.last_biases_grad_cache[index].shape)
                print(self.last_epoch_weights_grad_cache[index].shape, self.last_weights_grad_cache[index].shape)
        
        for index, x in enumerate(self.last_weights_grad_cache):
            self.last_epoch_weights_grad_cache[index] += self.last_weights_grad_cache[index]
            self.last_epoch_biases_grad_cache[index] += self.last_biases_grad_cache[index]
        self.last_weights_grad_cache = None
        self.last_biases_grad_cache = None

    def get_updated_parameters(self, weights, biases, neuron_values, gradient_values):
        gradient_values = gradient_values[::-1]
        
        updated_weights = [None] * len(weights)
        updated_biases = [None] * len(biases)
        
        if self.last_weights_grad_cache is None and self.last_biases_grad_cache is None:
            self.last_weights_grad_cache = [None] * len(weights)
            self.last_biases_grad_cache = [None] * len(biases)
            for index in range(len(weights) - 1, -1, -1):
                self.last_weights_grad_cache[index] = np.zeros(shape=weights[index].shape)
                self.last_biases_grad_cache[index] = np.zeros(shape=biases[index].shape)        
        
        for index in range(len(weights) - 1, -1, -1):
            weight_update, bias_update = self.calculate_update_from_grad(gradient_values[index], neuron_values[index])
            self.last_weights_grad_cache[index] += weight_update ** 2
            self.last_biases_grad_cache[index] += bias_update ** 2 
            
            if self.last_epoch_weights_grad_cache is not None and self.last_epoch_biases_grad_cache is not None:
                weight_update = -(self.learning_rate * weight_update) / (np.sqrt(self.last_epoch_weights_grad_cache[index] + self.epsilon))
                bias_update = -(self.learning_rate * (bias_update)) / (np.sqrt(self.last_epoch_biases_grad_cache[index] + self.epsilon))
            else:
                weight_update = -self.learning_rate * weight_update / (np.sqrt(weight_update**2 + self.epsilon))
                bias_update = -self.learning_rate * bias_update / (np.sqrt(bias_update**2 + self.epsilon))
                
            updated_weights[index] = weights[index] + weight_update
            updated_biases[index] = biases[index] + bias_update
        
        return updated_weights, updated_biases   
    

class AdadeltaOptimizer(Optimizer):
    def __init__(self, epsilon=1e-8, decay_rate=0.9, learning_rate=0.01):
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        self.last_weights_grad_cache = None
        self.last_biases_grad_cache = None
    
    def on_epoch_end(self):
        pass
        
    def get_updated_parameters(self, weights, biases, neuron_values, gradient_values):
        gradient_values = gradient_values[::-1]
        
        updated_weights = [None] * len(weights)
        updated_biases = [None] * len(biases)
        
        if self.last_weights_grad_cache is None and self.last_biases_grad_cache is None:
            self.last_weights_grad_cache = [None] * len(weights)
            self.last_biases_grad_cache = [None] * len(biases)
            for index in range(len(weights) - 1, -1, -1):
                self.last_weights_grad_cache[index] = np.zeros(shape=weights[index].shape)
                self.last_biases_grad_cache[index] = np.zeros(shape=biases[index].shape)      
                
                
        for index in range(len(weights) - 1, -1, -1):
            weight_update, bias_update = self.calculate_update_from_grad(gradient_values[index], neuron_values[index])
            
            self.last_weights_grad_cache[index] = self.decay_rate * self.last_weights_grad_cache[index] + (1 - self.decay_rate) * weight_update**2
            self.last_biases_grad_cache[index] = self.decay_rate * self.last_biases_grad_cache[index] + (1 - self.decay_rate) * bias_update**2

            if self.last_weights_grad_cache is not None and self.last_biases_grad_cache is not None:
                weight_update = weight_update / (np.sqrt(self.last_weights_grad_cache[index] + self.epsilon))
                bias_update = bias_update / (np.sqrt(self.last_biases_grad_cache[index] + self.epsilon))
            
            updated_weights[index] = weights[index] - (self.learning_rate * weight_update)
            updated_biases[index] = biases[index] - (self.learning_rate * bias_update)
            
            
        return updated_weights, updated_biases


class AdamOptimizer(Optimizer):
    def __init__(self, epsilon=1e-8, learning_rate=0.01, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.m_w = None
        self.v_w = None
    
        self.m_b = None
        self.v_b = None
    
    def get_updated_parameters(self, weights, biases, neuron_values, gradient_values):
        gradient_values = gradient_values[::-1]
        
        updated_weights = [None] * len(weights)
        updated_biases = [None] * len(biases)
        
        if self.m_w is None and self.v_w is None and self.m_b is None and self.v_b is None:
            self.m_w = [0] * len(weights)
            self.v_w = [0] * len(biases)
            self.m_b = [0] * len(weights)
            self.v_b = [0] * len(biases)
                
        for index in range(len(weights) - 1, -1, -1):
            n_samples = gradient_values[index].shape[0]
            weight_update, bias_update = self.calculate_update_from_grad(gradient_values[index], neuron_values[index])
            
            self.m_w[index] = self.beta1 * self.m_w[index] + (1 - self.beta1) * weight_update
            self.v_w[index] = self.beta2 * self.v_w[index] + (1 - self.beta2) * (weight_update ** 2)
            
            self.m_b[index] = self.beta1 * self.m_b[index] + (1 - self.beta1) * weight_update
            self.v_b[index] = self.beta2 * self.v_b[index] + (1 - self.beta2) * (weight_update ** 2)
            
            weight_update = self.m_w[index] / (np.sqrt(self.v_w[index] + self.epsilon))
            weight_update = self.m_b[index] / (np.sqrt(self.v_b[index] + self.epsilon))
                
            updated_weights[index] = weights[index] - (self.learning_rate * weight_update)
            updated_biases[index] = biases[index] - (self.learning_rate * bias_update)
            
            
        return updated_weights, updated_biases
