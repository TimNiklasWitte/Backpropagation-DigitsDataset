from Util import *
import numpy as np

class Layer:

    def __init__(self, input_dim, output_dim, activation_func):
        
        self.weights = np.random.normal(size=(input_dim, output_dim))
        self.bias = np.random.normal(size=(output_dim, ))
        
        # Function pointer
        self.activation_func = activation_func  
        
        # Weight and bias updates happen in adapt
        self.grad_weights = None
        self.grad_bias = None 
        

    def __call__(self, x):

        drive = x @ self.weights + self.bias
        activation = self.activation_func(drive)

        return activation
    

    def backward(self, delta, a_prev_layer):
        
        #
        # Get gradient for weights
        #
        
        #   a_prev_layer.shape = (batch_dim, input_dim)
        # a_prev_layer.T.shape = (input_dim, batch_dim)
        #          delta.shape = (batch_dim, output_dim)
        
        #           (input_dim, batch_dim) @ (batch_dim, output_dim)
        self.grad_weights = a_prev_layer.T @ delta 
        #   weights.shape = (input_dim, output_dim)


        #
        # Get gradient for bias
        #

        # delta.shape = (batch_dim, output_dim)
        # bias.shape  = (output_dim, )
        grad_bias = delta 
        grad_bias = np.average(grad_bias, axis=0)  # Average gradient

        self.grad_bias = grad_bias # (output_dim, )

        
    
    def adapt(self, epsilon):
        # weights.shape = grad_weights.shape
        self.weights = self.weights - epsilon * self.grad_weights

        # bias.shape = grad_bias.shape
        self.bias = self.bias - epsilon * self.grad_bias

    