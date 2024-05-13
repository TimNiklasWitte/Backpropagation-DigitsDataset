from Layer import *

class MultiLayerPerceptron:

    def __init__(self, units):
        
        self.layer_list = [None] # Input layer

        # Hidden layers
        self.num_layers = len(units) 
        for idx_layer in range(self.num_layers - 2):
            input_dim = units[idx_layer]
            output_dim = units[idx_layer + 1]

            layer = Layer(input_dim, output_dim, activation_func=sigmoid)
            self.layer_list.append(layer)

        # Output layer
        input_dim = units[-2]
        output_dim = units[-1]
        output_layer = Layer(input_dim, output_dim, activation_func=softmax)
        self.layer_list.append(output_layer)
        
    def __call__(self, x):
        # Forward pass
        for layer in self.layer_list[1:]: # Ignore input layer
            x = layer(x)
        
        return x
    
    def backprop_step(self, x, target, epsilon):
        
        #
        # Record activations of each layer
        #

        a = [x] # Network input
        for layer in self.layer_list[1:]: # Ignore input layer
            x = layer(x)
            a.append(x)
        y = x # Prediction
        
        #
        # Gradient taping: Calc gradients and save them for the next step
        # Only calculation, do not change/update weights and bias in this step! 
        # -> If you would do this here you would distort the gradients
        #

        # Difference to regression (here: classification!)
        #sigmoid_prime = a[-1] * (1 - a[-1])
        #delta = 2*(y - target) * sigmoid_prime

        # Get first delta
        delta = y - target # Element-wise operation
        output_layer = self.layer_list[-1]
        output_layer.backward(delta, a[-2])
     
        for l in reversed(range(1, self.num_layers - 1)):
            
            sigmoid_prime = a[l] * (1 - a[l]) # Element-wise operation

            #           delta.shape = (batch_dim, output_dim)
            #   weights^{L+1}.shape = (input_dim, output_dim)
            # weights^{L+1}.T.shape = (output_dim, input_dim)

            # (batch_dim, output_dim) @ (output_dim, input_dim) -> (batch_dim, input_dim)
            delta = (delta @ self.layer_list[l + 1].weights.T) * sigmoid_prime
            # notice: " * sigmoid_prime" is element-wise (aka hadamard product)

            self.layer_list[l].backward(delta, a[l - 1])


        #
        # Update layers (weights and bias)
        #

        for layer in self.layer_list[1:]: # Ignore input layer
            layer.adapt(epsilon)



    