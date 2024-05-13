import numpy as np

def softmax(x):
    # x.shape = (batch_dim, input_dim)
    e_x = np.exp(x) # (batch_dim, input_dim)
    e_x_sum = e_x.sum(axis=1) # (batch_dim, )
    
    # e_x.shape     = (batch_dim, input_dim)
    # e_x_sum.shape = (batch_dim, )
    # -> need 2nd axis for element wise division
    e_x_sum = np.expand_dims(e_x_sum, axis=1)

    return e_x / e_x_sum  # element wise division ;)

def cross_entropy(predictions, targets):

    epsilon = 0.000001 # trick: avoid log(0) by adding a tiny number inside log
    N = predictions.shape[-1]
    ce = -np.sum(targets*np.log(predictions+epsilon), axis=-1)/N

    return ce

def sigmoid(x):
    return 1/( 1 + np.exp(-x) )

def sigmoid_prime(x):
    y = sigmoid(x)
    return y * (1 - y)

