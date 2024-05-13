import numpy as np

from MultiLayerPerceptron import *

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def getData(batch_size):

    # Load data
    data, labels = load_digits(return_X_y=True)
    num_datapoints = data.shape[0]
    num_classes = 10

    # Shuffle
    shuffle_indexes = np.random.permutation(np.arange(0, num_datapoints))
    data = data[shuffle_indexes]
    labels = labels[shuffle_indexes]
    
    # Batching via reshape
    num_batches = num_datapoints//batch_size

    # Throw number of data points away which do not fit into the last batch
    num_datapoints = data.shape[0]
    num_datapoints_fitted_to_batch = num_datapoints - (num_datapoints%batch_size)

    data = data[:num_datapoints_fitted_to_batch]
    data = np.reshape(data, newshape=(num_batches, batch_size, -1))

    labels = labels[:num_datapoints_fitted_to_batch]
    labels = np.reshape(labels, newshape=(num_batches, batch_size))

    for img, label in zip(data, labels):
        
        img = img.astype(np.float32)
        # Documentation: "each element is an integer in the range 0..16."
        img = img/16 # -> Normalization: [0, 1]

        # Mapping: label (a number) -> target (one hot encoded vector)
        target = np.eye(num_classes)[label] 

        yield img, target


def getPerformance(mlp, batch_size):

    generator = getData(batch_size)

    loss_list = []
    accuarcy_list = []
    for img, target in generator:

        pred = mlp(img)

        # Get accuarcy
        pred_label = np.argmax(pred, axis=-1)
        target_label = np.argmax(target, axis=-1)

        matches = pred_label == target_label

        accuarcy_batch = np.sum(matches) / batch_size
        accuarcy_list.append(accuarcy_batch)
          
        # Get loss
        loss_batch = cross_entropy(pred, target)
        loss_list.append(loss_batch)

    # Average accuarcy and loss for the current epoch
    accuarcy = np.average(accuarcy_list)
    loss = np.average(loss_list)

    return accuarcy, loss


def main():

    num_epochs = 25

    # Hyperparameters
    batch_size = 16
    epsilon = 0.01 # Learning rate

    mlp = MultiLayerPerceptron([64,5,10])
    
    # Monitor loss and accuracy
    accuracy_list = []
    loss_list = []

    # Epoch = one iteration over the (train) dataset 
    for epoch in range(0, num_epochs):

        # Train
        generator = getData(batch_size=batch_size)
        for img, target in generator:
            #    img.shape = (batch_dim, 64)
            # target.shape = (batch_dim, 10)
            mlp.backprop_step(img, target, epsilon)
        
        # Get performance
        accuracy, loss = getPerformance(mlp, batch_size)
        
        accuracy_list.append(accuracy)
        loss_list.append(loss)
    
    #
    # There are two ways to display the results
    #

    #
    # First way: For both accuracy and loss, there is a subplot (Training_Accuracy_Loss_1.png)
    #
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    plt.suptitle("Accuracy and loss per epoch")

    # 1st subplot
    ax1.plot(accuracy_list, color='blue', marker='o')
    ax1.grid(True)
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")

    # 2nd subplot
    ax2.plot(loss_list, color='red', marker='o')
    ax2.grid(True)
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig("./Plots/Training_Accuracy_Loss_1.png")
    plt.show()

    #
    # Second way: Plot accuracy and loss together - two y axes (Training_Accuracy_Loss_2.png)
    # I don't like this way :)
    #
    fig, ax1 = plt.subplots()
    plt.grid()
    plt.title("Accuracy and loss per epoch")

    # 1st axis
    ax1.plot(accuracy_list, color='blue', marker='o')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.yaxis.label.set_color('blue')
    
    # 2nd axis
    ax2 = ax1.twinx() # create 2nd y axis
    ax2.plot(loss_list, color='red', marker='o')
    ax2.set_ylabel("Loss")
    ax2.yaxis.label.set_color('red')

    plt.tight_layout()
    plt.savefig("./Plots/Training_Accuracy_Loss_2.png")
    plt.show()

    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")