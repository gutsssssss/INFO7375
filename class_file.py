import cv2
import glob
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle


# class Dataset:
#     def __init__(self, img_height, img_width, num_classes):
#         # define the size and classes number
#         self.img_height = img_height
#         self.img_width = img_width
#         self.num_classes = num_classes
#         # create lists to restore img and labels
#         self.X = []  # img
#         self.y = []  # label
#
#     def loadData(self, address):
#         # get img and label
#         for folder in glob.glob(address):
#             label = folder[-1]
#             label = int(label)
#             for img_path in glob.glob(folder + '/*.png'):
#                 img = plt.imread(img_path)
#                 img = cv2.resize(img, (self.img_height, self.img_width))
#                 self.X.append(img)
#                 self.y.append(label)
#         # list to numpy
#         self.X = np.array(self.X).reshape(100, -1)
#         self.y = one_hot_encode(np.array(self.y), self.num_classes)


class Neurons:
    def __init__(self, ds, alpha):
        # Define the number of features, the number of classes, and the learning rate
        self.ds = ds
        self.num_features = self.ds.X.shape[1]  # Number of columns in X
        self.num_classes = self.ds.num_classes  # Number of possible output values
        self.alpha = alpha  # Learning rate
        # Initialize the weights and the bias randomly
        self.W = np.random.randn(self.num_features, self.num_classes)
        self.b = np.random.randn(self.num_classes)


class Training:
    def __init__(self, ds, model, epochs):
        self.ds = ds
        self.model = model
        self.epochs = epochs
        self.losses = []

    def run(self):
        # Train the model using gradient descent
        for epoch in range(self.epochs):
            # Loop over each training example
            for x, y_true in zip(self.ds.X, self.ds.y):
                # Forward pass
                # Compute the weighted sum
                z = np.dot(x, self.model.W) + self.model.b
                # Apply the activation function
                y_pred = sigmoid(z)
                # Compute the loss
                loss = categorical_crossentropy(y_true, y_pred)
                # Append the loss to the list
                self.losses.append(loss)

                # Backward pass
                # Compute the gradient of the loss with respect to the prediction
                grad_y_pred = categorical_crossentropy_derivative(y_true, y_pred)
                # Compute the gradient of the prediction with respect to the weighted sum
                grad_z = sigmoid_derivative(z) * grad_y_pred
                # Compute the gradient of the weighted sum with respect to the weights
                grad_W = np.outer(x, grad_z)
                # Compute the gradient of the weighted sum with respect to the bias
                grad_b = grad_z
                # Update the weights and the bias
                self.model.W = self.model.W - self.model.alpha * grad_W
                self.model.b = self.model.b - self.model.alpha * grad_b

            # Print the epoch number and the loss value
            print(f'Epoch {self.epochs}, Loss: {loss}')


class Test:
    def __init__(self, model, tset):
        self.predictions = []
        self.model = model
        self.tset = tset

    def run(self):
        for x in self.tset:
            # Compute the weighted sum
            z = np.dot(x, self.model.W) + self.model.b
            # Apply the activation function
            y_pred = sigmoid(z)
            # Convert the output to a class label
            prediction = np.argmax(y_pred)
            # Append the prediction to the list
            self.predictions.append(prediction)
        # Print the predictions and the true labels
        print('Predictions:', self.predictions)
        print('True labels:', np.argmax(self.tset.y, axis=1))


def plot_loss_curve(training):
    # Plot the loss curve
    matplotlib.use('TkAgg')
    plt.plot(training.losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()


# Define the activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Define the loss function and its derivative
# You can change this to other functions, such as mse, cross_entropy, etc.
def categorical_crossentropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))


def categorical_crossentropy_derivative(y_true, y_pred):
    return y_pred - y_true


# Define the one-hot encoding function
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]
