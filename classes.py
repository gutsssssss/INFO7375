import pickle

import cv2
import matplotlib
import numpy as np
import arguments as arg
import math_functions as mf
from matplotlib import pyplot as plt
import glob
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class Dataset:
    def __init__(self, address):
        # define the size and classes number
        self.img_height = arg.img_height
        self.img_width = arg.img_width
        self.num_classes = arg.num_classes
        self.address = address
        # create lists to restore img and labels
        self.x = []  # img
        self.y = []  # label

    def loadData(self):
        # get img and label
        for folder in glob.glob(self.address):
            label = folder[-1]
            label = int(label)
            for img_path in glob.glob(folder + '/*.png'):
                img = plt.imread(img_path)
                img = cv2.resize(img, (self.img_height, self.img_width))
                self.x.append(img)
                self.y.append(label)
        # list to numpy
        self.x = np.array(self.x).reshape(len(self.x), -1)
        self.y = one_hot_encode(np.array(self.y), self.num_classes)
        return DataLoader(self.toTorchDataset(), batch_size=arg.batch_size, shuffle=True)

    def toTorchDataset(self):
        x = torch.tensor(self.x)
        y = torch.tensor(self.y)
        return TensorDataset(x, y)


class Neuron:
    def __init__(self, num_inputs: int):
        self.weights = torch.randn(num_inputs, 1)
        self.gradients_w = torch.zeros(num_inputs, 1)

    def update(self, learning_rate: float, normalization: str = None) -> None:
        if normalization is None:
            self.weights -= torch.mul(learning_rate, self.gradients_w)
        if normalization == 'L2':
            self.weights = torch.mul(1 - learning_rate * arg.lamda, self.weights) - torch.mul(learning_rate,
                                                                                              self.gradients_w)

    def forward(self, inputs: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        neuron_output = torch.mm(inputs, self.weights) + bias
        return neuron_output

    def backward(self, gradient_z: torch.Tensor, inputs: torch.Tensor):
        self.gradients_w = torch.mm(torch.transpose(inputs, 0, 1), gradient_z) / len(gradient_z)


class Layer:
    def __init__(self, num_neurons: int, num_inputs: int, activation: str):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
        self.inputs = torch.zeros(arg.batch_size, num_neurons)
        self.activation_functions = activation_functions[activation]
        self.activation_derivative = activation_derivatives[activation]
        self.bias = torch.randn(1)
        self.z = torch.zeros(arg.batch_size, num_neurons)
        self.gradients_bias = torch.zeros(num_neurons)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.inputs = inputs
        for i in range(len(self.neurons)):
            self.z[:, i] = self.neurons[i].forward(self.inputs, self.bias).squeeze()
        self.z = torch.Tensor(self.z)
        output = self.activation_functions(self.z)
        return output

    def backward(self, gradients_a: torch.Tensor = None, y_true: torch.Tensor = None) -> torch.Tensor:
        if gradients_a is not None:
            gradients_z = torch.mul(self.activation_derivative(self.z), gradients_a).type(torch.float32)
        elif y_true is not None:
            gradients_z = self.activation_derivative(self.z, y_true).type(torch.float32)
        else:
            raise Exception("No gradient comes in!")
        for i, neuron in enumerate(self.neurons):
            neuron.backward(gradients_z[:, i].reshape(-1, 1), self.inputs)
        weights = self.collectWeight()
        gradients_next_a = torch.mm(gradients_z, torch.transpose(weights, 0, 1))
        return gradients_next_a

    def updateBias(self, learning_rate: float):
        self.bias -= torch.mul(learning_rate, torch.mean(self.gradients_bias))

    def updateWeight(self, learning_rate: float, normalization: str = None):
        for i in range(len(self.neurons)):
            self.neurons[i].update(learning_rate, normalization=normalization)

    def collectWeight(self) -> torch.Tensor:
        list_w = [n.weights for n in self.neurons]
        return torch.cat(list_w, dim=1)


class Model:
    def __init__(self, train_loader: DataLoader):
        self.train_loader = train_loader
        self.layers = []
        self.losses = []
        self.normalization = arg.normalization
        self.init_layers()

    def init_layers(self):
        self.layers.append(Layer(200, arg.size_input, 'relu'))
        self.layers.append(Layer(100, 200, 'relu'))
        self.layers.append(Layer(50, 100, 'relu'))
        self.layers.append(Layer(20, 50, 'relu'))
        self.layers.append(Layer(10, 20, 'softmax'))

    def forward(self, inputs: torch.Tensor, y_true: torch.Tensor):
        layer_output = [inputs]
        if len(self.layers) > 0:
            for i in range(len(self.layers)):
                layer_output.append(self.layers[i].forward(layer_output[i]))
            loss = mf.cross_entropy(layer_output[-1], y_true)
            return loss, layer_output[-1]
        else:
            raise Exception("No defined layers!")

    def backward(self, y_true: torch.Tensor):
        gradient = [self.layers[-1].backward(y_true=y_true)]
        for i in range(len(self.layers) - 1):
            gradient.append(self.layers[-i - 2].backward(gradients_a=gradient[i]))

    def update_all(self, learning_rate: float, normalization=None):
        for i in range(len(self.layers)):
            self.layers[i].updateWeight(learning_rate, normalization=normalization)
            self.layers[i].updateBias(learning_rate)

    def train(self):
        epochs, initial_lr, decay_rate = arg.epochs, arg.initial_lr, arg.decay_rate
        lr_arr = mf.lrArray(epochs, initial_lr, decay_rate)
        for epoch in range(epochs):
            for batch in self.train_loader:
                x, y_true = batch
                loss, _ = self.forward(x, y_true)
                self.losses.append(loss)
                self.backward(y_true)
                self.update_all(lr_arr[epoch], normalization=self.normalization)

            # Print the epoch number and the loss value
            print(f'Epoch {epoch}, Loss: {self.losses[epoch * len(self.train_loader)]}')

    def test(self, test_loader: DataLoader):
        acc = []
        for batch in test_loader:
            x, y_t = batch
            _, y_pred = self.forward(x, y_t)
            predicted_labels = torch.argmax(y_pred, dim=1)
            true_labels = torch.argmax(y_t, dim=1)
            accuracy = calculateAccuracy(predicted_labels, true_labels)
            acc.append(accuracy)
            # Print the epoch number and the accuracy
            print(f'Accuracy: {accuracy * 100}%')

    # def l2Normalization(self):
    #     sqr_weight = 0
    #     for i in range(5):
    #         w = self.layers[i].collectWeight()
    #         sqr_weight += np.sum(np.square(w))
    #     return 0.5 * sqr_weight


# Define the one-hot encoding function
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]


def plot_loss_curve(model):
    # Plot the loss curve
    matplotlib.use('TkAgg')
    plt.plot(model.losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()


# save model
def saveModel(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


# 加载模型参数
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def calculateAccuracy(pred_label, true_label):
    acc = 0
    for i in range(len(pred_label)):
        if pred_label[i] == true_label[i]:
            acc += 1
    return acc / len(pred_label)


# 创建一个字典来存储激活函数
activation_functions = {
    'linear': mf.linear,
    'sigmoid': mf.sigmoid,
    'relu': mf.relu,
    'tanh': mf.tanh,
    'softmax': mf.softmax
}

# 创建一个字典来存储激活函数的导数
activation_derivatives = {
    'linear': mf.linear_derivative,
    'sigmoid': mf.sigmoid_derivative,
    'relu': mf.relu_derivative,
    'tanh': mf.tanh_derivative,
    'softmax': mf.softmax_cross_entropy_derivative
}
