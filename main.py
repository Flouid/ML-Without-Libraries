import numpy as np
import utilities as util
import sys


# global hyper-parameters for training the network
DIMENSIONS = 1000
HIDDEN_UNITS = 2
OUTPUT_UNITS = 1
LEARNING_RATE = 1
EPOCHS = 10


# main driver code
def init():
    # create the network with two layers as specified by global parameters
    network = [generate_layer(HIDDEN_UNITS, DIMENSIONS), generate_layer(OUTPUT_UNITS, HIDDEN_UNITS)]

    # load the test and training data sets
    test_data, test_labels = get_data('a2-test-data.txt', 'a2-test-label.txt')
    train_data, train_labels = get_data('a2-train-data.txt', 'a2-train-label.txt')

    # train the network
    train(network, train_data, train_labels, EPOCHS)

    # generate a set of predictions for the test data and report accuracy
    predictions = generate_predictions(network, test_data)
    accuracy = calculate_accuracy(test_labels, predictions)
    print(f'The calculated accuracy was {100*accuracy}%')

    # generate deliverables
    write_network(network)
    write_predictions(predictions)


# ========
# TRAINING
# ========


def train(network, data, labels, epochs):
    for epoch in range(epochs):
        print(f'EPOCH {epoch+1}', end=' ')
        for i in range(len(data)):
            prediction = tensor_flow(network, data[i])
            ys = get_activations(network, data[i])
            weight_update(network, ys, prediction, labels[i])
        print('COMPLETE')


# get the network activations (y values) for each layer
def get_activations(network, x):
    ys = [x]
    for i in range(len(network)):
        z = np.dot(network[i], ys[-1])
        ys.append(util.sigmoid(z))
    return ys


# perform a weight update using the activations from a point
def weight_update(network, activations, prediction, label):
    # if the prediction matches the label, don't perform a weight update
    if prediction == label:
        return
    # start at the output layer of the network and work backwards
    for layer_idx in reversed(range(len(network))):
        # perform a weight update for every node in the layer
        for node_idx in range(len(network[layer_idx])):
            # the activations data structure is one layer deeper than the network data structure, adjust here
            y = activations[layer_idx + 1][node_idx]
            # formula for the partial derivative of "y = 2/(1 + e^-z) - 1" in terms of itself
            node_delta = (prediction - label)*0.5*(y + 1)*(1 - y)
            # perform an update on every weight coming into the node
            for weight_idx in range(len(network[layer_idx][node_idx])):
                activation = activations[layer_idx][weight_idx]
                network[layer_idx][node_idx][weight_idx] -= LEARNING_RATE * node_delta * activation


# ==========
# ALGORITHMS
# ==========


def calculate_accuracy(labels, predictions):
    correct = 0
    length = len(labels)
    for i in range(length):
        if labels[i] == predictions[i]:
            correct += 1
    return correct / length


# runs the neural network on an input data set and returns the accuracy of the classifier
def generate_predictions(network, data):
    predictions = []
    # for each point, the prediction will be the result of tensor flow on the network and point
    for i in range(len(data)):
        predictions.append(tensor_flow(network, data[i]))
    return predictions


# runs tensor flow algorithm to get an output value from a neural network and an input point
def tensor_flow(network, x):
    # set the initial output of the previous layer y to be the input vector x
    y = x
    # runs tensor flow algorithm
    for layer_idx in range(len(network)):
        z = np.dot(network[layer_idx], y)
        y = util.sigmoid(z)
    # map output to {-1, 1} by checking the sign
    if y[0] >= 0:
        return 1
    else:
        return -1


# ===============================
# NEURAL NETWORK HELPER FUNCTIONS
# ===============================

# generate a layer of a network with a u units each with w weights
def generate_layer(u, w):
    node_weights = np.empty((u, w))
    for node_idx in range(u):
        for weight_idx in range(w):
            node_weights[node_idx][weight_idx] = np.random.uniform(low=0, high=1)
    return node_weights


# print a layer of a network
def print_layer(layer):
    for node in layer:
        for weight in node:
            print(weight, end=' ')
        print(end='\n')


# print a network
def print_network(network):
    for i in reversed(range(len(network))):
        print_layer(network[i])


# =============
# DATA HANDLING
# =============


# writes the predictions for the network to an output file
def write_predictions(predictions):
    with open('a2-test-predictions.txt', 'w') as f:
        for prediction in predictions:
            f.write(str(prediction) + ' ')


# writes the network to an output file
def write_network(network):
    original_stdout = sys.stdout
    with open('a2-network.txt', 'w') as f:
        sys.stdout = f
        print(HIDDEN_UNITS)
        print_network(network)
        sys.stdout = original_stdout


# read a file and return an array of data
def get_data(data_file, label_file):
    # open data file
    with open(data_file) as f:
        # create one array to store the points and another to store the labels
        data = []
        while True:
            # read a line from the file, strip trailing whitespace, and split delimited by spaces
            line = f.readline().strip().split(' ')
            # if the line was empty, there are no more lines to read
            if line == ['']:
                break
            # store the point
            data.append([float(num) for num in line])
    # open label file, works as above
    with open(label_file) as f:
        labels = []
        while True:
            line = f.readline().strip()
            if line == '':
                break
            labels.append(int(float(line)))
    return util.normalize(data), labels


# fix the a2-test-labels so that they are the same format as the a2-train-labels
def fix_data():
    with open('a2-test-label.txt') as f:
        labels = ''
        for c in f.read().strip():
            if c != ',' and c != '[' and c != ']':
                labels += c
    with open('a2-test-label-fixed.txt', 'w') as f:
        for num in labels.split():
            f.write(num + '\n')


if __name__ == '__main__':
    init()
