import numpy as np


SIGMOID_BOUND = 300


# altered sigmoid activation function, range is (-1, 1)
def sigmoid(z):
    return 2/(1 + np.exp(-z)) - 1


# normalize a 2d array by row
def normalize(points):
    # for every point, find its magnitude and normalize to a unit vector
    for i in range(len(points)):
        magnitude = 0
        for coordinate in points[i]:
            magnitude += coordinate ** 2
        magnitude = magnitude ** (1/2)
        points[i] = [num / magnitude for num in points[i]]
    return points

