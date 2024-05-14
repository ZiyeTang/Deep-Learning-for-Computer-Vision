"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        decay_rate = 8
        self.w = np.random.randn(self.n_class, X_train.shape[1])
        for j in range(self.epochs):
            self.lr = self.lr / (1 + j * decay_rate)
            for i in range(len(y_train)):
                for c in range(self.n_class):
                    if self.w[c] @ X_train[i] > self.w[y_train[i]] @ X_train[i]:
                        self.w[int(y_train[i])] += self.lr * X_train[i] 
                        self.w[c] -= self.lr * X_train[i] 
            

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        return np.argmax(self.w @ X_test.T, axis=0)
