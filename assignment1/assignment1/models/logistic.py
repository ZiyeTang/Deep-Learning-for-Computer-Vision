"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        if np.isscalar(z):
            if z >= 0:
                return 1 / (1 + np.exp(-z))
            return 1 - 1 / (1 + np.exp(z))
        
        res = []
        for i in z:
            if i >= 0:
                res.append(1 / (1 + np.exp(-i)))
            else :
                res.append(1 - 1 / (1 + np.exp(i)))
        return np.array(res)

    def convert_label(self, y) -> int:
        return y*2-1


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        n = len(y_train)
        d = X_train.shape[1]
        self.w = np.random.randn(d)
        for j in range(self.epochs):
            self.lr = self.lr / (1 + j * 8)
            for i in range(n):
                y = self.convert_label(y_train[i])
                self.w += self.lr * self.sigmoid(-y * self.w @ X_train[i]) * y * X_train[i]


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
        return self.sigmoid(X_test @ self.w) > self.threshold
