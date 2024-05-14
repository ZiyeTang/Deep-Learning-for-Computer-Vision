"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt = opt

        self.t = 1

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        self.m = {}
        self.v = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

            self.m["W" + str(i)] = 0
            self.m["b" + str(i)] = 0

            self.v["W" + str(i)] = 0
            self.v["b" + str(i)] = 0


    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        return X @ W + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return np.maximum(X, 0)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        return (X > 0).astype(float)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        res = []
        for i in x:
            row = []
            for j in i:
                if j >= 0:
                    row.append(1 / (1 + np.exp(-j)))
                else :
                    row.append(1 - 1 / (1 + np.exp(j)))
            res.append(row)
        return np.array(res)

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        diff = y - p
        return np.mean(np.sum(diff ** 2, axis=1))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {"h0":X}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.
        res = X.copy()
        for i in range(1, self.num_layers+1):
            res = self.linear(self.params["W" + str(i)], res, self.params["b" + str(i)])
            
            if i == self.num_layers:
                res = self.sigmoid(res)
            else: 
                res= self.relu(res)
            
            self.outputs["h" + str(i)] = res
        
        return res

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.

        hk = self.outputs["h" + str(self.num_layers)]   
        e = self.mse(y, hk)
        
        self.gradients["h" + str(self.num_layers)] = 2 / hk.shape[0] * (hk - y)

        for i in range(self.num_layers,0,-1):
            hi = self.outputs["h" + str(i)]
            hi_1 = self.outputs["h" + str(i-1)]
            wi = self.params["W"+str(i)]
            
            e_hi = self.gradients["h" + str(i)]
            if i == self.num_layers:
                e_f = hi*(1-hi)* e_hi
            else:                
                e_f = self.relu_grad(hi) * e_hi

            e_hi_1 = e_f @ wi.T
            e_wi =  hi_1.T @ e_f
            e_bi = np.sum(e_f, axis = 0)


            self.gradients["h" + str(i-1)] = e_hi_1
            self.gradients["W" + str(i)] = e_wi
            self.gradients["b" + str(i)] = e_bi
        return e

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.

        for param_name, param in self.params.items():
            if opt == "SGD":
                self.params[param_name] -= lr * self.gradients[param_name]
            elif opt == "Adam":
                # First and second moment estimates
                self.m[param_name] = b1 * self.m[param_name] + (1 - b1) * self.gradients[param_name]
                self.v[param_name] = b2 * self.v[param_name] + (1 - b2) * self.gradients[param_name] ** 2
                # Bias correction
                m_hat = self.m[param_name] / (1 - b1 ** self.t)
                v_hat = self.v[param_name] / (1 - b2 ** self.t)
                # Update parameter
                self.params[param_name] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        self.t += 1

