from scipy.optimize import fmin_l_bfgs_b
from sklearn.neural_network import MLPClassifier
from sklearn import datasets, preprocessing
import numpy as np


class NeuralNetwork(MLPClassifier):

    def __init__(self, hidden_layer_sizes, alpha):
        '''

        :param hidden_layer_sizes: array with number of nodes in hidden layer
        :param alpha: degree of regularization
        '''
        super().__init__(hidden_layer_sizes, alpha)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.coefs_ = []
        self.n_layers_ = None
        self.n_outputs_ = None
        self.scaler = preprocessing.StandardScaler()
        self.order = []

    def set_data_(self, X, y):
        '''
        Setting all initial data.

        :param x: matrix with attribute of cases
        :param y: vector with class values
        '''
        self.X = self.scaler.fit_transform(X)
        self.y = y
        self.binarized_y = preprocessing.LabelBinarizer().fit(range(len(np.unique(y)))).transform(y)
        self.order.append(self.X.shape[1])
        for el in self.hidden_layer_sizes:
            self.order.append(el)
        self.order.append(len(np.unique(y)))

    def init_weights_(self):
        '''
        Randomly initialize weights between nodes

        :return: list of weights
        '''
        self.coefs_.append(np.random.normal(0, 1, size=[self.X.shape[1]+1, self.hidden_layer_sizes[0]]) / np.sqrt(self.hidden_layer_sizes[0]))
        for i in range(1, len(self.hidden_layer_sizes)):
            self.coefs_.append(np.random.normal(0, 1, size=[self.hidden_layer_sizes[i-1]+1, self.hidden_layer_sizes[i]]) / np.sqrt(self.hidden_layer_sizes[i]))
        self.coefs_.append(np.random.normal(0, 1, size=[self.hidden_layer_sizes[len(self.hidden_layer_sizes)-1]+1, len(np.unique(self.y))]) / np.sqrt(len(np.unique(self.y))))
        self.coefs_ = self.flatten_coefs(self.coefs_)
        return self.coefs_

    def flatten_coefs(self, coefs):
        '''
        Flatten list of weights

        :param coefs: list of weights
        :return: flattened list of weights
        '''
        return np.concatenate([el.flatten() for el in coefs])

    def unflatten_coefs(self, coefs):
        '''
        Unflatten list of weights

        :param coefs: 1D list of weight
        :return: oroginal matrix with weights
        '''
        start_index = 0
        result = []
        for i in range(1, len(self.order)):
            end_index = start_index + (self.order[i-1]+1) * self.order[i]
            result.append(np.reshape(coefs[start_index:end_index], [self.order[i-1]+1, self.order[i]]))
            start_index = end_index
        return result

    def cost(self, coefs):
        '''
        Calculate the cost

        :param coefs: weights
        :return: cost
        '''
        output_nodes = self.get_activations(self.unflatten_coefs(coefs))[-1]
        return np.sum((output_nodes - self.binarized_y)**2) / (self.X.shape[0] * 2) + (self.alpha * np.sum(coefs * self.mask(len(coefs))))

    def grad(self, coefs):
        '''
        Calculate gradients

        :param coefs: weights
        :return: list of gradients
        '''
        unflatten_coefs = self.unflatten_coefs(coefs)
        self.get_activations(unflatten_coefs)
        d = (self.activations[-1] - self.binarized_y) * (self.activations[-1] * (1 - self.activations[-1]))
        self.gradients = [np.dot(self.activations[-2].T, d) / self.X.shape[0]]

        for i in range(len(unflatten_coefs) - 1, 0, -1):
            d = (np.dot(d, unflatten_coefs[i].T) * (self.activations[i] * (1 - self.activations[i])))[:, 1:]
            self.gradients.append(np.dot(self.activations[i-1].T, d) / self.X.shape[0])

        self.gradients = self.flatten_coefs(reversed(self.gradients))
        self.gradients += self.alpha * coefs * self.mask(len(coefs))
        return self.gradients

    def grad_approx(self, coefs, e):
        '''
        Calculate approximation of gradients

        :param coefs: weight
        :param e: epsilon
        :return: list of gradients
        '''
        return np.array([(self.cost(coefs + eps) - self.cost(coefs - eps)) / (2 * e) for eps in np.identity(len(coefs)) * e])

    def fit(self, X, y):
        '''
        :param X: input data
        :param y: classes of input data
        '''
        self.set_data_(X, y)
        self.init_weights_()
        self.coefs_ = fmin_l_bfgs_b(self.cost, self.coefs_, fprime=self.grad)[0]

    def predict(self, X):
        '''
        Predict

        :param X: input data
        :return: classes
        '''
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        '''
        Returns estimates of probability
        :param X: data
        :return: probability
        '''
        return self.feed_forward(self.scaler.transform(X), self.unflatten_coefs(self.coefs_))

    def get_activations(self, coefs):
        '''
        Calculate activations for all nodes

        :param coefs: weights
        :return: list of activations
        '''
        row_len = self.X.shape[0]
        current_activation = np.append(np.ones((row_len, 1)), self.X, axis=1)
        all_activations = [current_activation]

        for row in coefs[:-1]:
            z = np.dot(current_activation, row)
            current_activation = np.append(np.ones((row_len, 1)), sigmoid(z), axis=1)
            all_activations.append(current_activation)

        current_activation = sigmoid(np.dot(current_activation, coefs[-1]))
        all_activations.append(current_activation)
        self.activations = all_activations
        return all_activations

    def feed_forward(self, X, coefs):
        '''
        feed forward
        :param X: data
        :param coefs: weights
        :return: activation of last nodes
        '''
        activation = X
        for el in coefs:
            z = np.dot(np.append(np.ones((X.shape[0], 1)), activation, axis=1), el)
            activation = sigmoid(z)
        return activation

    def mask(self, len):
        '''
        get regularization mask

        :param len: number of nodes
        :return: mask
        '''
        mask = np.ones(len)
        index = 0
        for i in range(0, self.order.__len__()-1):
            mask[index:(index + self.order[i] + 1)] = 0
            index += (self.order[i]+1) * self.order[i+1]
        return mask


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    pass