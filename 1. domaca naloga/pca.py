import numpy as np
from numpy import linalg as la
import csv
from sklearn import datasets
from matplotlib import pyplot as plt

CONVERGENCE = 1 * pow(10, -10)

class EigenPCA:

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.explained_variance_ = np.array([])
        self.components_ = np.array([[]])
        self.explained_variance_ratio_ = np.array([])
        self.mean_ = 0

    def fit(self, X):
        X = center(self, X)
        S = cov(X)
        self.explained_variance_, self.components_ = la.eigh(S)
        # arrange in descending order
        self.explained_variance_ = self.explained_variance_[::-1]
        self.components_ = self.components_.T[::-1]*(-1)
        # remove unnecessary components
        self.components_ = self.components_[:self.n_components]
        self.explained_variance_ = self.explained_variance_[:self.n_components]
        get_ratio(self)

    def transform(self, X):
        if (self.mean_ is not None):
            X = X - self.mean_
        return np.dot(X, self.components_.T)

class PowerPCA:

    def __init__(self, X, n_components=3):
        self.n_components = n_components
        self.explained_variance_ = np.array([])
        self.components_ = np.array([np.zeros(X.shape[1])])
        self.explained_variance_ratio_ = np.array([])
        self.mean_ = 0

    def fit(self, X):
        X = center(self, X)
        S = cov(X)
        for i in range(self.n_components):
            eig_val, eig_vec = self.calculate_eigen(S)
            self.explained_variance_ = np.append(self.explained_variance_, eig_val)
            self.components_ = np.append(self.components_, [eig_vec], axis=0)
            S = S - eig_val * np.outer(eig_vec, eig_vec)
        # remove zero vector from initialization
        self.components_ = np.delete(self.components_, 0, 0)
        get_ratio(self)
        return self.components_

    def transform(self, X):
        if (self.mean_ is not None):
            X = X - self.mean_
        return np.dot(X, self.components_.T)

    def calculate_eigen(self, S):
        x = np.random.rand(S.shape[0])
        epsilon = 1
        step = 0
        while (epsilon > CONVERGENCE and step < 1000):
            step += 1
            x_tmp = S.dot(x)
            x_tmp = x_tmp / la.norm(x_tmp)
            epsilon = la.norm(x_tmp - x)
            x = x_tmp
        return get_eigenvalue(S, x), x


class OrtoPCA:

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.explained_variance_ = np.array([])
        self.components_ = np.array([[]])
        self.explained_variance_ratio_ = np.array([])
        self.mean_ = 0

    def fit(self, X):
        X = center(self, X)
        S = cov(X)
        U = np.random.rand(self.n_components, S.shape[0])
        Utmp = np.zeros((self.n_components, S.shape[0]))
        U = self.norm_matrix(U)
        while(abs(U-Utmp).max() > CONVERGENCE):
            Utmp = np.copy(U)
            U = U.dot(S)
            for i in range(1, self.n_components):
                for j in range(i):
                    U[i] = self.gram_schmidt(U[j], U[i])
            U = self.norm_matrix(U)
        self.components_ = U
        for vect in U:
            self.explained_variance_ = np.append(self.explained_variance_, get_eigenvalue(S, vect))
        get_ratio(self)

    def transform(self, X):
        if (self.mean_ is not None):
            X = X - self.mean_
        return np.dot(X, self.components_.T)

    def gram_schmidt(self, u, v):
        return v - ((v.T.dot(u))/u.T.dot(u)) * u

    def norm_matrix(self, X):
        return np.array([np.array(v/la.norm(v)) for v in X])


def center(self, X):
    self.mean_ = X.mean(axis=0)
    return X - X.mean(axis=0)


def cov(X):
    return (X.T.dot(X)) / X.shape[0]


def get_eigenvalue(X, eigenvector):
    return eigenvector.T.dot(X.dot(eigenvector))


def get_ratio(self):
    total = self.explained_variance_.sum()
    self.explained_variance_ratio_ = self.explained_variance_ / total


def draw_plot(X, eign_vec, label):
    X = X.dot(eign_vec.T)
    plt.plot(X, 'ro')
    plt.show()
    print('shape: ', X.shape)


def read_file(file_path):
    f = open(file_path, "rt", encoding="UTF-8")
    reader = csv.reader(f, delimiter=",")
    next(reader)                            # preskocimo glavo tabele
    data = [d for d in reader]
    data = np.array(data).astype(np.float)  # string to float
    label = data[:, 0]
    data = np.delete(data, 1, 1)
    print('shape: ', data.shape)
    return label, data


if __name__ == "__main__":
    label, X = read_file('train.csv')
    print(label)
    pca = PowerPCA(X, 2)
    eign_vec = pca.fit(X)
    draw_plot(X, eign_vec, label)
