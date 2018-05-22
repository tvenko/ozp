from time import time

import numpy as np
from numpy import linalg as la
import Orange
import gzip
import csv

m = np.array([[1,1],[2,2],[1,2],[2,1]], np.double)

def pca_full(X):
    """
    Calculate full PCA transformation by using Numpy's np.linalg.eigh.

    Args:
        X (np.ndarray): Data matrix of shape [n_examples, n_features]

    Returns:
        eigenvectors (np.ndarray): Array of shape [n_features, n_features]
            containing eigenvectors as columns. The ith column should
            contain eigenvector corresponding to the ith largest eigenvalue.
    """

    X = center(X)
    print(la.eigh(X))
    pass


def gram_schmidt_orthogonalize(vecs):
    """
    Gram-Schmidt orthonormalization of column vectors.

    Args:
        vecs (np.adarray): Array of shape [n_features, k] with column
            vectors to orthogonalize.

    Returns:
        Orthonormalized vectors of the same shape as on input.
    """
    pass


def pca_2d(X, eps=1e-5):
    """
    Calculate the first two components of PCA transformation by using
    the power method with Gram-Schmidt orthonormalization.

    Args:
        X (np.ndarray): Data matrix of shape [n_examples, n_features]
        eps (float): Stopping criterion threshold for Frobenius norm.

    Returns:
        eigenvectors (np.ndarray): Array of shape [n_features, 2]
            containing the eigenvectors corresponding to the largest and
            the second largest eigenvalues.
    """
    pass


def project_data(X, vecs):
    """
    Project the data points in X into the subspace of eigenvectors.

    Args:
        X (np.ndarray): Data matrix of shape [n_examples, n_features]
        vecs: Array of shape [n_features, k] containing the eigenvectors.

    Returns:
        np.ndarray: Projected data of shape [n_examples, k].
    """
    pass

def center(X):
    center = np.sum(X, axis=0)/X.shape[0]
    # print('center: ', center)
    # print("shape0: ", X.shape[0], " shape1: ", X.shape[1])
    for i in range(0, X.shape[1]):
        X[:, i] = X[:, i] - center[i]
    return X

def read_file(file_path):
    #funkcija za branje podtkov iz datoteke
    f = open(file_path, "rt", encoding="UTF-8")
    reader = csv.reader(f, delimiter=",")
    next(reader)                                    #preskocimo glavo tabele
    data = [d for d in reader]
    data = np.array(data).astype(np.float)          #string to float
    return data


if __name__ == '__main__':
    # data = Orange.data.Table('mnist-1k.tab')
    data = read_file("train.csv")

    t1 = time()
    vecs_np = pca_full(data)
    print('Full time: {:.4f}s'.format(time() - t1))
    transformed_numpy = project_data(data, vecs_np[:, :2])

    # t1 = time()
    # vecs_pow = pca_2d(data)
    # print('2D time: {:.4f}s'.format(time() - t1))
    # transformed_power = project_data(data, vecs_pow)
