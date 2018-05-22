import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import csv


def k_dist(X, metric="euclidean", k=4):
    """
    function return array of distances for k-th neighbours.

    :param X: matrix with data
    :param metric: metric for measuring distance
    :param k: min number of neighbours
    :return: distance to k-th neighbour order ascending
    """
    distances = []
    for i in range(X.shape[0]):
        distances.append(getDistance(X, i, k))
    return np.sort(distances)


def getDistance(X, i, k):
    """
    method return distance to x-th neighbour of X[i].

    :param X: matrix with data
    :param i: index
    :param k: k-th neighbour
    :return: distance to k-th neighbour
    """
    tree = KDTree(X)
    dist, _ = tree.query([X[i]], k=k+1)
    return dist[0][k]


def draw_data(X):
    """
    draw the k-dist plot and scatter plot of groups.

    :param X: matrix of data
    :return: none
    """
    dist = k_dist(X, k=3)
    plt.plot(dist)
    plt.text(700, dist[700], 'k=3')

    dist = k_dist(X, k=7)
    plt.plot(dist)
    plt.text(800, dist[700], 'k=7')

    dist = k_dist(X, k=13)
    plt.plot(dist)
    plt.text(900, dist[700], 'k=13')
    plt.title('k-dist plot')
    plt.ylabel('dist')
    plt.xlabel('num')
    plt.savefig('k-dist.pdf')

    plt.close()

    dbs = DBSCAN(eps=0.045, min_samples=7)
    clusters = dbs.fit_predict(X)
    colors = ["#ffffff", "#33cc33", "#ccff33", "#0033cc", "#cc33ff",
              "#ff6600", "#ff0000", "#663300", "#660033", "#ff00ff",
              "#00ffff", "#666699", "#333300", "#669999", "#0000cc"]
    for i, g in enumerate(clusters):
        plt.scatter(X[i][0], X[i][1], color=colors[int(g) + 1], edgecolors='black')
    plt.title('eps=0.045, min samples=7')
    plt.savefig('groups.pdf')



def read_file(file_path):
    """
    Read the data from csv file.

    :param file_path: path to the csv file
    :return: matrix of data
    """
    f = open(file_path, "rt", encoding="UTF-8")
    reader = csv.reader(f, delimiter=",")
    next(reader)
    next(reader)
    next(reader)
    data = [d for d in reader]
    data = np.array(data).astype(np.float)  # string to float
    return data


class DBSCAN:

    def __init__(self, eps=0.1, min_samples=4, metric="euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric


    def fit_predict(self, X):
        """
        functon take the two dimensional data and return the array of dots sorted in groups.

        :param X: matrix of data
        :return: array of group labels
        """
        c = 0
        labels = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            if (labels[i] == 0):
                neighbours = self.find_neighbours(X, i)
                if (len(neighbours) < self.min_samples):
                    labels[i] = -1                          #oznacimo osamelca
                    continue
                c += 1
                labels[i] = c
                for q in neighbours:
                    if (labels[int(q)] == 0):
                        labels[int(q)] = c
                        neighbours_q = self.find_neighbours(X, q)
                        if (len(neighbours_q) >= self.min_samples):
                            neighbours = self.union(neighbours, neighbours_q)
        return labels


    def find_neighbours(self, X, i):
        """
        Function that return array of neighbours which distance to core is less then epsilon.

        :param X: matrix of data
        :param i: index
        :return: return the array of neighbours
        """
        tree = KDTree(X)
        dist, ind = tree.query([X[i]], k=X.shape[0])
        neighbours = []
        for distance, index in zip(dist[0], ind[0]):
            if (distance <= self.eps):
                neighbours.append(index)
        return neighbours


    def union(self, a, b):
        """
        Perform the union of two arrays so that it keep the order.

        :param a: array 1
        :param b: array 2
        :return: union of arrays
        """
        for el in b:
            if (el not in a):
                a.append(el)
        return a


if __name__ == '__main__':
    # X = read_file('data.csv')
    # draw_data(X)
    pass