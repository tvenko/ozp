import numpy as np
from zlib import compress


class SVM:

    def __init__(self, C, kernel, epochs, rate):
        '''
        initialize parameters

        :param C: cost
        :param kernel: string name of kernel
        :param epochs: number of iterations
        :param rate: intensity of learning
        '''
        self.C = C
        self.kernel = kernel
        self.epochs = epochs
        self.rate = rate

    def fit(self, X, y):
        '''
        Fits SVM on data

        :param X: data
        :param y: labels
        '''
        a = np.zeros(X.shape[0])
        y = (y * 2) - 1
        kernel = {}
        for i, x in enumerate(X):
            kernel[i] = self.get_kernel(x, X)
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                delta = self.rate * (1 - y[i] * np.sum(a * y * kernel[i]))
                a[i] = min(self.C, max(0, a[i] + delta))
        self.X = X
        self.y = y
        self.coef_ = a

    def predict(self, X):
        '''
        Predict on test data

        :param X: test data
        '''
        r = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            for j in range(self.X.shape[0]):
                r[i] += self.coef_[j] * self.y[j] * self.get_kernel(x, self.X[j])
        return (np.sign(r) + 1) / 2

    def get_weights(self):
        '''
        returns weights for linear kernel
        '''
        weights = np.zeros(self.X.shape[1])
        for i in range(len(self.coef_)):
            weights += self.coef_[i] * self.X[i] * self.y[i]
        return weights

    def get_kernel(self, x1, x2):
        '''
        returns a linear, rbf or text kernel\

        :param x1: case in data
        :param x2: data
        '''
        if self.kernel == 'linear':
            return np.dot(x1, x2.T)
        if self.kernel == 'rbf':
            return np.exp(-np.linalg.norm(x2 - x1) ** 2 / 2)
        else:
            if x2.ndim == 0:
                x2 = [x2]
            k = np.empty(len(x2))
            zip_a = len(compress(x1.encode('utf-8')))
            for i, b in enumerate(x2):
                zip_b = len(compress(b.encode('utf-8')))
                k[i] = 100 - 0.5 * ((len(compress(x1.encode('utf-8') + b.encode('utf-8'))) - zip_a) / zip_a + (len(compress(b.encode('utf-8') + x1.encode('utf-8'))) - zip_b) / zip_b)
            return k


if __name__ == '__main__':

    from sklearn import datasets
    import glob
    import matplotlib.pyplot as plt


    def add_ones(X):
        return np.column_stack((np.ones(len(X)), X))

    def generate_data(data_type, n_samples=200):
        ''' generate data for blobs and circle'''
        np.random.seed(70)
        if data_type == "blobs":
            X, y = datasets.make_blobs(
                n_samples=n_samples,
                centers=[[2, 2], [1, 1]],
                cluster_std=0.4
            )
        elif data_type == "circle":
            X = (np.random.rand(n_samples, 2) - 0.5) * 20
            y = (np.sqrt(np.sum(X ** 2, axis=1)) > 8).astype(int)

        X = add_ones(X)
        return X, y

    def get_text_data(origin="text-data"):
        ''' get text samples from train and test folders'''
        dirs = glob.glob(origin + "/*")
        X, y = [], []
        for i, d in enumerate(dirs):
            files = glob.glob(d + "/*")
            for file_name in files:
                with open(file_name, "rt", encoding="utf8") as file:
                    X.append(" ".join(file.readlines()))
            y.extend([i] * len(files))
        return np.array(X), np.array(y)


    def plot(clf, min, max):
        ''' draw a plot '''
        colors = ["r" if y == 1 else "b" for y in clf.y]
        plt.scatter(clf.X[:, 1], clf.X[:, 2], s=(clf.coef_ * 70) + 10, c=colors)

        x, y = np.meshgrid(np.linspace(min, max, 100), np.linspace(min, max, 100))
        z = clf.predict(add_ones(np.c_[x.ravel(), y.ravel()]))
        z = z.reshape(x.shape)

        plt.contourf(x, y, z, alpha=0.4)
        return plt

    X, y = generate_data("blobs")
    svm = SVM(C=1, rate=0.001, epochs=5000, kernel="linear")
    svm.fit(X, y)
    p = plot(svm, -0.5, 3.5)
    p.savefig('blobs.pdf')

    X, y = generate_data("circle", n_samples=300)
    svm = SVM(C=1, rate=0.001, epochs=5000, kernel="rbf")
    svm.fit(X, y)
    p = plot(svm, -11, 11)
    p.savefig('circle.pdf')

    X_train, y_train = get_text_data("data/train")
    X_test, y_test = get_text_data("data/test")
    svm = SVM(C=1, rate=0.001, epochs=5000, kernel="text")
    svm.fit(X_train, y_train)

    errors = np.sum(np.abs(svm.predict(X_test) - y_test))
    CA = 1 - (errors / len(y_test))
    print("CA: %f" % CA)

