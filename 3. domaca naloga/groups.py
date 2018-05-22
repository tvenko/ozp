from scipy.io import mminfo, mmread
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix

class Groups:

    def __init__(self, inputFile, outputFile):
        """
        Function that take big coo_matrix file on input, read it, eliminate the data and classify the groups from data.

        :param inputFile: file with coo_matrix
        :param outputFile: name of text file to store the labels of clusters
        """

        print(mminfo(inputFile))
        data = mmread(inputFile)
        x, y = data.shape
        newRow = []
        newCol = []
        newData = []
        # eliminate the data
        for row, col, data in zip(data.row, data.col, data.data):
            if (col < 2500):
                newRow.append(row)
                newCol.append(col)
                newData.append(1)
        cleaned_data = coo_matrix((newData, (newRow, newCol)), shape=(x, 2500))
        del data
        kmeans = KMeans(n_clusters=45).fit(cleaned_data)
        np.savetxt(outputFile, kmeans.labels_, fmt='%d', delimiter='\n', newline='\n')
        print('finish')
        pass

if __name__ == '__main__':
    Groups('train.mtx', 'output.txt')