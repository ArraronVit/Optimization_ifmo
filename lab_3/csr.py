import numpy as np
import matplotlib.pyplot as plt


class CSRMatrix:

    def __init__(self, regular_matrix=None):
        self.regular_matrix = np.array(regular_matrix, float)
        self.diag = np.array([], int)
        if regular_matrix is None:
            self.n = 0
            self.data = np.array([], float)
            self.columns = np.array([], int)
            self.rowIndices = np.array([], int)
            self.sparsity = None
        else:
            self.n = self.regular_matrix.shape[0]
            self.data, self.columns, self.rowIndices = self.__to_csr()
            self.sparsity = 1. - np.count_nonzero(self.regular_matrix) / self.regular_matrix.size

    def solve_lu(self, b):
        y = [0.0] * self.n

        # solve for U matrix
        for i in range(0, self.n):  # range(1, self.n)
            s1 = self.rowIndices[i]
            s2 = self.rowIndices[i + 1]
            y[i] = b[i]
            for k in range(s1, s2):  # foreach elem in str
                if self.columns[k] < i:  # traversal through U
                    y[i] = y[i] - self.data[k] * y[self.columns[k]]

        for i in range(self.n - 1, -1, -1):  # range(self.n, 1, -1)
            s1 = self.rowIndices[i]
            s2 = self.rowIndices[i + 1]
            for k in range(s1, s2):
                if self.columns[k] > i:  # traversal through L
                    y[i] = y[i] - self.data[k] * y[self.columns[k]]

            y[i] = y[i] / self.data[self.diag[i]]

        return y

    def __to_csr(self):
        data = []
        columns = []  # list stores the column index of each element in the data.
        rowIndices = []
        diag = []
        count = 0
        rowIndices.append(count)
        for i in range(self.n):
            for j in range(self.n):
                if self.regular_matrix[i, j] != 0:
                    data.append(self.regular_matrix[i, j])
                    columns.append(j)

                    if i == j:
                        diag.append(count)
                    count += 1

            rowIndices.append(count)

        self.diag = np.array(diag, int)

        return np.array(data, float), np.array(columns, int), np.array(rowIndices, int)

    def plot(self, matrix=None):
        if matrix is None:
            matrix = self.regular_matrix
        plt.spy(matrix, precision=0., markersize=3)
        plt.show()

    def read_from_file(self, file_name=None):
        self.regular_matrix = np.loadtxt(file_name)
        self.regular_matrix = np.array(self.regular_matrix, float)
        self.n = self.regular_matrix.shape[0]
        self.regular_matrix = self.lu()
        self.data, self.columns, self.rowIndices = self.__to_csr()
        self.sparsity = 1. - np.count_nonzero(self.regular_matrix) / self.regular_matrix.size

    def restore_from_csr(self, data=None, columns=None, rowIndices=None):
        if data is None:
            data = self.data
        if columns is None:
            columns = self.columns
        if rowIndices is None:
            rowIndices = self.rowIndices

        n = rowIndices.size - 1
        matrix = np.zeros([n, n], float)
        for i in range(n):
            for j in range(rowIndices[i], rowIndices[i + 1]):
                matrix[i, columns[j]] = data[j]
        return matrix

    def sparse_eye(self, data=None, columns=None, rowIndices=None):

        if data is None and columns is None and rowIndices is None:
            data = self.data
            columns = self.columns
            rowIndices = self.rowIndices
        else:
            pass

        for i in range(self.n):
            for j in range(rowIndices[i], rowIndices[i + 1]):
                data[j] = 1. if columns[j] == i else 0.

        return data, columns, rowIndices

    @staticmethod
    def __gauss(x):
        x = np.array(x, float)
        return x[1:] / x[0]

    @staticmethod
    def __gauss_app(C, t):
        C = np.array(C, float)
        t = np.array([[t[i]] for i in range(len(t))], float)
        C[1:, :] = C[1:, :] - t * C[0, :]
        return C

    def lu(self):
        LU = self.regular_matrix.copy()
        for k in range(self.n - 1):
            t = self.__gauss(LU[k:, k])
            LU[k + 1:, k] = t
            LU[k:, k + 1:] = self.__gauss_app(LU[k:, k + 1:], t)
        return LU

    def inverse(self):
        E = np.eye(self.n)
        inv = []
        for e in E:
            x = self.solve_lu(e)
            inv.append(x)
        return np.array(inv).T

    def diag_revolt(self, k):
        for i in range(self.diag.size):
            self.data[self.diag[i]] += 10. ** k
        for i in range(self.n):
            self.regular_matrix[i, i] += 10. ** k
