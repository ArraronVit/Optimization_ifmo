import numpy as np
import matplotlib.pyplot as plt


class DenseMatrix:
    def __init__(self, matrix=None):
        if matrix is None:
            self.n = 0
        else:
            self.n = matrix.shape[0]
            self.matrix = np.array(matrix, float)
            self.lu_matrix = self.lu()

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
        LU = self.matrix.copy()
        for k in range(self.n - 1):
            t = self.__gauss(LU[k:, k])
            LU[k + 1:, k] = t
            LU[k:, k + 1:] = self.__gauss_app(LU[k:, k + 1:], t)
        return LU

    def solve_lu(self, bb):
        b = bb.copy()
        b = np.array(b, float)
        for i in range(1, len(b)):
            b[i] = b[i] - np.dot(self.lu_matrix[i, :i], b[:i])
        for i in range(len(b) - 1, -1, -1):
            b[i] = (b[i] - np.dot(self.lu_matrix[i, i + 1:], b[i + 1:])) / self.lu_matrix[i, i]
        return b

    def inverse(self):
        E = np.eye(self.n)
        inv = []
        for e in E:
            x = self.solve_lu(e) # inverse matrix column
            inv.append(x)
        return np.array(inv).T

    def read_from_file(self, file_name=None):
        self.matrix = np.loadtxt(file_name)
        self.matrix = np.array(self.matrix, float)
        self.n = self.matrix.shape[0]
        self.lu_matrix = self.lu()

    def print_matrix(self, matrix=None):
        if matrix is None:
            matrix = self.matrix
        for row in matrix:
            for x in row:
                print("{:.3f}".format(x), end=" ")
            print()
        print()

    def plot(self, matrix=None):
        if matrix is None:
            matrix = self.matrix
        plt.spy(matrix, precision=0., markersize=3)
        plt.show()

    def diag_revolt(self, k):
        for i in range(self.n):
            self.matrix[i, i] += 10. ** k
        self.lu_matrix = self.lu()
