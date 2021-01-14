import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import lu
from scipy.stats import uniform
from scipy.linalg import hilbert
from pprint import pprint
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from random import random
from dense import DenseMatrix
from csr import CSRMatrix
from block import *


def create_diagonally_dominant_matrix(n):
    a = np.random.randint(-4, 1, size=(n, n)).astype(float)
    for i in range(n):
        off_diag_sum = abs(sum([a[i][j] for j in range(n) if i != j]))
        a[i, i] = off_diag_sum
    return a.astype(float)


def generate_random_answers(n):
    answers = []
    for _ in range(n):
        answer = random()
        answers.append(answer)
    return answers


# arr = np.array(hilbert(7))
# dm = DenseMatrix(arr)
# title = "hilbert matrix. Cond = "

dm = DenseMatrix()
dm.read_from_file('matrix_7_7_sparse.txt')
title = "Matrix_7_7_sparse. Cond = "

# dm = DenseMatrix()
# dm.read_from_file('matrix_10_10.txt')
# title = "Matrix_10_10. Cond = "

print("Original: ")
dm.print_matrix()

print("\nLU merged: ")
dm.print_matrix(dm.lu_matrix)

print("Inverse matrix: ")
dm.print_matrix(dm.inverse())

print("Original matrix * inverse matrix: ")
dm.print_matrix(np.dot(dm.matrix, dm.inverse()))

dm.plot()
dm.plot(dm.lu_matrix)
dm.plot(dm.inverse())

expected = np.array([-1. / 3, 1. / 3, 0, 2.2, 4, 1, 10], float)

normX = np.linalg.norm(expected)
b = np.dot(dm.matrix, expected)
normB = np.linalg.norm(b)
cond = np.linalg.cond(dm.matrix)

print("Expected answers: ", expected)
print("Answers received through LU factorization: ", dm.solve_lu(b))

table = PrettyTable()
title += str(cond)
table.field_names = ["10**(-k)", "norm(x{LU} - x)", "norm(x{INV} - x)", "norm(dx)/norm(x)", "norm(db)/norm(b)"]

noise = np.array([0, 0, 0, 0, 0, 0, 0], float)
list_DX = []
list_DB = []

for k in np.arange(-10, 0, 1.0):
    computedLU = dm.solve_lu(b + noise)
    computedINV = np.dot(dm.inverse(), b + noise)

    normLU = np.linalg.norm(computedLU - expected)
    normINV = np.linalg.norm(computedINV - expected)
    normDB = np.linalg.norm(noise)
    normDX = normLU / normX

    table.add_row([abs(k), normLU, normINV, normDX, normDB / normB])
    list_DX.append(normDX)
    list_DB.append(normDB / normB)

    noise = 10 ** k

print(table.get_string(title=title))

sm = CSRMatrix()
sm.read_from_file('matrix_7_7_sparse.txt')

print("Inverse matrix: ", sm.inverse())
print("Sparsity= ", sm.sparsity)

print("Expected answers: ", expected)
print("Answers received through LU factorization for CSR format: ", sm.solve_lu(b))

ddm = create_diagonally_dominant_matrix(100)
expected = generate_random_answers(100)

normX = np.linalg.norm(expected)

table = PrettyTable()
title = "Dependency of accuracy metrics from noise for dense matrix "
table.field_names = ["10**(-k)", "Cond", "norm(x{LU} - x)", "norm(dx)/norm(x)"]

for k in np.arange(-10, 1, 1):
    dm = DenseMatrix(ddm)
    dm.diag_revolt(k)
    b = np.dot(dm.matrix, expected)
    computedLU = dm.solve_lu(b)
    cond = np.linalg.cond(dm.matrix)
    normLU = np.linalg.norm(computedLU - expected)
    normDX = normLU / normX
    table.add_row([abs(k), cond, normLU, normDX])

print(table.get_string(title=title))

matrix = read_file('matrix_10_10.txt')
# matrix = read_file('matrix_6_6_dense.txt')

expected = np.array([-1./3, 1./3, 0, 2.2, 4, 1, 10, 2, 5, 0], np.double)
# expected = np.array([-1./3, 1./3, 0, 2.2, 4, 1], np.double)

b = np.dot(matrix, expected)

print("Right vector b: ", b)

x = np.linalg.solve(matrix, b)

print("Scipy x: ", x)

block_size = 2  # must be multiple to matrix.shape

lu_block_naive = naive_blu(matrix, block_size)
l_lu = np.tril(lu_block_naive, -1) + np.eye(len(lu_block_naive))
u_lu = np.triu(lu_block_naive)

print("Lower triangular matrix: ", l_lu)
print("Upper triangular matrix: ", u_lu)

y = forward_substitution(l_lu, b)
x = back_substitution(u_lu, y)

print("Block LU x: {}\n for block size: {} ".format(x, block_size))

