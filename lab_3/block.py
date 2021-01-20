import numpy as np
import matplotlib.pyplot as plt


def __forward_substitution(L, b):
    # Get number of rows
    n = L.shape[0]

    # Allocating space for the solution vector
    y = np.zeros_like(b, dtype=np.double)

    # Here we perform the forward-substitution.
    # Initializing  with the first row.
    y[0] = b[0] / L[0, 0]

    # Looping over rows in reverse (from the bottom  up),
    # starting with the second to last row, because  the
    # last row solve was completed in the last step.
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y


def __back_substitution(U, y):
    # Number of rows
    n = U.shape[0]
    # Allocating space for the solution vector
    x = np.zeros_like(y, dtype=np.double)

    # Here we perform the back-substitution.
    # Initializing with the last row.
    x[-1] = y[-1] / U[-1, -1]

    # Looping over rows in reverse (from the bottom up),
    # starting with the second to last row, because the
    # last row solve was completed in the last step.
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i, i]

    return x


def solve_lu(matrix, b, block_size=1):
    lu_block_naive = naive_blu(matrix, block_size)
    l_lu = np.tril(lu_block_naive, -1) + np.eye(len(lu_block_naive))
    u_lu = np.triu(lu_block_naive)

    # print("Lower triangular matrix: ", l_lu)
    # print("Upper triangular matrix: ", u_lu)

    y = __forward_substitution(l_lu, b)
    x = __back_substitution(u_lu, y)
    return x


def __lu(A):
    # Get the number of rows
    n = A.shape[0]

    U = A.copy()
    L = np.eye(n, dtype=np.double)

    # Loop over rows
    for i in range(n):
        # Eliminate entries below i with row operations
        # on U and reverse the row operations to
        # manipulate L
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]

    return L, U


def plot(matrix):
    plt.spy(matrix, precision=0., markersize=3)
    plt.show()


def read_file(file_name):
    matrix = np.loadtxt(file_name).astype(np.double)
    return matrix


def naive_blu(original_matrix, n_blocks):
    idx = [0]  # vector that indicates the first index in each block of matrix's data
    matrix = original_matrix.copy()
    idx = list(range(0, matrix.shape[0], n_blocks))  # forming list with step n_blocks

    for j in range(0, len(idx)):
        # !! single non matrix operation, all the rest ara matrix operations
        l_11, u_11 = __lu(__factor(idx[j], matrix, n_blocks))  # LU decomposition for Factor submatrix A11

        matrix[idx[j]:idx[j] + n_blocks, idx[j]:idx[j] + n_blocks] = np.tril(l_11, -1) + u_11

        # Compute blocks columns of L and rows of U
        if idx[j] == idx[-1]:
            break  # if it's last block

        a_21 = matrix[idx[j + 1]:, idx[j]:idx[j + 1]]
        a_12 = matrix[idx[j]:idx[j + 1], idx[j + 1]:]
        a_22 = matrix[idx[j + 1]:, idx[j + 1]:]

        matrix[idx[j + 1]:, idx[j]:idx[j + 1]] = np.dot(a_21, np.linalg.inv(u_11))
        matrix[idx[j]:idx[j + 1], idx[j + 1]:] = np.dot(np.linalg.inv(l_11), a_12)
        # Schur complement
        matrix[idx[j + 1]:, idx[j + 1]:] = a_22 - np.dot(a_21, a_12)
        __visualize_step(idx, original_matrix, n_blocks, j)

    return matrix


def __factor(inx, matrix, n_blocks):  # Factor A11 for each iteration
    a11 = matrix[inx:inx + n_blocks, inx:inx + n_blocks]
    return np.array(a11)


def __visualize_step(idx, matrix, n_blocks, position):
    portrait = np.zeros_like(matrix)
    portrait.fill(4)
    portrait[idx[position]:idx[position] + n_blocks, idx[position]:idx[position] + n_blocks].fill(5)  # A11
    if idx[position] != idx[-1]:
        portrait[idx[position + 1]:, idx[position]:idx[position + 1]].fill(2)  # a21
        portrait[idx[position]:idx[position + 1], idx[position + 1]:].fill(2)  # a12
        portrait[idx[position + 1]:, idx[position + 1]:].fill(1)  # a22

    plt.imshow(portrait)
    plt.show()
