import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def decompose_to_LU(a):
    lu_matrix = np.matrix(np.zeros([a.shape[0], a.shape[1]]))
    n = a.shape[0]
    for k in range(n):
        for j in range(k, n):
            lu_matrix[k, j] = a[k, j] - lu_matrix[k, :k] * lu_matrix[:k, j]
        for i in range(k + 1, n):
            lu_matrix[i, k] = (a[i, k] - lu_matrix[i, : k] * lu_matrix[: k, k]) / lu_matrix[k, k]
    return lu_matrix

def get_L(m):
    L = m.copy()
    for i in range(L.shape[0]):
            L[i, i] = 1
            L[i, i+1 :] = 0
    return np.matrix(L)

def get_U(m):
    U = m.copy()
    for i in range(1, U.shape[0]):
        U[i, :i] = 0
    return U

def solve_LU(lu_matrix, b):
    y = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
    for i in range(y.shape[0]):
        y[i, 0] = b[i, 0] - lu_matrix[i, :i] * y[:i]
    x = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
    for i in range(1, x.shape[0] + 1):
        x[-i, 0] = (y[-i] - lu_matrix[-i, -i:] * x[-i:, 0] )/ lu_matrix[-i, -i]
    return x


def squere(resolve,x):
    max_degree = resolve.shape[0] - 1
    res = 0
    for i in range(max_degree+1):
        res += resolve[i,0]*x**(max_degree-i)
    return res

if __name__ == '__main__':

    N = 5  # Степень полинома

    X = np.array([i for i in range(1,7)])# Вектор X - значений
    Y = np.array([1.0, 1.5, 3.0, 4.5, 7.0, 8.5])# Вектор Y - значений

    A = np.matrix(np.zeros([N+1, N+1])) #Матрица для решения СЛАУ и для LU разложения
    for i in range(N+1):
        for j in range(N+1):
            degree = (2*N-j)-i
            A[i,j] = (X**degree).sum()

    B = np.matrix(np.zeros([N + 1, 1])) # Сумма вектора произведений элементов X и Y
    for i in range(N + 1):
        degree = N - i
        B[i, 0] = (X ** degree * Y).sum()

    o = decompose_to_LU(A)  # LU - разложение в одну матрицу

    Resolve = np.matrix(np.zeros([N + 1, 1])) #Вектор решений

    Resolve = solve_LU(o, B)

    x = np.arange(1, 7, 0.1)
    plt.scatter(X, Y)
    plt.plot(x, squere(Resolve, x))

    plt.show()