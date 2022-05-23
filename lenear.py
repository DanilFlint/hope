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

def lenear(a,b,x):
    return a*x+b

if __name__ == '__main__':
    X = np.array([i for i in range(1, 7)])  # Вектор X - значений
    Y = np.array([1.0, 1.5, 3.0, 4.5, 7.0, 8.5])  # Вектор Y - значений

    sum_X = X.sum() #Сумма элементов X
    sum_X_sq = np.square(X).sum() #Сумма квадратов элементов X

    A = np.array([[sum_X_sq, sum_X],[sum_X, len(X)]]) #Матрица для решения СЛАУ и для LU разложения

    Mul_X_Y = X * Y # Вектор произведений элементов X и Y

    B = np.matrix([[Mul_X_Y.sum()], [Y.sum()]]) #Сумма вектора произведений элементов X и Y

    o = decompose_to_LU(A) #LU - разложение в одну матрицу

    L = get_L(o) # Получение нижнетреугольной матрицы L
    U = get_U(o) # Получение верхнетреугольной матрицы U

    a, b = solve_LU(o, B)

    x = np.arange(1, 7, 0.1)
    plt.scatter(X, Y)
    plt.plot(x, lenear(a[0,0],b[0,0],x))

    plt.show()