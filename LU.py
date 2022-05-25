import numpy as np

def decompose_to_LU(a):
    lu_matrix = np.matrix(np.zeros([a.shape[0], a.shape[1]]))
    n = a.shape[0]
    for k in range(n):
        for j in range(k, n):
            lu_matrix[k, j] = a[k, j] - lu_matrix[k, :k] * lu_matrix[:k, j]
        for i in range(k + 1, n):
            lu_matrix[i, k] = (a[i, k] - lu_matrix[i, : k] * lu_matrix[: k, k]) / lu_matrix[k, k]
    return lu_matrix

def solve_LU(lu_matrix, b):
    y = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
    for i in range(y.shape[0]):
        y[i, 0] = b[i, 0] - lu_matrix[i, :i] * y[:i]
    x = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
    for i in range(1, x.shape[0] + 1):
        x[-i, 0] = (y[-i] - lu_matrix[-i, -i:] * x[-i:, 0] )/ lu_matrix[-i, -i]
    return x








"""def decompose_to_LU(a):
    L = np.eye(a.shape[0], a.shape[1])
    U = a.copy()
    n = a.shape[0]#Количество строк
    for i in range(n):
        for j in range(i+1,n):
            L[j,i] = a[j,i]/a[i,i]
            U[j,:] = U[j,:] - L[j,i] * U[i,:]
    check = np.dot(L,U)
    return L + U - np.eye(a.shape[0], a.shape[1])"""