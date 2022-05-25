import numpy as np
import matplotlib.pyplot as plt
from LU import decompose_to_LU, solve_LU

def function(resolve,x):
    max_degree = resolve.shape[0] - 1
    res = 0
    for i in range(max_degree+1):
        res += resolve[i,0]*x**(max_degree-i)
    return res

if __name__ == '__main__':
    # Степень полинома
    N = 5

    # Вектор X - значений
    X = np.array([i for i in range(1,7)])

    # Вектор Y - значений
    Y = np.array([1.0, 1.5, 3.0, 4.5, 7.0, 8.5])

    # Матрица для решения СЛАУ и для LU разложения
    A = np.matrix(np.zeros([N+1, N+1]))
    for i in range(N+1):
        for j in range(N+1):
            degree = (2*N-j)-i
            A[i,j] = (X**degree).sum()

    # Сумма вектора произведений элементов X и Y
    B = np.matrix(np.zeros([N + 1, 1]))
    for i in range(N + 1):
        degree = N - i
        B[i, 0] = (X ** degree * Y).sum()

    # LU - разложение в одну матрицу
    LU = decompose_to_LU(A)

    # Вектор решений
    Resolve = solve_LU(LU, B)

    print("Столбец значений:")
    for i in Y:
        print(i)
    Y2 = function(Resolve, X)

    incline = ((Y2 - Y) ** 2)
    print("Вектор квадратов разности : ", incline)
    print("Сумма квадратов разности : ", incline.sum())

    plt.scatter(X, Y)
    plt.plot(X, Y2)

    plt.show()