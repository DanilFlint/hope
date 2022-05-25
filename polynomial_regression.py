import numpy as np
import matplotlib.pyplot as plt
from LU import decompose_to_LU, solve_LU

def regression(X,Y):
    A = np.dot(X.transpose(), X)
    b = np.dot(X.transpose(), Y)
    # LU - разложение в одну матрицу
    LU = decompose_to_LU(A)
    # Вектор решений
    B = solve_LU(LU, b)
    return np.array(B)

def function(X):
    res = X[:,1]**3 + np.sin(X[:,2])# На две влияющие величины
    #res = X[:,1] + X[:,2]**2 + X[:,3]**3# На три влияющие величины
    #res = (np.sin(X[:,1])*X[:,2])**X[:,3]*X[:,4]# На четыре влияющие величины
    return res

def solve_regression(X,B):
    res = np.dot(X,B)
    return res

if __name__ == '__main__':

    Count_experiment = 20
    Count_watch_greatness = 2
    X = np.random.sample((Count_experiment,Count_watch_greatness))
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)# Матрицана влияющих величин с единицами

    mul = np.array([[i] for i in X[:, 1] * X[:, 2]])
    X_poly = np.concatenate((X, mul),axis=1)  # Матрицана влияющих величин с единицами и перемноженными влияющими величинами для полинома

    Y = np.array([[i] for i in function(X)])  # Столбец зависящих величин

    B_lenear = regression(X, Y)
    B_polynomical = regression(X_poly, Y)

    solve_Y_lenear = solve_regression(X, B_lenear)[:,0]
    solve_Y_polynomical = solve_regression(X_poly, B_polynomical)[:,0]

    print("Матрица наблюдений:")
    for i in X:
        print(i)

    print("Матрица наблюдений + добавленные переменные для полиномиализации:")
    for i in X_poly:
        print(i)

    print("Столбец значений:")
    for i in Y:
        print(i)

    #Линейная регрессия
    print("Линейная регрессия")
    incline = ((solve_Y_lenear - Y[:, 0]) ** 2)  # Вектор квадратов разности
    print("Вектор квадратов разности : ", incline)
    print("Сумма квадратов разности : ", incline.sum())

    #Полиномиальная регрессия
    print("Полиномиальная регрессия")
    incline = ((solve_Y_polynomical - Y[:, 0]) ** 2)  # Вектор квадратов разности
    print("Вектор квадратов разности : ", incline)
    print("Сумма квадратов разности : ", incline.sum())

    x = np.arange(1, X.shape[0]+1, 1)

    plt.figure('Линейная регрессия')
    plt.title("Линейная регрессия")
    plt.plot(x, solve_Y_lenear, label='Решения Y')
    plt.plot(x, Y[:,0], label='Экспериментальные Y')
    plt.legend()

    plt.figure('Полиномиальная регрессия')
    plt.title("Полиномиальная регрессия")
    plt.plot(x, solve_Y_polynomical, label='Решения Y')
    plt.plot(x, Y[:, 0], label='Экспериментальные Y')
    plt.legend()

    plt.show()