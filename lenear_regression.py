import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def function(X):
    #res = X[:,1]**3 + np.sin(X[:,2])# На две влияющие величины
    #res = X[:,1] + X[:,2]**2 + X[:,3]**3# На три влияющие величины
    res = (np.sin(X[:,1])*X[:,2])**X[:,3]*X[:,4]# На четыре влияющие величины
    return res

def solve_lenear_regression(X,B):
    res = np.dot(X,B)
    return res

if __name__ == '__main__':

    Count_experiment = 20
    Count_watch_greatness = 4

    X = np.random.sample((Count_experiment,Count_watch_greatness))
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)# Матрицана влияющих величин с единицами
    Y = np.array([[i] for i in function(X)]) # Столбец зависящих величин
    B =  np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)), X.transpose()),Y)# Формула линейной регрессии

    solve_Y = solve_lenear_regression(X, B)[:,0]

    incline = ((solve_Y - Y[:, 0]) ** 2)  # Вектор квадратов разности
    print("Вектор квадратов разности : ", incline)
    print("Сумма квадратов разности : ", incline.sum())

    x = np.arange(1, X.shape[0]+1, 1)
    plt.scatter(x, solve_Y, label='Решения Y')
    plt.scatter(x, Y[:,0], label='Экспериментальные Y')
    plt.legend()
    plt.show()