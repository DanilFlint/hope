import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re
import matplotlib.pyplot as plt
from polynomial_regression import regression
import random

def read(path):
    f = open(path)  # Заменить на нужное имя файла. Файл подаётся без шапки и конца.
    # Только со значениями. Пример в репозитории
    l = [line.strip() for line in f]
    array = []  # Список списков.
    # Каждый элемент array - список значений [X,Y,Z]
    for i in range(1, len(l), 4):
        s = [float(s) for s in re.findall(r'-?\d+\.?\d*', l[i])]
        s[0] = s[0] * (10 ** (s[1]))
        del s[1]
        s[1] = s[1] * (10 ** s[2])
        del s[2]
        s[2] = s[2] * (10 ** s[3])
        del s[3]
        array.append(s)
    return array

def function(X, B):
    return B[0] + B[1]*X[:,1]+B[2]*X[:,2]
def get_coord(X, B):

    #Для получения координат необходимо решить квадратное уравнение
    D = (B[3]*X[:,1] + B[2])**2 - 4*B[4]*(B[1]*X[:,1] + X[:,1]**2 + B[0])
    Y1 =  np.array([[i] for i in (-(B[3]*X[:,1] + B[2]) - D**(1/2))/(2*B[4])])
    Y2 =  np.array([[i] for i in (-(B[3]*X[:,1] + B[2]) + D**(1/2))/(2*B[4])])

    bottom = np.array(np.concatenate((X[:,[1]], Y1), axis=1))
    up = np.array(np.concatenate((X[:,[1]], Y2), axis=1))
    ans = np.array(np.concatenate((bottom, up), axis=0))
    return ans


if __name__ == '__main__':

    coords = np.array(read('./Входные данные/kviila2.txt'))[:,[0,1]]/10000000 #Для меньшего размера графика делим.
    mpl.rcParams['legend.fontsize'] = 10

    #X = np.array([coords[:,0], coords[:,1]])
    X = np.concatenate((np.ones((coords.shape[0], 1)), coords), axis=1)  # Матрицана влияющих величин с единицами

    noise_X = np.array([random.uniform(-1, 1) for i in range(X.shape[0])])
    noise_Y = np.array([random.uniform(-1, 1) for i in range(X.shape[0])])

    dis_X = np.amax(abs(noise_X))/abs(np.amax(abs(X[:, 1])))*100
    dis_Y = np.amax(abs(noise_Y))/abs(np.amax(abs(X[:, 2])))*100

    X[:, 1] += noise_X
    X[:, 2] += noise_Y

    print(dis_X)
    print(dis_Y)

    B = regression(X, (np.array([[i] for i in -X[:,1]**2])))

    R = function(X, B)

    print("Вычисленные параметры: ")
    print(B)

    solve_Y = function(X,B)

    x = np.arange(1, X.shape[0] + 1, 1)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta = np.linspace(-50000 * np.pi, 50000 * np.pi, 10000000)
    ax.scatter(X[:,1], X[:,2], -X[:,1]**2, label='Экспериментальные данные',s=0.7)
    ax.scatter(X[:,1], X[:,2], solve_Y, label='Вычисленные данные',s=0.7)
    ax.legend()

    plt.show()