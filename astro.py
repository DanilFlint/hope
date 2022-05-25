import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re
import matplotlib.pyplot as plt
from polynomial_regression import regression

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
    return np.dot(X,B) + X[:,1]**2

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

    coords = np.array(read('./kviila2.txt'))[:,[0,1]]/10000000 #Для меньшего размера графика делим.
    mpl.rcParams['legend.fontsize'] = 10

    #X = np.array([coords[:,0], coords[:,1]])
    X = np.concatenate((np.ones((coords.shape[0], 1)), coords), axis=1)  # Матрицана влияющих величин с единицами

    mul = np.array([[i] for i in X[:,1]*X[:,2]])# произведение координат x и y
    X = np.concatenate((X, mul), axis=1)

    mul = np.array([[i] for i in X[:,2]*X[:,2]]) # квадрат координат y
    X = np.concatenate((X, mul), axis=1)

    B = regression(X, (np.array([[i] for i in -X[:,1]**2])))

    R = function(X, B)

    ans = get_coord(X,B)

    fig = plt.figure('Орбита астероида 9162 Kwiila (1987 OA)')
    plt.plot(X[:,1], X[:,2], label='Орбита астероида 9162 Kwiila (1987 OA)')
    plt.plot(ans[:,0], ans[:,1])
    plt.legend()

    plt.show()