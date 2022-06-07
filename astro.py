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
    return B[0]+B[1]*X[:,1]+B[2]*X[:,2]+B[3]*X[:,3]+B[4]*X[:,4]

def get_coord_el(X, B):
    #Для получения координат необходимо решить квадратное уравнение
    D = (B[3]*X[:,1] + B[2])**2 - 4*B[4]*(B[1]*X[:,1] + X[:,1]**2 + B[0])
    Y1 =  np.array([[i] for i in (-(B[3]*X[:,1] + B[2]) - D**(1/2))/(2*B[4])])
    Y2 =  np.array([[i] for i in (-(B[3]*X[:,1] + B[2]) + D**(1/2))/(2*B[4])])

    print(np.array(np.concatenate((X[:,[1]],X[:, [2]], Y1,Y2), axis=1)))
    ans = np.array(np.concatenate((X[:,[1]], Y1), axis=1))
    ans = np.array(np.concatenate((ans, Y2), axis=1))
    return ans

def get_coord_plate(X, A):
    #Координаты Z

    return -(A[0] + A[1] * X[:, 1] + A[2] * X[:, 2])


if __name__ == '__main__':
    #Начало региона данных
    coords = np.array(read('./Входные данные/kviila2.txt'))[:,[0,1]]/10000000 #Для меньшего размера графика делим.
    coords_Z = np.array(read('./Входные данные/kviila2.txt'))[:,[2]]/10000000 #Для меньшего размера графика делим.
    mpl.rcParams['legend.fontsize'] = 10
    #X = np.array([coords[:,0], coords[:,1]])
    ellipse_matrix = np.concatenate((np.ones((coords.shape[0], 1)), coords), axis=1)  # Матрицана влияющих величин с единицами
    plane_matrix = np.concatenate((np.ones((coords.shape[0], 1)), coords), axis=1)

    noise_X = np.array([random.uniform(-1, 1) for i in range(ellipse_matrix.shape[0])])#Шум по X
    noise_Y = np.array([random.uniform(-1, 1) for i in range(ellipse_matrix.shape[0])])#Шум по Y
    noise_Z = np.array([random.uniform(-0.2, 0.2) for i in range(ellipse_matrix.shape[0])])#Шум по Z

    dis_X = np.amax(abs(noise_X)) / abs(np.amax(abs(ellipse_matrix[:, 1]))) * 100
    dis_Y = np.amax(abs(noise_Y)) / abs(np.amax(abs(ellipse_matrix[:, 2]))) * 100
    dis_Z = np.amax(abs(noise_Z)) / abs(np.amax(abs(coords_Z)))*100

    ellipse_matrix[:, 1] += noise_X
    ellipse_matrix[:, 2] += noise_Y
    coords_Z[:,0] += noise_Z
    #Конец региона данных

    #Регион формирования матриц наблюдений

    #Матрица для параметров эллипса
    mul = np.array([[i] for i in ellipse_matrix[:, 1] * ellipse_matrix[:, 2]])# произведение координат x и y
    ellipse_matrix = np.concatenate((ellipse_matrix, mul), axis=1)
    mul = np.array([[i] for i in ellipse_matrix[:, 2] * ellipse_matrix[:, 2]]) # квадрат координат y
    ellipse_matrix = np.concatenate((ellipse_matrix, mul), axis=1)

    #Матрица для параметров плоскости составлена, добавлений не требуется

    A = regression(plane_matrix, -coords_Z)
    B = regression(ellipse_matrix, (np.array([[i] for i in -ellipse_matrix[:, 1] ** 2])))

    #R = function(ellipse_matrix, B)

    ans = get_coord_el(ellipse_matrix, B)

    ans_plate = get_coord_plate(plane_matrix, A)

    sum_incline = 0
    for i in (ans[:,0] - ellipse_matrix[:, 1]):
        if not np.isnan(i):
            sum_incline+=i**2
    #print("Сумма квадратов разности : ", sum_incline)

    print("Вычисленные параметры A: ")
    print(A)
    print("Вычисленные параметры B: ")
    print(B)

    #Вывод графика
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(plane_matrix[:, 1], ellipse_matrix[:, 2], coords_Z, label='Экспериментальные данные', s=0.7)
    ax.scatter(ans[:, 0], ans[:, 1], ans_plate, label='Вычисленные данные',s=0.7)
    ax.legend()

    plt.show()