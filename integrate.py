import re
import math
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def Function_integral(x, int):
    # x - значение, int - номер нужного интеграла
    if int == 0:
        return 1/(1+x**29)
    if int == 1:
        return (math.log10(x))/(1 +  x*x)**2
    if int == 2:
        return -x*math.log(x, math.e)/(1 + x**2)**2

# Чтение значений из файла
if __name__ == '__main__':
    step = 50
    integration_info = [[0,4, 1.00196],[0, 2, -0.355793], [0,1, 0.1732867951399863273543080]]
    # список интегралов где содержится упорядоченная информация:
    # границы интегрирования, табличное значение интеграла
    # затем в список добавляются значения интеграла, расчитанные различными методами
    for integral in range(0, len(integration_info)):
        trapezoid_method_summ = 0 # Интеграл, посчитанный методом трапеции
        simpsons_formula_summ = 0 # Интеграл, посчитанный формулой Симпсона
        formula_3_8_summ = 0 # Интеграл, посчитанный методом 3/8
        trapezoid_method = [] # массив значений, полученный методом трапеции
        simpsons_formula = [] # массив значений, полученный формулой Симпсона
        formula_3_8 = [] # массив значений, полученный методом 3/8
        X = np.array([i / step for i in range((integration_info[integral][0] + 1), integration_info[integral][1] * step)])
        Y = np.array([Function_integral(i, integral)/step for i in X])
        for limit in range(1,X.size):
            trapezoid_method.append(((X[limit] - X[limit - 1])/2)*(Function_integral(X[limit], integral) + Function_integral(X[limit - 1], integral)))
            trapezoid_method_summ += trapezoid_method[-1]
            formula_3_8.append(((X[limit] - X[limit - 1])/8)*((Function_integral(X[limit], integral) + 3*Function_integral((2*X[limit] + X[limit - 1])/3, integral) +
                                                         + 3*Function_integral((X[limit] + 2*X[limit - 1])/3, integral) + Function_integral(X[limit - 1], integral))))
            formula_3_8_summ += formula_3_8[-1]
            simpsons_formula.append(((X[limit] - X[limit - 1])/6)*(Function_integral(X[limit],integral) + 4*Function_integral((X[limit] - X[limit - 1])/2 + X[limit - 1], integral)
                                                                   + Function_integral(X[limit - 1], integral)))
            simpsons_formula_summ += simpsons_formula[-1]
        integration_info[integral].append(trapezoid_method_summ)
        integration_info[integral].append(simpsons_formula_summ)
        integration_info[integral].append(formula_3_8_summ)
        plt.scatter(X[1:], trapezoid_method, label='Метод трапеции', c = "r", lw = 16)
        plt.scatter(X[1:], formula_3_8, label='Метод 3/8', c="y", lw = 11)
        plt.scatter(X[1:], simpsons_formula, label='Формула Симпсона', c="b", lw = 4)
        plt.scatter(X, Y, label='Интеграл', c="m", lw = 0.001)
        plt.legend()
        plt.show()
    print (integration_info)

