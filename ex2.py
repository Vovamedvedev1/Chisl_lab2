import numpy as np
from math import sqrt
from lagrange_polinom import function, get_max_derivative


def get_gauss_polinom(x_0, function_str, a, b, eps):
    h = (b - a)
    R = float('inf')   
    max_iter = 10000   
    y_0 = function(x_0, math_value=function_str)
    count_iter = 0 
    while abs(R) > eps and count_iter <= max_iter:
        h /= 2
        x = np.arange(a, b + h, h)
        n = len(x)
        y = function(x, math_value=function_str)
        imin = 0
        d_min = abs(x[0] - x_0)
        for i in range(1, n):
            d = abs(x[i] - x_0)
            if d < d_min:
                imin, d_min = i, d
        if imin == 0:
            imin = 1
        if imin == n-1:
            imin = n - 2
        y_i = y[imin]
        y_i_1 = y[imin + 1] - y[imin]
        y_i_2 = y[imin - 1] - 2 * y[imin] + y[imin + 1]
        t = (x_0 - x[imin]) / h
        L = y_i + y_i_1 * t + y_i_2 * t * (t - 1) / 2
        R = abs(get_max_derivative(function_str, 3, x[imin-1], x[imin+1]) *(h ** 3) * t * (t * t - 1) / 6)
        print(f"Итерация: {count_iter}; h = {h}; L = {L}; R = {R}")
        count_iter += 1
    return L

try:
    func_string = input("Введите функцию: ")
    a = float(input("Введите a: "))
    b = float(input("Введите b: "))
    if a >= b:
        raise Exception("а должно быть строго меньше b")
    x_0 = float(input("Введите x0: "))
    if x_0 < a or x_0 > b:
        raise Exception("x_0 не должно выходить за пределы указанного диапазона")
    teor = function(x_0, math_value=func_string)
    pract = get_gauss_polinom(x_0, func_string, a, b,eps=10**(-4))
    print("Теоретическое значение: ", teor)
    print("Вычиссленное значение: ", pract)
    print("Абсолютная погрешность: ", abs(teor - pract))
    print("Относительная погрешность: ", abs(teor - pract) / abs(pract))
except Exception as e:
    print(str(e))