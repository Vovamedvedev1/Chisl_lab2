import numpy as np
from math import sqrt
from lagrange_polinom import function, get_max_derivative

def get_gauss_polinom(x_0, function_str, a, b, eps):
    hh = min(abs(x_0 - a), abs(b - x_0))
    h = 2 * hh
    R = float('inf')   
    max_iter = 10000   
    count_iter = 0 
    while abs(R) > eps and count_iter <= max_iter: 
        x__0, x__1, x__2 = x_0 - h / 2, x_0 + h / 2, x_0 + 3 * h / 2 
        y_i = function(x__1, math_value=function_str)
        y_i_1 = function(x__2, math_value=function_str) - function(x__1, math_value=function_str)
        y_i_2 = function(x__0, math_value=function_str) - 2 * function(x__1, math_value=function_str) + function(x__2, math_value=function_str)
        t = (x_0 - x__1) / h
        L = y_i + y_i_1 * t + y_i_2 * t * (t - 1) / 2
        R = abs(get_max_derivative(function_str, 3, x_0, x__2) *(h ** 3) * t * (t * t - 1) / 6)
        print(f"Итерация: {count_iter}; h = {h}; L = {L}; R = {R}")
        count_iter += 1
        h = h / 2
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
    pract = get_gauss_polinom(x_0, func_string, a, b,eps=10**(-10))
    print("Теоретическое значение: ", teor)
    print("Вычиссленное значение: ", pract)
    print("Абсолютная погрешность: ", abs(teor - pract))
    print("Относительная погрешность: ", abs(teor - pract) / abs(pract))
except Exception as e:
    print(str(e))