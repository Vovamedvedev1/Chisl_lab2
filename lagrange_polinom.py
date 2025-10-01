import functools
import numpy as np
import tkinter as tk
from GUI import GUI
import matplotlib.pyplot as plt
import math
from math import factorial
import sympy as sp
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, function_exponentiation

def function(x, math_value):
    compiled_value = compile(math_value, '<string>', 'eval')
    return eval(compiled_value, {'np': np, 'math': math}, {'x': x})


def get_max_derivative(function_str, n_dir, a, b):
    transformations = standard_transformations + (function_exponentiation,)
    cleaned_string = function_str.replace('np.', '')
    x = sp.Symbol('x')
    expr = parse_expr(cleaned_string, transformations=transformations)
    nth_deriv_exact = expr.diff(x, n_dir) 
    nth_deriv_func = lambdify(x, nth_deriv_exact, 'numpy')
    x_values = np.linspace(a, b, 1000)
    deriv_values = np.abs(nth_deriv_func(x_values))
    return deriv_values.max()

class InterpolationApp(GUI):
    def __init__(self, root):
        self.x_plot, self.y_plot, self.y_lagrange = None, None, None
        self.x, self.y = None, None
        self.a, self.b, self.n = None, None, None
        self.function_string = None
        self.fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, width_ratios=[2, 1])
        self.ax_main, self.ax_theor, self.ax_actual = plt.subplot(gs[:, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 1])  
        self.axs = [self.ax_main, self.ax_theor, self.ax_actual]
        super().__init__(root, self.fig)

    def get_lagrange_polinom(self, t):
        result = np.zeros_like(t)
        for i in range(self.n):
            el = self.y[i]
            for j in range(self.n):
                if i != j:
                    el *= (t - self.x[j]) / (self.x[i] - self.x[j])
            result += el
        return result

    def get_t_error_lagrange(self, t):
        w = np.ones_like(t)
        for i in range(self.n):
            w *= (t - self.x[i])
        fact = factorial(self.n + 1)  
        M = get_max_derivative(self.function_string, self.n + 1, self.a, self.b)
        return np.abs(M * w / fact)

    def calculate(self):
        try:
            self.function_string = str(self.entry_func.get())
            self.a = float(self.entry_a.get())
            self.b = float(self.entry_b.get())
            self.n = int(self.entry_n.get())
            if self.n <= 0:
                raise ValueError("Количество узлов должно быть положительным")
            if self.a >= self.b:
                raise ValueError("Левая граница должна быть меньше правой")
            self.x_plot = np.linspace(self.a, self.b, 100000)
            self.y_plot = function(self.x_plot, math_value=self.function_string)
            self.x = np.linspace(self.a, self.b, self.n)
            self.y = function(self.x, math_value=self.function_string)
            self.y_lagrange = self.get_lagrange_polinom(self.x_plot)
            theoretical_error_lagrange = self.get_t_error_lagrange(self.x_plot)
            actual_error_lagrange = np.abs(self.y_plot - self.y_lagrange)
            for ax in self.axs:
                ax.clear()
            self.ax_main.plot(self.x_plot, self.y_plot, label='Исходная функция', linewidth=2)
            self.ax_main.plot(self.x_plot, self.y_lagrange, 'g--', label='Полином Лагранжа', linewidth=1.5)
            self.ax_main.scatter(self.x, self.y, c='r', s=50, label='Узлы интерполяции', zorder=5)
            self.ax_main.set_xlabel('x')
            self.ax_main.set_ylabel('y')
            self.ax_main.set_title(f'Интерполяция полиномом Лагранжа (n={self.n})')
            self.ax_main.legend()
            self.ax_main.grid(True, alpha=0.3)
            
            self.ax_theor.plot(self.x_plot, theoretical_error_lagrange, 'r-', linewidth=2)
            self.ax_theor.set_xlabel('x')
            self.ax_theor.set_ylabel('Погрешность')
            teor_max = f"{theoretical_error_lagrange.max():.2e}"
            self.ax_theor.set_title(f'Теоретическая погрешность\nmax = {teor_max}')
            self.ax_theor.grid(True, alpha=0.3)
            #self.ax_theor.set_yscale('log')
            
            self.ax_actual.plot(self.x_plot, actual_error_lagrange, 'b-', linewidth=2)
            self.ax_actual.set_xlabel('x')
            self.ax_actual.set_ylabel('Погрешность')
            actual_max = f"{actual_error_lagrange.max():.2e}"
            self.ax_actual.set_title(f'Фактическая погрешность\nmax = {actual_max}')
            self.ax_actual.grid(True, alpha=0.3)
            #self.ax_actual.set_yscale('log')
            
            self.fig.tight_layout()
            self.canvas.draw()
            self.error_label.config(text="")
        except Exception as e:
            self.error_label.config(text=f"Ошибка: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = InterpolationApp(root)
    root.mainloop()