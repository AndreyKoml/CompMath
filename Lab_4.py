import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from math import factorial

def f(x):
    return x * (2**x) - 1

def taylor_approx(x, c, n):
    ln2 = np.log(2)
    if n == 1:
        f_c = f(c)
        df_c = 2**c * (1 + c * ln2)
        return f_c + df_c * (x - c)
    elif n == 2:
        f_c = f(c)
        df_c = 2**c * (1 + c * ln2)
        d2f_c = 2**c * ln2 * (2 + c * ln2)
        return f_c + df_c * (x - c) + (d2f_c / 2) * (x - c)**2
    return f(c)

a, b = 0, 1
x_fine = np.linspace(a, b, 200)
y_true = f(x_fine)

points_c = [a, b, (a + b) / 2]
print("Таблица 1: Ошибка аппроксимации Тейлора")
print(f"{'n':<3} | {'c = a':<10} | {'c = b':<10} | {'c = (a+b)/2':<10}")
for n in [1, 2]:
    errors = []
    for c in points_c:
        y_taylor = taylor_approx(x_fine, c, n)
        max_err = np.max(np.abs(y_true - y_taylor))
        errors.append(f"{max_err:.5f}")
    print(f"{n:<3} | {errors[0]:<10} | {errors[1]:<10} | {errors[2]:<10}")

def get_lagrange_error(n_nodes):
    nodes_x = np.linspace(a, b, n_nodes)
    nodes_y = f(nodes_x)
    poly = lagrange(nodes_x, nodes_y)
    y_interp = poly(x_fine)
    return np.abs(y_true - y_interp), np.max(np.abs(y_true - y_interp))

err_5, max_5 = get_lagrange_error(5)
err_10, max_10 = get_lagrange_error(10)

print(f"\nТаблица 2: Максимальная ошибка интерполяции")
print(f"Степень (узлы 5): {max_5:.10f}")
print(f"Степень (узлы 10): {max_10:.10f}")

plt.figure(figsize=(10, 6))
plt.plot(x_fine, err_5, label='Ошибка (5 узлов)')
plt.plot(x_fine, err_10, label='Ошибка (10 узлов)', linestyle='--')
plt.title('Распределение ошибки глобальной интерполяции')
plt.xlabel('x')
plt.ylabel('|f(x) - L(x)|')
plt.legend()
plt.grid(True)
plt.show()