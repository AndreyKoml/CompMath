import numpy as np
import sympy as sp
from scipy.optimize import fsolve
import time

x1_s, x2_s = sp.symbols('x1 x2')

f1 = sp.cos(x1_s + 0.5) + x2_s - 0.8
f2 = sp.sin(x2_s) - 2*x1_s - 1.6

phi1_s = (sp.sin(x2_s) - 1.6) / 2
phi2_s = 0.8 - sp.cos(x1_s + 0.5)

J_s = sp.Matrix([[sp.diff(f1, x1_s), sp.diff(f1, x2_s)],
                 [sp.diff(f2, x1_s), sp.diff(f2, x2_s)]])

f_func = sp.lambdify((x1_s, x2_s), [f1, f2], 'numpy')
phi_func = sp.lambdify((x1_s, x2_s), [phi1_s, phi2_s], 'numpy')
jac_func = sp.lambdify((x1_s, x2_s), J_s, 'numpy')

def F(x): return np.array(f_func(x[0], x[1]), dtype=float)
def PHI(x): return np.array(phi_func(x[0], x[1]), dtype=float)
def JACOBIAN(x): return np.array(jac_func(x[0], x[1]), dtype=float)

def method_newton(x0, eps):
    start = time.perf_counter()
    x = np.array(x0, dtype=float)
    for i in range(1, 101):
        J = JACOBIAN(x)
        f_val = F(x)
        dx = np.linalg.solve(J, -f_val)
        x = x + dx
        if np.linalg.norm(dx, ord=np.inf) < eps:
            return x, i, time.perf_counter() - start
    return x, 100, time.perf_counter() - start
def method_iterations(x0, eps):
    start = time.perf_counter()
    x = np.array(x0, dtype=float)
    for i in range(1, 1000):
        x_new = PHI(x)
        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            return x_new, i, time.perf_counter() - start
        x = x_new
    return x, 1000, time.perf_counter() - start
def method_scipy(x0):
    start = time.perf_counter()
    res = fsolve(F, x0)
    return res, time.perf_counter() - start

if __name__ == "__main__":
    x_start = [-0.7, -0.1]
    epsilon = 1e-5

    print(f"Система варианта 9. Старт из точки: {x_start}\n")
    print(f"{'Метод':<20} | {'Результат (x1, x2)':<25} | {'Итер.':<5} | {'Время (сек)':<10}")
    print("-" * 75)

    r_n, i_n, t_n = method_newton(x_start, epsilon)
    print(f"{'Метод Ньютона':<20} | {str(np.round(r_n, 6)):<25} | {i_n:<5} | {t_n:.8f}")

    r_i, i_i, t_i = method_iterations(x_start, epsilon)
    print(f"{'Метод итераций':<20} | {str(np.round(r_i, 6)):<25} | {i_i:<5} | {t_i:.8f}")

    r_s, t_s = method_scipy(x_start)
    print(f"{'Scipy (fsolve)':<20} | {str(np.round(r_s, 6)):<25} | {'-':<5} | {t_s:.8f}")