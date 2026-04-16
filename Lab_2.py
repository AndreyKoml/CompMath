import numpy as np

a, b, c, d = 0, 0, 0, 0 #Заменить значения

A = np.array([
    [1 + a, 14, -15, 23],
    [16, 1 + b, -22, 29],
    [18, 20, -(1 + c), 32],
    [10, 12, -16, 1 + d]
], dtype=float)

b_vec = np.array([5, 8, 9, 4], dtype=float)

def solve_gauss(A, b):
    n = len(b)

    Ab = np.column_stack((A, b))

    for i in range(n):
        max_row = np.argmax(abs(Ab[i:n, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]

        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]
    return x

print("Решение методом Гаусса:", solve_gauss(A, b_vec))

def solve_lu(A, b):
    n = len(b)
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    return x

print("Решение LU-методом:", solve_lu(A, b_vec))

def solve_jacobi(A, b, eps=1e-5, max_iterations=100):
    n = len(b)
    x = np.zeros(n)

    for k in range(max_iterations):
        x_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - s) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            return x_new, k + 1
        x = x_new
    return x, max_iterations

result_jacobi, iters = solve_jacobi(A, b_vec)
print(f"Решение методом Якоби (итераций {iters}):", result_jacobi)