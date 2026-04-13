import math


def f(x):
    """Исходное уравнение f(x) = 0"""
    return 2 * x ** 2 - 0.5 ** x - 3


def df(x):
    """Первая производная f'(x)"""
    return 4 * x - (0.5 ** x) * math.log(0.5)


def d2f(x):
    """Вторая производная f''(x)"""
    return 4 - (0.5 ** x) * (math.log(0.5)) ** 2


def phi(x):
    """Уравнение вида x = phi(x) для метода итераций"""
    return math.sqrt((0.5 ** x + 3) / 2)


def isolate_roots(a, b, step):
    """Ищет отрезки [x1, x2], на которых функция меняет знак"""
    roots = []
    x1 = a
    while x1 < b:
        x2 = x1 + step
        if f(x1) * f(x2) < 0:
            roots.append((x1, x2))
        x1 = x2
    return roots


def newton_method(x0, eps, max_iter=100):
    """Метод Ньютона (касательных)"""
    x = x0
    for i in range(max_iter):
        x_next = x - f(x) / df(x)
        if abs(x_next - x) < eps:
            return x_next, i + 1
        x = x_next
    return None, max_iter


def iteration_method(x0, eps, max_iter=1000):
    """Метод простой итерации"""
    x = x0
    for i in range(max_iter):
        x_next = phi(x)
        if abs(x_next - x) < eps:
            return x_next, i + 1
        x = x_next
    return None, max_iter


def combined_method(a, b, eps, max_iter=100):
    """Метод хорд и касательных (комбинированный)"""
    for i in range(max_iter):

        if f(a) * d2f(a) < 0:
            a_next = a - f(a) * (b - a) / (f(b) - f(a))
        else:
            a_next = a - f(a) / df(a)

        if f(b) * d2f(b) < 0:
            b_next = b - f(b) * (a - b) / (f(a) - f(b))
        else:
            b_next = b - f(b) / df(b)

        a, b = a_next, b_next
        if abs(b - a) < eps:
            return (a + b) / 2, i + 1

    return None, max_iter


if __name__ == "__main__":
    intervals = isolate_roots(0, 5, 0.5)
    print(f"Найденные интервалы с корнями: {intervals}")

    if not intervals:
        print("Корни не найдены.")
    else:
        a, b = intervals[0]
        print(f"Выбран интервал для поиска: [{a}, {b}]\n")

        eps_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

        print(
            f"| {'Точность (eps)':<14} | {'Ньютона (итер.)':<20} | {'Итераций (итер.)':<20} | {'Комбинированный (итер.)':<25} |")
        print("-" * 88)

        for eps in eps_values:
            x0_newton = b if f(b) * d2f(b) > 0 else a
            root_n, iter_n = newton_method(x0_newton, eps)

            root_i, iter_i = iteration_method((a + b) / 2, eps)

            root_c, iter_c = combined_method(a, b, eps)

            str_eps = str(eps)
            print(
                f"| {str_eps:<14} | {root_n:.6f} ({iter_n:<5})       | {root_i:.6f} ({iter_i:<6})       | {root_c:.6f} ({iter_c:<7})             |")