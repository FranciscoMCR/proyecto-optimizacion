import numpy as np


def quadratic(x: np.ndarray, A=None, b=None, c=0):
    A = A if A is not None else np.eye(len(x))
    b = b if b is not None else np.zeros(len(x))
    return float(0.5 * x.T @ A @ x + b.T @ x + c)


def rosenbrock(x: np.ndarray, a=1, b=100):
    return sum(b * (x[1:] - x[:-1] ** 2) ** 2 + (a - x[:-1]) ** 2)


def rastrigin(x: np.ndarray, A=10):
    n = len(x)
    return A * n + sum(x**2 - A * np.cos(2 * np.pi * x))


def himmelblau(x: np.ndarray):
    """
    Himmelblau's function:
    f(x, y) = (x² + y - 11)² + (x + y² - 7)²
    Tiene múltiples mínimos locales.
    """
    if len(x) != 2:
        raise ValueError("Himmelblau function is only defined for 2D inputs.")
    x1, x2 = x
    return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2
