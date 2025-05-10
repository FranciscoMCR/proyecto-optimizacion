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


def ackley(x: np.ndarray, a=20, b=0.2, c=2 * np.pi):
    """
    Ackley function:
    f(x) = -a * exp(-b * sqrt(sum(x_i^2)/n)) - exp(sum(cos(c*x_i))/n) + a + exp(1)
    Tiene mínimo global en x=0 con f(x)=0
    """
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + a + np.exp(1)
