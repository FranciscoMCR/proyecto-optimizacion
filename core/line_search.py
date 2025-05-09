import numpy as np


def armijo_backtracking(f, grad_f, x, d, alpha_init=1.0, rho=0.5, c=1e-4, max_iter=20):
    """
    Búsqueda lineal por retroceso usando la condición de Armijo.

    Parámetros:
    - f: función objetivo
    - grad_f: gradiente de la función
    - x: punto actual (np.ndarray)
    - d: dirección de descenso (np.ndarray)
    - alpha_init: tamaño de paso inicial (típicamente 1.0)
    - rho: factor de reducción (típicamente 0.5)
    - c: constante de Armijo (típicamente 1e-4)
    - max_iter: máximo de reducciones del paso

    Retorna:
    - alpha: tamaño de paso aceptado
    """
    alpha = alpha_init
    f_x = f(*x)
    grad_x = np.array(grad_f(*x), dtype=float)
    directional_derivative = np.dot(grad_x, d)

    for _ in range(max_iter):
        x_new = x + alpha * d
        f_new = f(*x_new)
        if f_new <= f_x + c * alpha * directional_derivative:
            return alpha
        alpha *= rho

    return alpha  # devolver el último valor aunque no cumpla
