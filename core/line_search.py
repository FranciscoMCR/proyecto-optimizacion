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


def wolfe_line_search(
    f, grad_f, x, d, alpha_init=1.0, c1=1e-4, c2=0.9, max_iter=20, rho=0.5
):
    """
    Búsqueda lineal usando condiciones de Wolfe.

    Parámetros:
    - f: función objetivo
    - grad_f: gradiente
    - x: punto actual
    - d: dirección de descenso
    - alpha_init: paso inicial (típicamente 1.0)
    - c1, c2: constantes de Wolfe
    - max_iter: máximo de iteraciones
    - rho: factor de reducción del paso (si no cumple condiciones)

    Retorna:
    - alpha: tamaño de paso aceptado
    """
    alpha = alpha_init
    grad_x = np.array(grad_f(*x), dtype=float)
    phi0 = f(*x)
    phi0_prime = np.dot(grad_x, d)

    for _ in range(max_iter):
        x_new = x + alpha * d
        phi = f(*x_new)
        grad_new = np.array(grad_f(*x_new), dtype=float)
        phi_prime = np.dot(grad_new, d)

        if phi > phi0 + c1 * alpha * phi0_prime:
            alpha *= rho  # falla Armijo
        elif phi_prime < c2 * phi0_prime:
            alpha *= 1 / rho  # aumenta si no cumple curvatura
        else:
            return alpha  # ambas condiciones satisfechas

    return alpha  # devuelve último alpha aunque no cumpla
