import numpy as np


def gradient_descent(
    f,
    grad_f,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
    step_size: float = 0.01,
    line_search=None,
    callback=None,
):
    """
    Método de descenso por gradiente con paso fijo o búsqueda lineal.

    Parámetros:
    - f: función objetivo f(x)
    - grad_f: función gradiente ∇f(x)
    - x0: punto inicial (np.ndarray)
    - tol: tolerancia para ||∇f(x)||
    - max_iter: número máximo de iteraciones
    - step_size: paso inicial (usado si no hay búsqueda lineal)
    - line_search: función line_search(f, grad_f, x, d), opcional
    - callback: función que recibe info por iteración: callback(k, x, f_x, grad_x)

    Retorna:
    - x_opt: punto final
    - history: lista con los registros por iteración [(k, x, f(x), ||grad||)]
    """
    x = x0.copy()
    history = []

    for k in range(1, max_iter + 1):
        grad = grad_f(*x)
        grad = np.array(grad, dtype=float)  # por si viene como lista sympy
        norm_grad = np.linalg.norm(grad)

        f_x = f(*x)
        history.append((k, x.copy(), f_x, norm_grad))

        if callback:
            callback(k, x.copy(), f_x, grad.copy(), norm_grad)

        if norm_grad < tol:
            break

        d = -grad  # dirección de descenso

        if line_search:
            alpha = line_search(f, grad_f, x, d)
        else:
            alpha = step_size

        x = x + alpha * d

    return x, history
