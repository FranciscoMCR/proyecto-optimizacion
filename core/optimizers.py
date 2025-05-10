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
    alpha=None,
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
            callback(k, x.copy(), f_x, grad.copy(), norm_grad, alpha)

        if norm_grad < tol:
            break

        d = -grad  # dirección de descenso

        if line_search:
            alpha = line_search(f, grad_f, x, d)
        else:
            alpha = step_size

        x = x + alpha * d

    return x, history


def bfgs(
    f,
    grad_f,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
    line_search=None,
    callback=None,
):
    """
    Método BFGS (quasi-Newton) con opción de búsqueda lineal.

    Parámetros:
    - f: función objetivo
    - grad_f: gradiente
    - x0: punto inicial
    - tol: tolerancia sobre ||∇f||
    - max_iter: máximo de iteraciones
    - line_search: función line_search(f, grad_f, x, d), opcional
    - callback: función de monitoreo por iteración

    Retorna:
    - x_opt: punto óptimo
    - history: lista con registros por iteración
    """
    x = x0.copy()
    n = len(x)
    H = np.eye(n)  # Aproximación inicial de la Hessiana inversa
    history = []
    alpha = None

    for k in range(1, max_iter + 1):
        grad = np.array(grad_f(*x), dtype=float)
        norm_grad = np.linalg.norm(grad)
        f_x = f(*x)

        history.append((k, x.copy(), f_x, norm_grad))
        if callback:
            callback(k, x.copy(), f_x, grad.copy(), norm_grad, alpha)

        if norm_grad < tol:
            break

        # Dirección de descenso: -H ∇f
        d = -H @ grad

        # Paso
        alpha = line_search(f, grad_f, x, d) if line_search else 1.0
        x_new = x + alpha * d

        # Diferencias
        s = x_new - x
        y = np.array(grad_f(*x_new), dtype=float) - grad

        # Actualización de H con fórmula BFGS
        rho = 1.0 / (y @ s)
        I = np.eye(n)
        H = (I - rho * np.outer(s, y)) @ H @ (
            I - rho * np.outer(y, s)
        ) + rho * np.outer(s, s)

        x = x_new

    return x, history


def adam(
    f,
    grad_f,
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
    alpha: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    callback=None,
):
    """
    Adam optimizer para funciones multivariables.

    Parámetros:
    - f: función objetivo f(x)
    - grad_f: función gradiente ∇f(x)
    - x0: punto inicial
    - tol: tolerancia para ||∇f(x)||
    - max_iter: número máximo de iteraciones
    - alpha: tasa de aprendizaje
    - beta1: decaimiento de primer momento
    - beta2: decaimiento de segundo momento
    - epsilon: valor pequeño para estabilidad numérica
    - callback: función para logging de iteraciones

    Retorna:
    - x_opt: punto encontrado
    - history: lista de tuplas (iter, x, f(x), ||grad||, alpha)
    """
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = []

    for k in range(1, max_iter + 1):
        grad = np.array(grad_f(*x), dtype=float)
        f_x = f(*x)
        norm_grad = np.linalg.norm(grad)

        history.append((k, x.copy(), f_x, norm_grad, alpha))

        if callback:
            callback(k, x.copy(), f_x, grad.copy(), norm_grad, alpha)

        if norm_grad < tol:
            break

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)

        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)

        x = x - alpha * m_hat / (np.sqrt(v_hat) + epsilon)

    return x, history
