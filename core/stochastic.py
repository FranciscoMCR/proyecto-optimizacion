import numpy as np


def stochastic_gradient_descent(
    f,
    grad_f,
    x0,
    step_size=0.01,
    tol=1e-6,
    max_iter=100,
    noise_scale=1e-3,
    callback=None,
):
    """
    Stochastic Gradient Descent (SGD)

    Parámetros:
    - f: función objetivo
    - grad_f: función gradiente
    - x0: punto inicial (np.ndarray)
    - step_size: tamaño de paso base
    - tol: tolerancia para ||∇f||
    - max_iter: número máximo de iteraciones
    - noise_scale: amplitud del ruido gaussiano aplicado al gradiente
    - callback: función opcional que recibe (k, x, f(x), grad, ||grad||, alpha)

    Retorna:
    - x_opt: punto final
    - history: lista con tuplas (k, x, f(x), ||grad||)
    """
    x = x0.copy()
    history = []

    for k in range(1, max_iter + 1):
        grad = np.array(grad_f(*x), dtype=float)
        noise = np.random.normal(0, noise_scale, size=grad.shape)
        grad_noisy = grad + noise
        norm_grad = np.linalg.norm(grad)

        f_x = f(*x)
        history.append((k, x.copy(), f_x, norm_grad))

        if callback:
            callback(k, x.copy(), f_x, grad_noisy, norm_grad, step_size)

        if norm_grad < tol:
            break

        x = x - step_size * grad_noisy

    return x, history


def simulated_annealing(*args, **kwargs):
    raise NotImplementedError("Simulated Annealing will be implemented soon.")
