class OptimizerLogger:
    def __init__(self):
        self._logs = []

    def __call__(self, iteration, x, f_x, grad_x, norm_grad, alpha=None):
        """
        Guarda los datos de una iteración de optimización.

        Parameters:
            iteration (int): Número de iteración
            x (np.ndarray): Punto actual
            f_x (float): Valor de la función objetivo
            grad_x (np.ndarray): Gradiente en el punto
            norm_grad (float): Norma del gradiente
            alpha (float | None): Tamaño de paso (si aplica)
        """
        self._logs.append(
            {
                "iter": iteration,
                "x": x.copy(),
                "f_x": f_x,
                "grad": grad_x.copy(),
                "norm_grad": norm_grad,
                "alpha": alpha,
            }
        )

    def get_log(self):
        return self._logs

    def get_last(self):
        return self._logs[-1] if self._logs else None

    def reset(self):
        self._logs.clear()

    def print_summary(self):
        print(f"\n{'Iter':>4} | {'f(x)':>12} | {'‖∇f‖':>12} | {'Step α':>10}")
        print("-" * 46)
        for log in self._logs:
            alpha = f"{log['alpha']:.6f}" if log["alpha"] is not None else "-"
            print(
                f"{log['iter']:>4} | {log['f_x']:>12.6f} | {log['norm_grad']:>12.6f} | {alpha:>10}"
            )
