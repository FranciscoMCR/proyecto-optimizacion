class OptimizerLogger:
    def __init__(self):
        self.logs = []

    def __call__(self, iteration, x, f_x, grad_x, norm_grad, alpha=None):
        """
        Guarda los datos de una iteración de optimización.

        Params:
        - iteration: número de iteración
        - x: punto actual (np.ndarray)
        - f_x: valor de la función en x
        - grad_x: gradiente en x
        - norm_grad: norma del gradiente
        - alpha: tamaño de paso (si aplica)
        """
        self.logs.append(
            {
                "iter": iteration,
                "x": x.copy(),
                "f_x": f_x,
                "norm_grad": norm_grad,
                "alpha": alpha,
            }
        )

    def get_log(self):
        return self.logs

    def print_summary(self):
        print(f"\n{'Iter':>4} | {'f(x)':>12} | {'‖∇f‖':>12} | {'Step α':>8}")
        print("-" * 46)
        for log in self.logs:
            iter = log["iter"]
            f_x = log["f_x"]
            norm = log["norm_grad"]
            alpha = log["alpha"] if log["alpha"] is not None else "-"
            print(f"{iter:>4} | {f_x:>12.6f} | {norm:>12.6f} | {alpha:>8}")
