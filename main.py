import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.gradients import symbolic_function, symbolic_gradient
from core.line_search import armijo_backtracking, wolfe_line_search
from core.logger import OptimizerLogger
from core.optimizers import adam, bfgs, gradient_descent


class OptimizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Optimization Playground")
        self.geometry("800x600")

        # Contenedor base con scrollbar
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.scroll_canvas = tk.Canvas(self.container)
        self.scrollbar = ttk.Scrollbar(
            self.container, orient="vertical", command=self.scroll_canvas.yview
        )
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)

        # Frame interno donde se colocarán todos los widgets
        self.content_frame = ttk.Frame(self.scroll_canvas)
        self.scroll_canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        # Activar scroll dinámico
        self.content_frame.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.configure(
                scrollregion=self.scroll_canvas.bbox("all")
            ),
        )

        self.create_widgets()

    def create_widgets(self):
        # Entrada de función
        ttk.Label(self.content_frame, text="Function (f):").grid(
            row=0, column=0, sticky="w"
        )
        self.func_entry = ttk.Entry(self.content_frame, width=50)
        self.func_entry.insert(0, "x**2 + y**2")
        self.func_entry.grid(row=0, column=1, columnspan=3, pady=5)

        # Variables
        ttk.Label(self.content_frame, text="Variables:").grid(
            row=1, column=0, sticky="w"
        )
        self.vars_entry = ttk.Entry(self.content_frame)
        self.vars_entry.insert(0, "x,y")
        self.vars_entry.grid(row=1, column=1, sticky="ew", pady=5)

        # Punto inicial
        ttk.Label(self.content_frame, text="Initial Point:").grid(
            row=2, column=0, sticky="w"
        )
        self.x0_entry = ttk.Entry(self.content_frame)
        self.x0_entry.insert(0, "3,4")
        self.x0_entry.grid(row=2, column=1, pady=5)

        # Tolerancia
        ttk.Label(self.content_frame, text="Tolerance:").grid(
            row=3, column=0, sticky="w"
        )
        self.tol_entry = ttk.Entry(self.content_frame)
        self.tol_entry.insert(0, "1e-6")
        self.tol_entry.grid(row=3, column=1, pady=5)

        # Método
        ttk.Label(self.content_frame, text="Method:").grid(row=4, column=0, sticky="w")
        self.method_combo = ttk.Combobox(
            self.content_frame, values=["Gradient Descent", "BFGS", "Adam"]
        )
        self.method_combo.set("Gradient Descent")
        self.method_combo.grid(row=4, column=1, pady=5)

        # Estrategia de búsqueda
        ttk.Label(self.content_frame, text="Line Search:").grid(
            row=5, column=0, sticky="w"
        )
        self.search_combo = ttk.Combobox(
            self.content_frame, values=["None", "Armijo", "Wolfe"]
        )
        self.search_combo.set("None")
        self.search_combo.grid(row=5, column=1, pady=5)

        # Botón de ejecución
        self.run_button = ttk.Button(
            self.content_frame, text="Run", command=self.run_optimization
        )
        self.run_button.grid(row=6, column=0, columnspan=2, pady=10)

        # Tabla de resultados
        columns = ("iter", "f_x", "norm_grad", "alpha")
        self.tree = ttk.Treeview(
            self.content_frame, columns=columns, show="headings", height=20
        )
        for col in columns:
            self.tree.heading(col, text=col)
        self.tree.grid(row=7, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")

        self.content_frame.grid_columnconfigure(1, weight=1)

        # Gráfica f(x)
        self.figure = plt.Figure(figsize=(6, 2.5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.graph_canvas = FigureCanvasTkAgg(self.figure, self.content_frame)
        self.graph_canvas.get_tk_widget().grid(
            row=8, column=0, columnspan=4, padx=10, pady=10, sticky="nsew"
        )

        # Gráfica ||∇f||
        self.figure2 = plt.Figure(figsize=(6, 2.5), dpi=100)
        self.ax2 = self.figure2.add_subplot(111)
        self.graph_canvas2 = FigureCanvasTkAgg(self.figure2, self.content_frame)
        self.graph_canvas2.get_tk_widget().grid(
            row=9, column=0, columnspan=4, padx=10, pady=10, sticky="nsew"
        )

    def run_optimization(self):
        try:
            func_str = self.func_entry.get()
            variables = [v.strip() for v in self.vars_entry.get().split(",")]
            f = symbolic_function(func_str, variables)
            grad_f = symbolic_gradient(func_str, variables)
            x0 = np.array([float(val) for val in self.x0_entry.get().split(",")])
            tol = float(self.tol_entry.get())

            logger = OptimizerLogger()

            # Line search
            ls_option = self.search_combo.get()
            if ls_option == "Armijo":
                line_search = armijo_backtracking
            elif ls_option == "Wolfe":
                line_search = wolfe_line_search
            else:
                line_search = None

            # Método
            method = self.method_combo.get()
            if method == "BFGS":
                x_opt, _ = bfgs(
                    f, grad_f, x0, tol=tol, line_search=line_search, callback=logger
                )
            elif method == "Adam":
                x_opt, _ = adam(
                    f, grad_f, x0, tol=tol, max_iter=1000, alpha=0.05, callback=logger
                )
            else:
                x_opt, _ = gradient_descent(
                    f, grad_f, x0, tol=tol, line_search=line_search, callback=logger
                )

            # Tabla de resultados
            self.tree.delete(*self.tree.get_children())
            for log in logger.get_log():
                self.tree.insert(
                    "",
                    "end",
                    values=(
                        log["iter"],
                        f"{log['f_x']:.6f}",
                        f"{log['norm_grad']:.6f}",
                        f"{log['alpha']:.6f}" if log["alpha"] else "-",
                    ),
                )

            # Gráfico f(x)
            self.ax.clear()
            fx_vals = [log["f_x"] for log in logger.get_log()]
            iterations = [log["iter"] for log in logger.get_log()]
            self.ax.plot(iterations, fx_vals, marker="o", linestyle="-")
            self.ax.set_title("Convergence of f(x)")
            self.ax.set_xlabel("Iteration")
            self.ax.set_ylabel("f(x)")
            self.ax.grid(True)
            self.graph_canvas.draw()

            # Gráfico ||∇f||
            self.ax2.clear()
            grad_vals = [log["norm_grad"] for log in logger.get_log()]
            self.ax2.plot(
                iterations, grad_vals, marker="o", linestyle="-", color="orange"
            )
            self.ax2.set_title("Convergence of ‖∇f(x)‖")
            self.ax2.set_xlabel("Iteration")
            self.ax2.set_ylabel("‖∇f‖")
            self.ax2.grid(True)
            self.graph_canvas2.draw()

        except Exception as e:
            messagebox.showerror("Error", f"{type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    app = OptimizerApp()
    app.mainloop()
