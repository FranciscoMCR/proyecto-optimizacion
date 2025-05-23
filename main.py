import os
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.gradients import symbolic_function, symbolic_gradient
from core.line_search import armijo_backtracking, wolfe_line_search
from core.logger import OptimizerLogger
from core.optimizers import adam, bfgs, gradient_descent
from core.plotting import contour_plot, show_3d_plot
from core.stochastic import stochastic_gradient_descent


class OptimizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Optimization Playground")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width-100}x{screen_height-100}")
        self.state("zoomed")

        if hasattr(sys, "_MEIPASS"):
            theme_path = os.path.join(sys._MEIPASS, "azure", "azure.tcl")
        else:
            theme_path = os.path.join("azure", "azure.tcl")
        self.tk.call("source", theme_path)
        self.tk.call("set_theme", "light")

        # Canvas + Scrollbar
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.canvas_container = tk.Canvas(self.container)
        self.scrollbar = ttk.Scrollbar(
            self.container, orient="vertical", command=self.canvas_container.yview
        )
        self.scrollbar_x = ttk.Scrollbar(
            self.container, orient="horizontal", command=self.canvas_container.xview
        )
        self.canvas_container.bind_all("<MouseWheel>", self.on_mousewheel)
        self.scrollbar.pack(side="right", fill="y")
        self.scrollbar_x.pack(side="bottom", fill="x")
        self.canvas_container.pack(
            side="left", fill="both", expand=True, padx=10, pady=5
        )
        self.canvas_container.configure(yscrollcommand=self.scrollbar.set)
        self.canvas_container.configure(xscrollcommand=self.scrollbar_x.set)

        # Frame dentro del canvas
        style = ttk.Style().configure(
            "General.TFrame",
            background="#F3F3F3",
        )
        self.content_frame = ttk.Frame(
            self.canvas_container, style="General.TFrame", padding=10
        )
        self.canvas_container.create_window(
            (0, 0), window=self.content_frame, anchor="nw"
        )
        self.content_frame.bind(
            "<Configure>",
            lambda e: self.canvas_container.configure(
                scrollregion=self.canvas_container.bbox("all")
            ),
        )

        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()

        style.configure(
            "Borde.TFrame",
            background="#FFFFFF",  # Fondo dorado
            borderwidth=2,  # Grosor del borde
            relief="solid",
        )

        """ style.configure(
            "Run.TButton",
            background="#007FFF",
            foreground="white",
            ) """

        self.primary_frame = ttk.Frame(
            self.content_frame, style="Borde.TFrame", padding=10
        )

        self.primary_frame.grid(column=0, row=0, sticky="n")
        ttk.Label(
            self.primary_frame, text="Parameters:", font=("Arial", 12, "bold")
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(self.primary_frame, text="Function (f):").grid(
            row=1, column=0, sticky="w"
        )
        self.func_entry = ttk.Entry(self.primary_frame, width=50)
        self.func_entry.insert(0, "x**2 + y**2")
        self.func_entry.grid(row=1, column=1, columnspan=3, pady=5)

        ttk.Label(self.primary_frame, text="Variables:").grid(
            row=2, column=0, sticky="w"
        )
        self.vars_entry = ttk.Entry(self.primary_frame)
        self.vars_entry.insert(0, "x,y")
        self.vars_entry.grid(row=2, column=1, pady=5)

        ttk.Label(self.primary_frame, text="Initial Point:").grid(
            row=3, column=0, sticky="w", pady=5
        )
        self.x0_entry = ttk.Entry(self.primary_frame)
        self.x0_entry.insert(0, "3,4")
        self.x0_entry.grid(row=3, column=1, pady=5)

        ttk.Label(self.primary_frame, text="Tolerance:").grid(
            row=4, column=0, sticky="w"
        )
        self.tol_entry = ttk.Entry(self.primary_frame)
        self.tol_entry.insert(0, "1e-6")
        self.tol_entry.grid(row=4, column=1, pady=5)

        ttk.Label(self.primary_frame, text="Method:").grid(row=5, column=0, sticky="w")
        self.method_combo = ttk.Combobox(
            self.primary_frame, values=["Gradient Descent", "BFGS", "Adam", "SGD"]
        )
        self.method_combo.set("Gradient Descent")
        self.method_combo.grid(row=5, column=1, pady=5)

        ttk.Label(self.primary_frame, text="Max Iterations:").grid(
            row=8, column=0, sticky="w"
        )
        self.max_iter_entry = ttk.Entry(self.primary_frame)
        self.max_iter_entry.insert(0, "100")
        self.max_iter_entry.grid(row=8, column=1, pady=5)

        ttk.Label(self.primary_frame, text="Learning Rate:").grid(
            row=6, column=0, sticky="w"
        )
        self.lr_entry = ttk.Entry(self.primary_frame)
        self.lr_entry.insert(0, "0.01")
        self.lr_entry.grid(row=6, column=1, pady=5)

        ttk.Label(self.primary_frame, text="Line Search:").grid(
            row=7, column=0, sticky="w"
        )
        self.search_combo = ttk.Combobox(
            self.primary_frame, values=["None", "Armijo", "Wolfe"]
        )
        self.search_combo.set("None")
        self.search_combo.grid(row=7, column=1, pady=5)

        self.run_button = ttk.Button(
            self.primary_frame,
            text="Run",
            command=self.run_optimization,
        )

        self.run_button.grid(row=9, column=0, columnspan=2, pady=10)
        self.plot3d_button = ttk.Button(
            self.primary_frame, text="Show 3D Plot", command=self.on_show_3d_plot
        )
        self.plot3d_button.grid(row=9, column=2, columnspan=2, pady=10)

        self.plot3d_contour_button = ttk.Button(
            self.primary_frame,
            text="Show Contour \n     3D Plot",
            command=self.on_show_contour_plot,
        )
        self.plot3d_contour_button.grid(row=10, column=0, columnspan=2, pady=10)

        self.plot3d_button_wpoints = ttk.Button(
            self.primary_frame,
            text="Show 3D Plot \n  With Points",
            command=self.on_show_3d_plot_points,
        )
        self.plot3d_button_wpoints.grid(row=10, column=2, columnspan=2, pady=10)

        self.secondary_frame = ttk.Frame(
            self.content_frame, style="Borde.TFrame", padding=10
        )
        self.secondary_frame.grid(column=3, row=0, padx=53)

        ttk.Label(
            self.secondary_frame, text="Iterations:", font=("Arial", 12, "bold")
        ).grid(row=1, column=0, sticky="w", padx=5)
        columns = ("iter", "f_x", "norm_grad", "alpha")
        self.tree = ttk.Treeview(
            self.secondary_frame, columns=columns, show="headings", height=19
        )
        for col in columns:
            self.tree.heading(col, text=col)
        self.tree.grid(row=2, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")

        self.grid_columnconfigure(1, weight=1)

        self.terciary_frame = ttk.Frame(
            self.content_frame, style="Borde.TFrame", padding=10
        )
        self.terciary_frame.grid(column=0, columnspan=4, row=2, pady=10)

        self.figure = plt.Figure(figsize=(6, 2.5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.terciary_frame)
        self.canvas.get_tk_widget().grid(
            row=0, column=0, columnspan=4, padx=10, pady=10, sticky="nsew"
        )

        self.figure2 = plt.Figure(figsize=(6, 2.5), dpi=100)
        self.ax2 = self.figure2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.figure2, self.terciary_frame)
        self.canvas2.get_tk_widget().grid(
            row=0, column=4, columnspan=4, padx=10, pady=10, sticky="nsew"
        )

        self.stats_label = ttk.Label(self.terciary_frame, text="")
        self.stats_label.grid(
            row=1,
            column=2,
            columnspan=4,
            pady=10,
        )

    def on_mousewheel(self, event):
        widget = event.widget

        if hasattr(event, "delta"):
            delta = -1 * (event.delta // 120)
        elif event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            return

        if widget.winfo_class() == "Treeview":
            widget.yview_scroll(delta, "units")
            return "break"
        else:
            self.canvas_container.yview_scroll(delta, "units")

    def run_optimization(self):
        try:
            func_str = self.func_entry.get()
            variables = [v.strip() for v in self.vars_entry.get().split(",")]
            f = symbolic_function(func_str, variables)
            grad_f = symbolic_gradient(func_str, variables)
            x0 = np.array([float(val) for val in self.x0_entry.get().split(",")])
            tol = float(self.tol_entry.get())
            learning_rate = float(self.lr_entry.get())
            max_iter = int(self.max_iter_entry.get())

            logger = OptimizerLogger()
            logger.reset()  # por si se ejecuta múltiples veces

            eval_count = {"f": 0, "grad": 0}

            def wrapped_f(*args):
                eval_count["f"] += 1
                return f(*args)

            def wrapped_grad_f(*args):
                eval_count["grad"] += 1
                return grad_f(*args)

            ls_option = self.search_combo.get()
            if ls_option == "Armijo":
                line_search = armijo_backtracking
            elif ls_option == "Wolfe":
                line_search = wolfe_line_search
            else:
                line_search = None

            method = self.method_combo.get()
            start_time = time.perf_counter()

            if method == "BFGS":
                x_opt, _ = bfgs(
                    wrapped_f,
                    wrapped_grad_f,
                    x0,
                    tol=tol,
                    max_iter=max_iter,
                    line_search=line_search,
                    callback=logger,
                )
            elif method == "Adam":
                x_opt, _ = adam(
                    wrapped_f,
                    wrapped_grad_f,
                    x0,
                    tol=tol,
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    callback=logger,
                )
            elif method == "SGD":
                x_opt, _ = stochastic_gradient_descent(
                    wrapped_f,
                    wrapped_grad_f,
                    x0,
                    tol=tol,
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    callback=logger,
                )
            else:
                x_opt, _ = gradient_descent(
                    wrapped_f,
                    wrapped_grad_f,
                    x0,
                    tol=tol,
                    max_iter=max_iter,
                    line_search=line_search,
                    callback=logger,
                )

            end_time = time.perf_counter()
            elapsed = end_time - start_time

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

            self.ax.clear()
            fx_vals = [log["f_x"] for log in logger.get_log()]
            iterations = [log["iter"] for log in logger.get_log()]
            self.ax.plot(iterations, fx_vals, marker="o", linestyle="-")
            self.ax.set_title("Convergence of f(x)")
            self.ax.set_xlabel("Iteration")
            self.ax.set_ylabel("f(x)")
            self.ax.grid(True)
            self.canvas.draw()

            self.ax2.clear()
            grad_vals = [log["norm_grad"] for log in logger.get_log()]
            self.ax2.plot(
                iterations, grad_vals, marker="o", linestyle="-", color="orange"
            )
            self.ax2.set_title("Convergence of ‖∇f(x)‖")
            self.ax2.set_xlabel("Iteration")
            self.ax2.set_ylabel("‖∇f‖")
            self.ax2.grid(True)
            self.canvas2.draw()

            x = [log["f_x"] for log in logger.get_log()]
            y = [log["norm_grad"] for log in logger.get_log()]

            self.iteration_points = [x, y]
            self.point = np.round(x_opt, 6)
            solution_text = f"Punto óptimo encontrado: {np.round(x_opt, 6)}"
            self.stats_label.config(
                text=solution_text
                + f"\n⏱️ Time: {elapsed:.4f}s | f(x) calls: {eval_count['f']} | ∇f(x) calls: {eval_count['grad']}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"{type(e).__name__}: {str(e)}")

    def on_show_3d_plot(self):
        func_str = self.func_entry.get()
        variables = [v.strip() for v in self.vars_entry.get().split(",")]

        show_3d_plot(self, func_str, variables)

    def on_show_3d_plot_points(self):
        func_str = self.func_entry.get()
        variables = [v.strip() for v in self.vars_entry.get().split(",")]

        if hasattr(self, "iteration_points"):
            show_3d_plot(self, func_str, variables, self.iteration_points)

    def on_show_contour_plot(self):
        func_str = self.func_entry.get()
        variables = [v.strip() for v in self.vars_entry.get().split(",")]

        if hasattr(self, "point"):
            contour_plot(self, func_str, variables, self.point)
        else:
            contour_plot(self, func_str, variables)


if __name__ == "__main__":
    app = OptimizerApp()
    app.mainloop()
