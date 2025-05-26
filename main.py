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

        # load azure theme whether in PyInstaller bundle or not
        if hasattr(sys, "_MEIPASS"):
            theme_path = os.path.join(sys._MEIPASS, "azure", "azure.tcl")
        else:
            theme_path = os.path.join("azure", "azure.tcl")
        self.tk.call("source", theme_path)
        self.tk.call("set_theme", "light")

        # Canvas + Scrollbars
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

        # Inner frame
        ttk.Style().configure("General.TFrame", background="#F3F3F3")
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
        s = ttk.Style()
        s.configure("Borde.TFrame", background="#FFFFFF", borderwidth=2, relief="solid")

        # Primary parameters frame
        self.primary_frame = ttk.Frame(
            self.content_frame, style="Borde.TFrame", padding=10
        )
        self.primary_frame.grid(column=0, row=0, sticky="nw")

        ttk.Label(
            self.primary_frame, text="Parameters:", font=("Arial", 12, "bold")
        ).grid(row=0, column=0, sticky="w")
        # Function
        ttk.Label(self.primary_frame, text="Function (f):").grid(
            row=1, column=0, sticky="w"
        )
        self.func_entry = ttk.Entry(self.primary_frame, width=50)
        self.func_entry.insert(0, "x**2 + y**2")
        self.func_entry.grid(row=1, column=1, columnspan=3, pady=5, sticky="w")
        # Variables
        ttk.Label(self.primary_frame, text="Variables:").grid(
            row=2, column=0, sticky="w"
        )
        self.vars_entry = ttk.Entry(self.primary_frame)
        self.vars_entry.insert(0, "x,y")
        self.vars_entry.grid(row=2, column=1, pady=5, sticky="w")
        # Initial Point
        ttk.Label(self.primary_frame, text="Initial Point:").grid(
            row=3, column=0, sticky="w"
        )
        self.x0_entry = ttk.Entry(self.primary_frame)
        self.x0_entry.insert(0, "3,4")
        self.x0_entry.grid(row=3, column=1, pady=5, sticky="w")
        # Tolerance
        ttk.Label(self.primary_frame, text="Tolerance:").grid(
            row=4, column=0, sticky="w"
        )
        self.tol_entry = ttk.Entry(self.primary_frame)
        self.tol_entry.insert(0, "1e-6")
        self.tol_entry.grid(row=4, column=1, pady=5, sticky="w")
        # Method
        ttk.Label(self.primary_frame, text="Method:").grid(row=5, column=0, sticky="w")
        self.method_combo = ttk.Combobox(
            self.primary_frame, values=["Gradient Descent", "BFGS", "Adam", "SGD"]
        )
        self.method_combo.set("Gradient Descent")
        self.method_combo.grid(row=5, column=1, pady=5, sticky="w")
        # Learning Rate
        ttk.Label(self.primary_frame, text="Learning Rate:").grid(
            row=6, column=0, sticky="w"
        )
        self.lr_entry = ttk.Entry(self.primary_frame)
        self.lr_entry.insert(0, "0.01")
        self.lr_entry.grid(row=6, column=1, pady=5, sticky="w")
        # Line Search
        ttk.Label(self.primary_frame, text="Line Search:").grid(
            row=7, column=0, sticky="w"
        )
        self.search_combo = ttk.Combobox(
            self.primary_frame, values=["None", "Armijo", "Wolfe"]
        )
        self.search_combo.set("None")
        self.search_combo.grid(row=7, column=1, pady=5, sticky="w")
        # Max iterations
        ttk.Label(self.primary_frame, text="Max Iterations:").grid(
            row=8, column=0, sticky="w"
        )
        self.max_iter_entry = ttk.Entry(self.primary_frame)
        self.max_iter_entry.insert(0, "100")
        self.max_iter_entry.grid(row=8, column=1, pady=5, sticky="w")

        # Line Search Hyperparameters
        ttk.Label(self.primary_frame, text="LS Max Iter:").grid(
            row=9, column=2, sticky="w"
        )
        self.ls_max_iter_entry = ttk.Entry(self.primary_frame, width=6)
        self.ls_max_iter_entry.insert(0, "20")
        self.ls_max_iter_entry.grid(row=9, column=3, pady=5, sticky="w")

        ttk.Label(self.primary_frame, text="ρ (rho):").grid(
            row=10, column=2, sticky="w"
        )
        self.rho_entry = ttk.Entry(self.primary_frame, width=6)
        self.rho_entry.insert(0, "0.5")
        self.rho_entry.grid(row=10, column=3, pady=5, sticky="w")

        ttk.Label(self.primary_frame, text="c (Armijo):").grid(
            row=11, column=2, sticky="w"
        )
        self.c_entry = ttk.Entry(self.primary_frame, width=6)
        self.c_entry.insert(0, "1e-4")
        self.c_entry.grid(row=11, column=3, pady=5, sticky="w")

        ttk.Label(self.primary_frame, text="c1 (Wolfe):").grid(
            row=12, column=2, sticky="w"
        )
        self.c1_entry = ttk.Entry(self.primary_frame, width=6)
        self.c1_entry.insert(0, "1e-4")
        self.c1_entry.grid(row=12, column=3, pady=5, sticky="w")

        ttk.Label(self.primary_frame, text="c2 (Wolfe):").grid(
            row=13, column=2, sticky="w"
        )
        self.c2_entry = ttk.Entry(self.primary_frame, width=6)
        self.c2_entry.insert(0, "0.9")
        self.c2_entry.grid(row=13, column=3, pady=5, sticky="w")

        # Buttons
        self.run_button = ttk.Button(
            self.primary_frame, text="Run", command=self.run_optimization
        )
        self.run_button.grid(row=14, column=0, columnspan=2, pady=10)
        self.plot3d_button = ttk.Button(
            self.primary_frame, text="Show 3D Plot", command=self.on_show_3d_plot
        )
        self.plot3d_button.grid(row=14, column=2, columnspan=2, pady=10)
        self.plot_contour_button = ttk.Button(
            self.primary_frame,
            text="Show Contour Plot",
            command=self.on_show_contour_plot,
        )
        self.plot_contour_button.grid(row=15, column=0, columnspan=2, pady=5)
        self.plot3d_points_button = ttk.Button(
            self.primary_frame,
            text="Show 3D With Points",
            command=self.on_show_3d_plot_points,
        )
        self.plot3d_points_button.grid(row=15, column=2, columnspan=2, pady=5)

        # Results table
        self.secondary_frame = ttk.Frame(
            self.content_frame, style="Borde.TFrame", padding=10
        )
        self.secondary_frame.grid(column=1, row=0, padx=20, sticky="n")
        ttk.Label(
            self.secondary_frame, text="Iterations:", font=("Arial", 12, "bold")
        ).grid(row=0, column=0, sticky="w")
        cols = ("iter", "f_x", "norm_grad", "alpha")
        self.tree = ttk.Treeview(
            self.secondary_frame, columns=cols, show="headings", height=20
        )
        for c in cols:
            self.tree.heading(c, text=c)
        self.tree.grid(row=1, column=0, columnspan=4, pady=5)

        # Convergence plots (tertiary frame)
        self.terciary_frame = ttk.Frame(
            self.content_frame, style="Borde.TFrame", padding=10
        )
        self.terciary_frame.grid(column=0, columnspan=2, row=1, pady=10)

        # First convergence plot
        self.figure = plt.Figure(figsize=(5, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.terciary_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0)

        # Second convergence plot
        self.figure2 = plt.Figure(figsize=(5, 3), dpi=100)
        self.ax2 = self.figure2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.figure2, self.terciary_frame)
        self.canvas2.get_tk_widget().grid(row=0, column=1)

        # Stats label
        self.stats_label = ttk.Label(self.terciary_frame, text="")
        self.stats_label.grid(row=1, column=0, columnspan=2, pady=5)

    def on_mousewheel(self, event):
        widget = event.widget
        delta = (
            -1 * (event.delta // 120)
            if hasattr(event, "delta")
            else (-1 if event.num == 4 else 1)
        )
        if widget.winfo_class() == "Treeview":
            widget.yview_scroll(delta, "units")
            return "break"
        self.canvas_container.yview_scroll(delta, "units")

    def run_optimization(self):
        try:
            # Read inputs
            func_str = self.func_entry.get()
            vars_ = [v.strip() for v in self.vars_entry.get().split(",")]
            f = symbolic_function(func_str, vars_)
            grad = symbolic_gradient(func_str, vars_)
            x0 = np.array([float(v) for v in self.x0_entry.get().split(",")])
            tol = float(self.tol_entry.get())
            lr = float(self.lr_entry.get())
            max_it = int(self.max_iter_entry.get())

            # Line search params
            ls_it = int(self.ls_max_iter_entry.get())
            rho = float(self.rho_entry.get())
            c_arm = float(self.c_entry.get())
            c1 = float(self.c1_entry.get())
            c2 = float(self.c2_entry.get())

            # Logger + counters
            logger = OptimizerLogger()
            logger.reset()
            evals = {"f": 0, "grad": 0}

            def wf(*args):
                evals["f"] += 1
                return f(*args)

            def wg(*args):
                evals["grad"] += 1
                return grad(*args)

            # Select line search
            ls_opt = self.search_combo.get()
            if ls_opt == "Armijo":
                line_search = lambda f_, g_, x, d: armijo_backtracking(
                    f_, g_, x, d, alpha_init=lr, rho=rho, c=c_arm, max_iter=ls_it
                )
            elif ls_opt == "Wolfe":
                line_search = lambda f_, g_, x, d: wolfe_line_search(
                    f_, g_, x, d, alpha_init=lr, c1=c1, c2=c2, rho=rho, max_iter=ls_it
                )
            else:
                line_search = None

            # Run optimizer
            start = time.perf_counter()
            method = self.method_combo.get()
            if method == "BFGS":
                x_opt, _ = bfgs(
                    wf,
                    wg,
                    x0,
                    tol=tol,
                    max_iter=max_it,
                    line_search=line_search,
                    callback=logger,
                )
            elif method == "Adam":
                x_opt, _ = adam(
                    wf,
                    wg,
                    x0,
                    tol=tol,
                    max_iter=max_it,
                    learning_rate=lr,
                    callback=logger,
                )
            elif method == "SGD":
                x_opt, _ = stochastic_gradient_descent(
                    wf,
                    wg,
                    x0,
                    tol=tol,
                    max_iter=max_it,
                    learning_rate=lr,
                    callback=logger,
                )
            else:
                x_opt, _ = gradient_descent(
                    wf,
                    wg,
                    x0,
                    tol=tol,
                    max_iter=max_it,
                    step_size=lr,
                    line_search=line_search,
                    callback=logger,
                )
            elapsed = time.perf_counter() - start

            # Populate table
            for i in self.tree.get_children():
                self.tree.delete(i)
            for log in logger.get_log():
                self.tree.insert(
                    "",
                    "end",
                    values=(
                        log["iter"],
                        f"{log['f_x']:.6f}",
                        f"{log['norm_grad']:.6f}",
                        f"{log['alpha']:.6f}" if log["alpha"] is not None else "-",
                    ),
                )

            # Update convergence plots
            its = [l["iter"] for l in logger.get_log()]
            fx = [l["f_x"] for l in logger.get_log()]
            ng = [l["norm_grad"] for l in logger.get_log()]

            self.ax.clear()
            self.ax.plot(its, fx, marker="o")
            self.ax.set_title("f(x)")
            self.ax.grid(True)
            self.canvas.draw()

            self.ax2.clear()
            self.ax2.plot(its, ng, marker="o")
            self.ax2.set_title("||∇f||")
            self.ax2.grid(True)
            self.canvas2.draw()

            # Stats
            self.stats_label.config(
                text=(
                    f"Optimum: {np.round(x_opt,6)}   "
                    f"Time: {elapsed:.4f}s   f calls: {evals['f']}   grad calls: {evals['grad']}"
                )
            )

            # Save points for 3D plotting
            self.iter_points = (
                [l["x"][0] for l in logger.get_log()],
                [l["x"][1] for l in logger.get_log()],
            )
            self.opt_point = x_opt

        except Exception as e:
            messagebox.showerror("Error", f"{type(e).__name__}: {e}")

    def on_show_3d_plot(self):
        vars_ = [v.strip() for v in self.vars_entry.get().split(",")]
        show_3d_plot(self, self.func_entry.get(), vars_)

    def on_show_3d_plot_points(self):
        vars_ = [v.strip() for v in self.vars_entry.get().split(",")]
        if hasattr(self, "iter_points"):
            show_3d_plot(self, self.func_entry.get(), vars_, self.iter_points)

    def on_show_contour_plot(self):
        vars_ = [v.strip() for v in self.vars_entry.get().split(",")]
        contour_plot(
            self, self.func_entry.get(), vars_, getattr(self, "opt_point", None)
        )


if __name__ == "__main__":
    app = OptimizerApp()
    app.mainloop()
