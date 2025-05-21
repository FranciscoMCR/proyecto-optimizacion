import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sympy import lambdify, symbols


def show_3d_plot(root, func_str, variables, point=None):
    if len(variables) != 2:
        return

    x_sym, y_sym = symbols(variables)
    f_lambdified = lambdify((x_sym, y_sym), func_str, "numpy")

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    try:
        Z = f_lambdified(X, Y)
    except Exception:
        from tkinter import messagebox

        messagebox.showerror("Error", "Could not evaluate the function.")
        return

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_title("3D Surface of f(x, y)")
    ax.set_xlabel(variables[0])
    ax.set_ylabel(variables[1])
    ax.set_zlabel("f(x, y)")
    
    if point is not None:
        x_point, y_point = point
        z_point = f_lambdified(x_point, y_point)
        ax.scatter(x_point, y_point, z_point, color='red', s=50, label=f'Optimal point \n[{x_point:.6f}, {y_point:.6f}]')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), borderaxespad=0.1)
    
    # Mostrar en ventana aparte
    import tkinter as tk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    window = tk.Toplevel(root)
    window.title("3D Visualization")
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
