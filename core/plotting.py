import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sympy import lambdify, symbols


def show_3d_plot(root, func_str, variables, points=None):
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
    
    if points is not None:
        points = np.array(points)  # Asegurar que es un array numpy

        percent_to_show = 0.6  # 60% de los puntos más recientes
        
        total_points = len(points[0])
        num_points_to_show = int(total_points * percent_to_show)
        
        x_points = points[0][-num_points_to_show:]
        y_points = points[1][-num_points_to_show:]
        z_points = f_lambdified(x_points, y_points)
        
        valid_mask = (x_points >= -5) & (x_points <= 5) & (y_points >= -5) & (y_points <= 5)
        x_points = x_points[valid_mask]
        y_points = y_points[valid_mask]
        z_points = z_points[valid_mask]
        
        # Graficar todos los puntos
        ax.scatter(x_points, y_points, z_points, color='red', s=50, label='Points by Iteration')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=1, borderaxespad=0.5)
    
    # Mostrar en ventana aparte
    import tkinter as tk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    window = tk.Toplevel(root)
    window.title("3D Visualization")
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

def contour_plot(root, func_str, variables, point=None):
    if len(variables) != 2:
        return
    
    x_sym, y_sym = symbols(variables)
    f_lambdified = lambdify((x_sym, y_sym), func_str, "numpy")
        
    x_range = np.linspace(-10, 10, 100)
    y_range = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    try:
        Z = f_lambdified(X, Y)
    except Exception:
        from tkinter import messagebox

        messagebox.showerror("Error", "Could not evaluate the function.")
        return

    fig = plt.figure(figsize=(6, 4))
    ax_contour = fig.add_subplot(111)
    contour = ax_contour.contourf(X, Y, Z, levels=20, cmap='cividis')
    ax_contour.set_xlabel('x1')
    ax_contour.set_ylabel('x2')
    ax_contour.set_title('Contour Map')
    ax_contour.set_aspect('equal', adjustable='box')

    if point is not None:
        ax_contour.plot(point[0], point[1], 'ro', markersize=8, label=f'Optimal Point ({point[0]:.6f}, {point[1]:.6f})')
        ax_contour.legend()
    
    # Mostrar en ventana aparte
    import tkinter as tk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    window = tk.Toplevel(root)
    window.title("Contour Map Visualization")
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)