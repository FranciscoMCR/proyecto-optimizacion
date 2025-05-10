import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from core.functions import (ackley, griewank, himmelblau, quadratic, rastrigin,
                            rosenbrock)
from core.gradients import symbolic_function, symbolic_gradient
from core.line_search import armijo_backtracking, wolfe_line_search
from core.logger import OptimizerLogger
from core.optimizers import adam, bfgs, gradient_descent
from core.utils import parse_input_vector

# Test 1: Quadratic function
print("🔹 Test: Quadratic Function")
x = np.array([1.0, 2.0])
A = np.array([[2, 0], [0, 2]])
b = np.array([-2, -8])
c = 0
result = quadratic(x, A, b, c)
print(f"f(x) = {result:.4f} (Expected: -13.0)")

# Test 2: Rosenbrock function
print("\n🔹 Test: Rosenbrock Function")
x = np.array([1.0, 1.0])
result = rosenbrock(x)
print(f"f(x) = {result:.4f} (Expected: 0.0)")

# Test 3: Rastrigin function
print("\n🔹 Test: Rastrigin Function")
x = np.array([0.0, 0.0])
result = rastrigin(x)
print(f"f(x) = {result:.4f} (Expected: 0.0)")

# Test 4: Symbolic Gradient
print("\n🔹 Test: Symbolic Gradient")
func_str = "x**2 + y**2"
variables = ["x", "y"]
grad_func = symbolic_gradient(func_str, variables)
grad_val = grad_func(3.0, 4.0)
print(f"∇f(3,4) = {grad_val} (Expected: [6.0, 8.0])")

# Test 5: Symbolic Function Evaluation
print("\n🔹 Test: Symbolic Function Evaluation")
f = symbolic_function(func_str, variables)
print(f"f(3,4) = {f(3.0, 4.0)} (Expected: 25.0)")

# Test 6: Vector parsing
print("\n🔹 Test: Input Parsing")
parsed = parse_input_vector("1.0, 2.0, 3.5")
print(f"Parsed: {parsed} (Expected: [1.0 2.0 3.5])")

# Test 6: Gradient
print("\n🔹 Test: Descenso por Gradiente (sin búsqueda lineal)")

# Función y gradiente simbólicos
func_str = "x**2 + y**2"
vars = ["x", "y"]
f = symbolic_function(func_str, vars)
grad_f = symbolic_gradient(func_str, vars)

# Punto inicial
x0 = np.array([3.0, 4.0])

# Ejecutar optimizador
x_opt, history = gradient_descent(
    f=f, grad_f=grad_f, x0=x0, step_size=0.1, tol=1e-6, max_iter=100
)

print(f"Iteraciones: {len(history)}")
print(f"x óptimo ≈ {x_opt}")
print(f"f(x) ≈ {f(*x_opt):.6f} (Expected ≈ 0)")

# Test 7: Gradient Descent with Armijo Line Search
print("\n🔹 Test: Gradiente con búsqueda lineal (Armijo)")

# Reusar f y grad_f de antes
x0 = np.array([3.0, 4.0])

x_opt, history = gradient_descent(
    f=f, grad_f=grad_f, x0=x0, tol=1e-6, max_iter=100, line_search=armijo_backtracking
)

print(f"Iteraciones: {len(history)}")
print(f"x óptimo ≈ {x_opt}")
print(f"f(x) ≈ {f(*x_opt):.6f} (Expected ≈ 0)")

# Test 8: Himmelblau function

print("\n🔹 Test: Himmelblau Function")
x = np.array([3.0, 2.0])
result = himmelblau(x)
print(f"f(3,2) = {result:.4f} (Expected: 0.0)")

# Test 9: Ackley function

print("\n🔹 Test: Ackley Function")
x = np.zeros(2)
result = ackley(x)
print(f"f(0,0) = {result:.6f} (Expected: 0.0)")

# Test 10: Optimización con Ackley Function (gradiente simbólico)

print("\n🔹 Test: Optimización con Ackley Function (gradiente simbólico)")

# Redefinir la función y gradiente de forma simbólica
ackley_str = "-20 * exp(-0.2 * sqrt(0.5*(x**2 + y**2))) - exp(0.5 * (cos(2*pi*x) + cos(2*pi*y))) + 20 + exp(1)"
vars = ["x", "y"]
f_ackley = symbolic_function(ackley_str, vars)
grad_ackley = symbolic_gradient(ackley_str, vars)

x0 = np.array([2.0, 2.0])

x_opt, history = gradient_descent(
    f=f_ackley,
    grad_f=grad_ackley,
    x0=x0,
    tol=1e-6,
    max_iter=200,
    line_search=armijo_backtracking,
)

print(f"Iteraciones: {len(history)}")
print(f"x óptimo ≈ {x_opt}")
print(f"f(x) ≈ {f_ackley(*x_opt):.6f} (Expected ≈ 0)")

# Test 11: BFGS con Ackley Function (sin búsqueda lineal)

print("\n🔹 Test: BFGS con Ackley Function (Wolfe)")

x0 = np.array([2.0, 2.0])
x_opt, history = bfgs(
    f=f_ackley,
    grad_f=grad_ackley,
    x0=x0,
    tol=1e-6,
    max_iter=200,
    line_search=wolfe_line_search,
)

print(f"Iteraciones: {len(history)}")
print(f"x óptimo ≈ {x_opt}")
print(f"f(x) ≈ {f_ackley(*x_opt):.6f} (Expected ≈ 0)")

# Test 12: BFGS con Ackley Function + Logger

print("\n🔹 Test: BFGS con Ackley Function + Logger")

logger = OptimizerLogger()

x0 = np.array([2.0, 2.0])
x_opt, history = bfgs(
    f=f_ackley,
    grad_f=grad_ackley,
    x0=x0,
    tol=1e-6,
    max_iter=200,
    line_search=wolfe_line_search,
    callback=logger,
)

print(f"x óptimo ≈ {x_opt}")
print(f"f(x) ≈ {f_ackley(*x_opt):.6f}")
logger.print_summary()

# Test 13: Griewank function
print("\n🔹 Test: Griewank Function")
x = np.zeros(2)
result = griewank(x)
print(f"f(0,0) = {result:.6f} (Expected: 0.0)")

# Test 14: Adam Optimizer with Rastrigin function

print("\n🔹 Test: Optimización con Adam (cuadrática)")

func_str = "x**2 + y**2"
vars = ["x", "y"]
f = symbolic_function(func_str, vars)
grad_f = symbolic_gradient(func_str, vars)
x0 = np.array([3.0, 4.0])

x_opt, history = adam(
    f=f,
    grad_f=grad_f,
    x0=x0,
    tol=1e-6,
    max_iter=1000,
    alpha=0.05,  # tasa de aprendizaje
)

print(f"Iteraciones: {len(history)}")
print(f"x óptimo ≈ {x_opt}")
print(f"f(x) ≈ {f(*x_opt):.6f} (Expected ≈ 0)")
