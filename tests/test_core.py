import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from core.functions import quadratic, rastrigin, rosenbrock
from core.gradients import symbolic_function, symbolic_gradient
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
