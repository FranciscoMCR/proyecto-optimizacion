# 🧠 Optimization Playground

An interactive Python application to test and visualize optimization algorithms using symbolic functions. Built with **Tkinter**, **SymPy**, **NumPy**, **Matplotlib**, and custom optimizers, this project is ideal for experimenting with descent methods and benchmark functions.

---

## 🚀 Features

### ✅ Core Functionality
- Symbolic input of functions (e.g., `x**2 + y**2`)
- Symbolic gradient computation via SymPy
- Support for:
  - **Gradient Descent** (fixed step or line search)
  - **BFGS** (with Armijo or Wolfe conditions)
  - **Adam Optimizer**

### ✅ Line Search Methods
- Armijo Backtracking
- Wolfe Conditions
- Fixed Step Size

### ✅ Benchmark Functions Included
- Quadratic
- Rosenbrock
- Rastrigin
- Himmelblau
- Ackley
- Griewank

### ✅ GUI Features
- Input function, variables, initial point, tolerance, method, and line search
- Real-time convergence plots:
  - `f(x)` per iteration
  - `‖∇f(x)‖` per iteration
- Iteration table with:
  - Iteration number
  - Function value `f(x)`
  - Gradient norm `‖∇f‖`
  - Step size `alpha`
- **Execution summary** after optimization:
  - Runtime in seconds
  - Number of function evaluations
  - Number of gradient evaluations

---

## 📦 Requirements

Install dependencies:
pip install -r requirements.txt

## 🖥️ How to Run
To launch the GUI:
python main.py

Steps:

1. Enter a symbolic function (e.g., x**2 + y**2)

2. Enter variables separated by commas (e.g., x,y)

3. Provide the initial point (e.g., 3,4)

4. Set tolerance (e.g., 1e-6)

5. Choose the optimizer (Gradient Descent, BFGS, or Adam)

6. Select the line search method if needed (None, Armijo, or Wolfe)

7. Click Run

## 🧪 Running Tests

To validate core components:
python tests/test_core.py

This script tests:
- Function evaluations
- Symbolic gradient correctness
- Optimizer behavior on various functions

## 📁 Project Structure

proyecto_optimizacion/
├── core/
│   ├── functions.py         # Benchmark objective functions
│   ├── gradients.py         # Symbolic gradient and function tools
│   ├── line_search.py       # Line search implementations
│   ├── optimizers.py        # Optimization algorithms (GD, BFGS, Adam)
│   ├── utils.py             # Input parsing and validation
│   ├── logger.py            # Per-iteration logging for GUI
├── tests/
│   └── test_core.py         # Unit tests for all modules
├── main.py                  # Tkinter-based graphical interface
├── requirements.txt
└── README.md


