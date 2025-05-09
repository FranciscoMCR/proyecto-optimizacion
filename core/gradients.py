import numpy as np
import sympy as sp


def symbolic_gradient(func_str: str, variables: list[str]):
    syms = sp.symbols(variables)
    expr = sp.sympify(func_str)
    grad_exprs = [sp.diff(expr, var) for var in syms]
    grad_func = sp.lambdify(syms, grad_exprs, "numpy")
    return grad_func


def symbolic_function(func_str: str, variables: list[str]):
    syms = sp.symbols(variables)
    expr = sp.sympify(func_str)
    func = sp.lambdify(syms, expr, "numpy")
    return func
