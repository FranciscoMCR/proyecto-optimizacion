import numpy as np


def parse_input_vector(input_str: str) -> np.ndarray:
    try:
        return np.array([float(val.strip()) for val in input_str.split(",")])
    except Exception as e:
        raise ValueError("Vector inválido. Usa formato: 1.0, 2.0, 3.0") from e


def check_tolerance(tol: float):
    if not (1e-12 <= tol <= 1e-1):
        raise ValueError("La tolerancia debe estar entre 1e-12 y 1e-1.")
