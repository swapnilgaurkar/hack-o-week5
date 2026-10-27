import numpy as np

def vector_operations(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return {
        "addition": (a + b).tolist(),
        "subtraction": (a - b).tolist(),
        "magnitude_a": float(np.linalg.norm(a)),
        "magnitude_b": float(np.linalg.norm(b)),
        "dot_product": float(np.dot(a, b))
    }
