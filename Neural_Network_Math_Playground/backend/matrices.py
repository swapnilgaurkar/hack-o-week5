import numpy as np

def matrix_operations(A, B, vector):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    v = np.array(vector, dtype=float)

    result = {
        "A_plus_B": (A + B).tolist(),
        "A_times_B": (A @ B).tolist(),
        "A_times_vector": (A @ v).tolist()
    }

    if A.shape[0] == A.shape[1]:
        result["determinant_A"] = float(np.linalg.det(A))
    else:
        result["determinant_A"] = None

    return result
