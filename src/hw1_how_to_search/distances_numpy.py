import numpy as np

def euclidean_distance(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = queries if queries.ndim == 2 else queries[None, :]
    return np.sqrt(((q[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2))

def cosine_similarity(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = queries if queries.ndim == 2 else queries[None, :]
    q_norm = q / np.linalg.norm(q, axis=1, keepdims=True)
    c_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return q_norm @ c_norm.T

def manhattan_distance(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = queries if queries.ndim == 2 else queries[None, :]
    return np.abs(q[:, None, :] - matrix[None, :, :]).sum(axis=2)

def dot_product(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = queries if queries.ndim == 2 else queries[None, :]
    return q @ matrix.T