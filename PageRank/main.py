import numpy as np

def pagerank(M, d=0.85, tol=1e-6):
    n = M.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v, 1)
    last_v = np.ones((n, 1), dtype=np.float32) * float('inf')
    M_hat = d * M + (1 - d) / n
    while np.linalg.norm(v - last_v, 2) > tol:
        last_v = v
        v = M_hat @ v
    return v
