import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans

def compute_kernel_matrix(X, kernel_func):
    n = X.shape[1]
    K = np.zeros((n, n))
    for i in range(n):
        K[i, :] = kernel_func(X[:, i][:, None], X)
    print("Kernel matrix shape:", K.shape)
    print("Kernel matrix sample:")
    print(K[:5, :5])
    return K

def update_H(X, W, H, U, G, alpha, beta, lambda_param):
    XW = X.T @ W
    numerator = beta * XW
    denominator = G @ H + beta * (H - XW) + lambda_param * H
    H = np.where(denominator != 0, H * numerator / (denominator + 1e-10), H)
    H = np.maximum(H, 1e-10)  # 确保 H 中没有零值
    return H

def update_W(X, W, H, U, alpha):
    XH = X @ H
    denominator = (X @ X.T @ W + alpha * U @ W)
    W = np.where(denominator != 0, W * XH / (denominator + 1e-10), W)
    W = np.maximum(W, 1e-10)  # 确保 W 中没有零值
    return W

def update_U(W):
    norm_W = np.linalg.norm(W, axis=1)
    U = np.diag(1 / (2 * norm_W + 1e-10))
    return U

def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

def gevd(A, B):
    eigvals, eigvecs = eigh(A, B)
    ind = np.argsort(eigvals)[::-1]  # Sort in descending order
    eigvals_sorted = eigvals[ind]
    eigvecs_sorted = eigvecs[:, ind]
    model = {'W': eigvecs_sorted, 'D': np.diag(eigvals_sorted)}
    print("GEVD eigenvalues:", eigvals_sorted[:5])
    return model

def KFDRL(X, c, t, alpha, beta, sigma, mu, lambda_param, s):
    print("Starting KFDRL...")
    G = compute_kernel_matrix(X, lambda x, y: gaussian_kernel(x, y, sigma))
    d, n = X.shape
    U = np.eye(d)
    H = np.ones((n, c))
    W = np.random.rand(d, c)

    for i in range(t):
        if i % 10 == 0:
            print(f"KFDRL iteration {i+1}/{t}")
        H = update_H(X, W, H, U, G, alpha, beta, lambda_param)
        W = update_W(X, W, H, U, alpha)
        U = update_U(W)

    feature_scores = np.linalg.norm(W, axis=1)
    selected_features = np.argsort(feature_scores)[-s:]
    print("Selected features:", selected_features[:10])
    return selected_features, W

def unkfdapc(X, c, Npc, Ninit=20, gamma=1e-6, tol=1e-6, max_iter=300, Ntry=20, center=True, no_pca=False, alpha=0.5, beta=0.5, sigma=1.0,
              mu=1e-12, lambda_param=1e8):
    print("Starting unkfdapc...")
    n, d = X.shape
    print(f"Input data shape: {n} samples, {d} features")

    if center:
        H = np.eye(n) - np.ones((n, n)) / n
    else:
        H = np.eye(n)

    St = X.T @ H @ X
    Stt = St + gamma * np.eye(d)

    it = 0
    obj_old = -np.inf
    obj_new = 0.0
    Ypre = None
    T = None

    # m = min(d, c - 1, Npc)
    m = Npc
    if no_pca:
        W = X.T[:, :m]
    else:
        selected_features, W = KFDRL(X.T, c, max_iter, alpha, beta, sigma, mu, lambda_param, s=m)
    print("Initial W shape:", W.shape)

    W2 = W

    obj_log = []

    while (not np.isclose(obj_old, obj_new, atol=tol) or it == 0) and it < max_iter:
        it += 1
        obj_old = obj_new

        T = (W.T @ X.T @ H).T
        print(f"Iteration {it}: T shape:", T.shape)

        kmeans = KMeans(n_clusters=c, tol=tol, max_iter=300, n_init=20, random_state=42)
        Ypre = kmeans.fit_predict(T)
        print(f"Iteration {it}: Unique cluster labels:", np.unique(Ypre))

        Yp = np.eye(c)[Ypre]

        YpTYp_inv = np.linalg.inv(Yp.T @ Yp + 1e-10 * np.eye(c))
        Sb = X.T @ H @ Yp @ YpTYp_inv @ Yp.T @ H.T @ X

        model = gevd(Sb, Stt)
        W2 = model['W'][:, :m]

        obj_new = np.trace(np.linalg.inv(W2.T @ Stt @ W2) @ W2.T @ Sb @ W2)
        print(f"Iteration {it}: Objective value:", obj_new)

        obj_log.append(obj_new)

    if it == max_iter:
        print(f"Warning: The un_rtlda did not converge within {max_iter} iterations!")
    else:
        print(f"Converged after {it} iterations")

    print("Final T shape:", T.shape)
    print("Final Ypre shape:", Ypre.shape)
    print("Final W2 shape:", W2.shape)
    print("Objective value log:", obj_log)

    return T, Ypre, W2, obj_log