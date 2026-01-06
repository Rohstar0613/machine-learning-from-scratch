import numpy as np

def fc_forward(A, w, b):
    """
    A: 이전 레이어의 출력 (in_dim, N)
    w: 가중치 행렬 (out_dim, in_dim)
    b: bias (out_dim, 1)
    """

    # Fully Connected layer의 선형 변환
    # Z = W * A + b
    # (out_dim, in_dim) @ (in_dim, N) -> (out_dim, N)
    Z = np.dot(w, A) + b

    # backward 단계에서 gradient 계산에 필요한 값 저장
    # dW 계산에는 A가 필요하고, dA 계산에는 w가 필요하므로
    # 두 값만 cache로 보관
    cache = (A, w)

    return Z, cache

def fc_backward(dZ, cache):
    """
    dZ: (out_dim, N)
        FC layer 출력 Z에 대한 gradient
    cache: forward 단계에서 저장한 값 (A, w)
           A: (in_dim, N)
           w: (out_dim, in_dim)
    """

    A, w = cache

    # =============================
    # 1️⃣ 가중치 gradient
    # =============================
    # Z = W @ A + b
    # ∂L/∂W = dZ @ A^T
    # (out_dim, N) @ (N, in_dim) -> (out_dim, in_dim)
    dW = np.dot(dZ, A.T)

    # =============================
    # 2️⃣ bias gradient
    # =============================
    # bias는 각 뉴런의 출력에 동일하게 더해지므로
    # batch 방향으로 dZ를 모두 더함
    # (out_dim, N) -> (out_dim, 1)
    db = np.sum(dZ, axis=1, keepdims=True)

    # =============================
    # 3️⃣ 입력 gradient
    # =============================
    # ∂L/∂A = W^T @ dZ
    # (in_dim, out_dim) @ (out_dim, N) -> (in_dim, N)
    dZ = np.dot(w.T, dZ)

    return dW, db, dZ