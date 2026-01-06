from layers import *

def backward(caches, dZ):
    """
    caches: forward 단계에서 저장한 cache 리스트
            (레이어 순서대로 쌓여 있음)
    dZ: 마지막 layer 출력에 대한 gradient
        (softmax + cross entropy의 경우 probs - onehot(y))
    """

    # 각 파라미터의 gradient를 저장할 딕셔너리
    grads = {}

    # backward는 forward의 역순으로 진행
    for cache in reversed(caches):

        # =============================
        # Fully Connected Layer
        # =============================
        if cache['type'] == "fc":
            name = cache['name']

            # dZ -> (이전 레이어 출력에 대한 gradient)
            # dW, db 계산
            dW, db, dZ = fc_backward(dZ, cache['cache'])

            grads[name] = {"dW": dW, "db": db}

        # =============================
        # ReLU Layer
        # =============================
        elif cache["type"] == "relu":
            Z = cache["cache"]

            # ReLU backward
            # Z > 0 인 위치로만 gradient 전달
            dZ = dZ * (Z > 0)

        # =============================
        # Flatten Layer
        # =============================
        elif cache["type"] == "flatten":
            N, C, H, W = cache['cache']

            # FC로 넘어가기 전에 transpose + flatten 했으므로
            # backward에서는 원래 conv feature map 형태로 복원
            # (D, N) -> (N, D) -> (N, C, H, W)
            dZ = dZ.T.reshape(N, C, H, W)

        # =============================
        # Pooling Layer
        # =============================
        elif cache["type"] == "pool":
            # pooling backward는 mask 기반으로 gradient 전달
            dZ = pool_backward(dZ, cache["cache"])

        # =============================
        # Convolution Layer
        # =============================
        elif cache["type"] == "conv":
            name = cache['name']

            # dZ -> (이전 레이어 출력에 대한 gradient)
            # dW, db 계산
            dZ, dW, db = conv_backward(dZ, cache["cache"])

            grads[name] = {"dW": dW, "db": db}

    return grads