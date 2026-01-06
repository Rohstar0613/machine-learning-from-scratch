from layers import *

def forward(X, layers, parameters):
    """
    X: 입력 데이터
    layers: 네트워크 설계도 (순서 중요)
    parameters: 각 conv / fc 레이어의 파라미터 딕셔너리
    """

    # backward를 위해 각 레이어의 cache를 순서대로 저장
    caches = []

    # 현재 활성값 (초기 입력)
    A = X

    # conv / fc 레이어 인덱스 관리
    conv = 0
    fc_count = 0

    # 레이어 설계도를 앞에서부터 순차적으로 실행
    for layer_type, cfg in layers:

        # =============================
        # Convolution Layer
        # =============================
        if layer_type == "conv":
            name = f"conv{conv}"

            stride = cfg["stride"]
            pad = cfg["pad"]

            params = parameters[name]
            w = params["W"]
            b = params["b"]

            # Conv forward
            Z, cache = conv_forward(A, w, b, stride, pad)

            # backward를 위한 cache 저장
            caches.append({
                "type": "conv",
                "name": name,
                "cache": cache
            })

            conv += 1

        # =============================
        # ReLU Layer
        # =============================
        elif layer_type == "relu":
            # ReLU는 직전 conv / fc의 Z에 적용
            A, cache = ReLU(Z)

            caches.append({
                "type": "relu",
                "cache": cache
            })

        # =============================
        # Pooling Layer
        # =============================
        elif layer_type == "pool":
            poolsize = cfg["size"]
            stride = cfg["stride"]

            A, cache = maxpool_forward(A, poolsize, stride)

            caches.append({
                "type": "pool",
                "cache": cache
            })

        # =============================
        # Flatten Layer
        # =============================
        elif layer_type == "flatten":
            N, C, H, W = A.shape

            # (N, C, H, W) -> (N, D) -> (D, N)
            A = A.reshape(N, -1).T

            caches.append({
                "type": "flatten",
                "cache": (N, C, H, W)
            })

        # =============================
        # Fully Connected Layer
        # =============================
        elif layer_type == "fc":
            name = f"fc{fc_count}"

            params = parameters[name]
            w = params["W"]
            b = params["b"]

            Z, cache = fc_forward(A, w, b)

            caches.append({
                "type": "fc",
                "name": name,
                "cache": cache
            })

            fc_count += 1

    # 마지막 레이어의 출력 Z와 전체 cache 반환
    return Z, caches
    