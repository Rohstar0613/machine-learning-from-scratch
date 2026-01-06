import numpy as np

def initialize_parameters(X, layers):
    """
    X: 입력 데이터 (N, C, H, W)
    layers: 네트워크 설계도
            예) [("conv", {...}), ("pool", {...}), ("fc", {...})]
    """

    # 모든 레이어의 파라미터(W, b)를 저장할 딕셔너리
    parameters = {}

    # 입력 데이터 차원
    # N은 초기화에는 직접 쓰이지 않지만, 전체 구조 이해를 위해 같이 언패킹
    N, C, H, W = X.shape

    # conv, fc 레이어가 여러 개 있을 수 있으므로 인덱스 관리
    conv = 0
    fc = 0

    # FC 레이어 차원 추적용 변수
    # 첫 FC는 conv/pool 결과를 flatten한 크기에서 시작
    prev_dim = None

    # 레이어 설계도를 앞에서부터 하나씩 순회
    for layer_type, cfg in layers:

        # =============================
        # Conv 레이어 초기화
        # =============================
        if layer_type == "conv":
            # conv 설정값
            filters = cfg["filters"]   # 출력 채널 수
            kernel = cfg["kernel"]     # 커널 크기 (정사각형 가정)
            stride = cfg["stride"]
            pad = cfg["pad"]

            # fan_in = 한 필터가 바라보는 입력 뉴런 개수
            # (입력 채널 * 커널 높이 * 커널 너비)
            fan_in = C * kernel * kernel

            # He initialization
            # ReLU 계열 활성화 함수에서 분산 유지를 위해 사용
            # shape: (filters, C, kernel, kernel)
            w = np.random.randn(filters, C, kernel, kernel) * np.sqrt(2 / fan_in)

            # conv bias는 필터 개수만큼
            b = np.zeros(filters)

            # 파라미터 저장
            parameters[f"conv{conv}"] = {
                "W": w,
                "b": b
            }

            conv += 1

            # conv 연산 이후 feature map 크기 계산
            # forward에서 실제로 나올 출력 크기를 그대로 추적
            layer_h_out = (H + 2 * pad - kernel) // stride + 1
            layer_w_out = (W + 2 * pad - kernel) // stride + 1

            # 다음 레이어를 위한 상태 업데이트
            H = layer_h_out
            W = layer_w_out
            C = filters  # 출력 채널 수는 다음 입력 채널 수가 됨

        # =============================
        # Pool 레이어 (파라미터 없음)
        # =============================
        elif layer_type == "pool":
            poolsize = cfg["size"]
            stride = cfg["stride"]

            # pooling은 공간 크기만 줄이고 채널 수는 유지
            layer_h_out = (H - poolsize) // stride + 1
            layer_w_out = (W - poolsize) // stride + 1

            H = layer_h_out
            W = layer_w_out
            # C는 변하지 않음

        # =============================
        # Fully Connected 레이어 초기화
        # =============================
        elif layer_type == "fc":
            out_dim = cfg["out_dim"]

            # 첫 FC는 conv/pool 결과를 flatten한 크기가 입력
            if prev_dim is None:
                in_dim = C * H * W
            else:
                in_dim = prev_dim

            # fan_in = FC 입력 차원 수
            fan_in = in_dim

            # He initialization (ReLU 기준)
            # W shape: (out_dim, in_dim)
            w = np.random.randn(out_dim, in_dim) * np.sqrt(2 / fan_in)

            # bias는 column vector 형태로 관리
            b = np.zeros((out_dim, 1))

            parameters[f"fc{fc}"] = {
                "W": w,
                "b": b
            }

            fc += 1

            # 다음 FC 레이어를 위해 출력 차원 저장
            prev_dim = out_dim

    # 모든 레이어의 파라미터 초기화 완료
    return parameters

