import numpy as np

def conv_forward(A, w, b, stride=1, pad=0):
    """
    A: 입력 데이터 (N, C, H, W)
    w: 필터 가중치 (F, C, KH, KW)
    b: bias (F,)
    stride: stride 크기
    pad: zero padding 크기
    """

    # 입력 데이터 차원 언패킹
    # N: 배치 크기, C: 채널 수, H/W: 입력 이미지 높이/너비
    N, C, H, W = A.shape

    # 필터 정보 언패킹
    # F: 필터 개수, KH/KW: 필터 높이/너비
    F, _, KH, KW = w.shape

    # 출력 feature map의 공간 크기 계산
    # Conv 연산의 공식 그대로 사용
    H_out = (H + 2 * pad - KH) // stride + 1
    W_out = (W + 2 * pad - KW) // stride + 1

    # padding이 필요한 경우 zero padding 수행
    # 가장자리에 대한 정보 손실을 방지하기 위함
    if pad > 0:
        A_pad = np.pad(
            A,
            ((0, 0), (0, 0), (pad, pad), (pad, pad)),
            mode="constant"
        )
    else:
        # padding이 없으면 원본 입력 그대로 사용
        A_pad = A

    # 출력 feature map을 0으로 초기화
    # shape: (N, F, H_out, W_out)
    Z = np.zeros((N, F, H_out, W_out))

    # 배치 단위로 convolution 수행
    for n in range(N):

        # 각 필터에 대해 convolution 수행
        for f in range(F):

            # 출력 feature map의 세로 방향
            for i in range(H_out):

                # 출력 feature map의 가로 방향
                for j in range(W_out):

                    # stride를 고려한 현재 receptive field의 시작 위치
                    h_start = i * stride
                    w_start = j * stride

                    # 현재 위치에서 필터가 볼 입력 영역(receptive field) 추출
                    patch = A_pad[n, :, h_start : h_start + KH, w_start : w_start + KW]

                    # 필터와 입력 패치의 element-wise 곱 → 합
                    # bias를 더해 최종 convolution 결과 계산
                    Z[n, f, i, j] = np.sum(patch * w[f]) + b[f]

    # backward 단계에서 gradient 계산을 위해 필요한 값들을 cache로 저장
    # A_pad를 따로 저장해두면 padding 관련 역전파 구현이 수월해짐
    cache = (A, w, b, stride, pad, A_pad)

    # forward 결과 Z와 cache 반환
    return Z, cache


def conv_backward(dZ, cache):
    """
    dZ: (N, F, H_out, W_out)
        convolution layer 출력에 대한 gradient
    cache: forward 단계에서 저장한 값
           (A_prev, W, b, stride, pad, A_prev_pad)

    returns:
        dA_prev: (N, C, H, W)   입력 A에 대한 gradient
        dW:      (F, C, KH, KW) 필터 가중치 gradient
        db:      (F,)           bias gradient
    """

    # forward 단계에서 저장한 값 복원
    A_prev, W, b, stride, pad, A_prev_pad = cache

    # 차원 정보
    N, C, H, W_in = A_prev.shape
    F, _, KH, KW = W.shape
    _, _, H_out, W_out = dZ.shape

    # =============================
    # 1️⃣ gradient 초기화
    # =============================

    # padding까지 포함한 입력 gradient
    # (forward에서 padding을 했으므로 backward에서도 동일하게 계산)
    dA_prev_pad = np.zeros_like(A_prev_pad)

    # 가중치 gradient
    dW = np.zeros_like(W)

    # bias gradient
    db = np.zeros(F)

    # =============================
    # 2️⃣ bias gradient
    # =============================
    # 각 필터의 출력 dZ를 모두 더하면 bias에 대한 gradient
    db = np.sum(dZ, axis=(0, 2, 3))

    # =============================
    # 3️⃣ 핵심 loop (Chain Rule)
    # =============================
    # dZ 하나가
    #  - dW에는: 해당 입력 patch에 비례하여 누적
    #  - dA에는: 해당 필터 W를 통해 책임이 분배됨
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                h_start = i * stride
                for j in range(W_out):
                    w_start = j * stride

                    dz = dZ[n, f, i, j]

                    # forward 단계에서 사용한 receptive field
                    patch = A_prev_pad[n, :,
                                       h_start : h_start + KH,
                                       w_start : w_start + KW]

                    # ∂L/∂W = input patch * dZ
                    dW[f] += patch * dz

                    # ∂L/∂A = W * dZ
                    # 해당 필터가 본 영역에 gradient를 흩뿌림
                    dA_prev_pad[n, :,
                                h_start : h_start + KH,
                                w_start : w_start + KW] += W[f] * dz

    # =============================
    # 4️⃣ padding 제거
    # =============================
    # 실제 입력 크기에 맞게 gradient 복원
    if pad > 0:
        dA_prev = dA_prev_pad[:, :, pad:-pad, pad:-pad]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
