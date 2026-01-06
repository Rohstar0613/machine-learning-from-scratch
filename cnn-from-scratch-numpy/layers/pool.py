import numpy as np

def maxpool_forward(A, poolsize, stride):
    """
    A: 입력 feature map (N, C, H, W)
    poolsize: pooling window 크기
    stride: stride 크기
    """

    # 입력 차원 언패킹
    N, C, H, W = A.shape

    # pooling 이후 출력 feature map 크기 계산
    H_out = (H - poolsize) // stride + 1
    W_out = (W - poolsize) // stride + 1

    # 출력 feature map 초기화
    A_out = np.zeros((N, C, H_out, W_out))

    # backward 단계에서 gradient를 전달할 위치를 기록하기 위한 mask
    # 입력과 동일한 shape으로 생성
    mask = np.zeros_like(A)

    # 배치 단위로 pooling 수행
    for n in range(N):

        # 채널별로 독립적으로 max pooling 수행
        for f in range(C):

            # 출력 feature map의 세로 방향
            for i in range(H_out):

                # 출력 feature map의 가로 방향
                for j in range(W_out):

                    # 현재 pooling window의 시작 위치
                    h0 = i * stride
                    w0 = j * stride

                    # pooling window 추출
                    window = A[n, f,
                               h0 : h0 + poolsize,
                               w0 : w0 + poolsize]

                    # window 내 최대값 계산
                    max_val = np.max(window)

                    # forward 결과 저장
                    A_out[n, f, i, j] = max_val

                    # backward를 위해 max 값의 위치 기록
                    # 동일한 값이 여러 개일 경우 해당 위치들 모두 기록
                    max_mask = (window == max_val)

                    mask[n, f,
                         h0 : h0 + poolsize,
                         w0 : w0 + poolsize] += max_mask

    # backward에서 필요한 정보 cache로 저장
    # mask를 저장해두면 gradient를 max 위치로만 전달 가능
    cache = (A.shape, poolsize, stride, mask)

    return A_out, cache


def pool_backward(dA, cache):
    """
    dA: pooling layer의 출력에 대한 gradient (N, C, H_out, W_out)
    cache: forward 단계에서 저장한 정보
           (A_shape, poolsize, stride, mask)
    """

    # forward 단계에서 저장해둔 입력 shape, pooling 설정, mask 복원
    A_shape, poolsize, stride, mask = cache
    N, C, H, W = A_shape

    # 입력 A에 대한 gradient 초기화
    # pooling은 입력 크기를 줄이므로, backward에서는 다시 원래 크기로 복원
    dA_prev = np.zeros((N, C, H, W))

    # pooling 출력의 공간 크기
    H_out = dA.shape[2]
    W_out = dA.shape[3]

    # 배치 단위로 gradient 전달
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):

                    # forward에서 사용했던 pooling window의 시작 위치
                    h0 = i * stride
                    w0 = j * stride

                    # max pooling에서는
                    # 해당 window 내에서 max였던 위치로만 gradient를 전달
                    # mask는 max 위치에만 1이 있음
                    dA_prev[n, c,
                            h0 : h0 + poolsize,
                            w0 : w0 + poolsize] += (
                        mask[n, c,
                             h0 : h0 + poolsize,
                             w0 : w0 + poolsize] * dA[n, c, i, j]
                    )

    return dA_prev