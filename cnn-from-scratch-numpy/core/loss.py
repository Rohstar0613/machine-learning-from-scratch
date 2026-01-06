import numpy as np

def softmax_cross_entropy(Z, y):
    """
    Z: (C, N) logits
       C: 클래스 수
       N: 배치 크기
    y: (N,) 정답 레이블 (0 ~ C-1)
    """

    # 수치 안정성을 위해 각 샘플별로 max logit을 빼줌
    # softmax 값은 변하지 않지만 exp overflow를 방지
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)

    # exp 계산
    exp_Z = np.exp(Z_shift)

    # 각 샘플별 exp 합
    sum_exp = np.sum(exp_Z, axis=0, keepdims=True)

    # softmax 확률 계산
    # shape: (C, N)
    probs = exp_Z / sum_exp

    # log 계산 시 log(0) 방지를 위한 작은 값 추가
    # (이론적으로는 없어도 되지만, 수치 안정성 확보)
    log_probs = np.log(probs + 1e-12)

    # 정답 클래스에 해당하는 log probability만 선택
    # 각 샘플마다 정답 클래스의 loss 계산
    loss = -log_probs[y, np.arange(Z.shape[1])]

    # 배치 평균 loss 반환
    return np.mean(loss), probs


def get_initial_dZ(y_hat, y):
    """
    y_hat: (C, N) logits (softmax 이전 값)
    y: (N,) 정답 레이블
    """

    # 수치 안정성을 위한 shift
    # softmax의 결과는 변하지 않지만 exp overflow를 방지
    exp_Z = np.exp(y_hat - np.max(y_hat, axis=0, keepdims=True))

    # softmax 확률 계산
    probs = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    # 배치 크기
    N = y.shape[0]

    # softmax + cross entropy의 미분 결과
    # 정답 클래스 위치에서 (p - 1),
    # 나머지 클래스에서는 p
    probs[y, np.arange(N)] -= 1

    # dZ = softmax(Z) - onehot(y)
    return probs