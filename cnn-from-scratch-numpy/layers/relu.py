import numpy as np

def ReLU(Z):
    """
    Z: linear 또는 conv layer의 출력
    """

    # ReLU 활성화 함수
    # 음수는 0으로 잘라내고, 양수는 그대로 통과시킨다
    # 계산이 단순해 forward/backward 모두 안정적임
    A = np.maximum(0, Z)

    # backward 단계에서 gradient 계산을 위해
    # 입력 Z를 그대로 cache에 저장
    # (Z > 0 여부만 필요하지만, 구현 단순화를 위해 전체 Z 저장)
    cache = (Z)

    return A, cache