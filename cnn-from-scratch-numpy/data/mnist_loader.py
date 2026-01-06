from sklearn.datasets import fetch_openml
import numpy as np

def mnist_loader():
    # OpenML에서 MNIST 데이터셋 로드
    # as_frame=False로 설정해 numpy array 형태로 바로 사용
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    # 입력 데이터
    # 원본 픽셀 값은 [0, 255] 범위이므로 float32로 변환 후 정규화
    # shape: (70000, 784)
    X = mnist.data.astype(np.float32) / 255.0

    # 정답 레이블
    # 문자열 형태이므로 int64로 변환
    y = mnist.target.astype(np.int64)

    # CNN 입력 형태로 reshape
    # (N, C, H, W) 형태를 맞추기 위함
    # MNIST는 grayscale 이미지이므로 C = 1
    X = X.reshape(-1, 1, 28, 28)

    # 전체 데이터를 그대로 쓰기보다
    # from-scratch CNN 검증을 위해 소량 샘플만 사용
    # (연산량 감소 + 디버깅 용이)
    np.random.seed(0)
    idx = np.random.choice(len(X), 100, replace=False)

    X = X[idx]
    y = y[idx]

    # 데이터 차원 확인
    # CNN forward/backward에서 shape mismatch를 방지하기 위함
    print("X:", X.shape)  # (100, 1, 28, 28)
    print("y:", y.shape)  # (100,)

    return X, y