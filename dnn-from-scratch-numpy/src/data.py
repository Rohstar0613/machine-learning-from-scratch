from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
import numpy as np

def make_data():
    from sklearn.datasets import fetch_covtype

    data = fetch_covtype()
    X_data = data.data
    y_data = data.target

    # 1 또는 2인 애들만 남기기
    mask = (y_data == 1) | (y_data == 2)
    X = X_data[mask]
    y_sub = y_data[mask]

    # 1 -> 1, 2 -> 0 으로 매핑 (이진 분류)
    Y = (y_sub == 1).astype(int)

    #데이터 정규화
    mu = np.mean(X, axis=1, keepdims=True)
    sigma = np.std(X, axis=1, keepdims=True) + 1e-8
    X = (X - mu) / sigma

    #행렬을 연산이 가능한 형태로 만들기 위한 전치
    X = X.T
    Y = Y.reshape(1, -1)

    return X, Y