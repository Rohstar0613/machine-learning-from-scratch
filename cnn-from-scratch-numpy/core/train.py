from model.backward import *
from model.forward import *
from model.initialize_parameters import *
from core.loss import *
from core.optim import *
import numpy as np

def train(X, y, num_iters, lr, layers):
    # =============================
    # Training Loop
    # =============================

    costs = []

    # 가장 낮은 loss를 기록한 모델 저장용
    best_cost = np.inf

    # 파라미터 초기화
    parameters = initialize_parameters(X, layers)

    for i in range(num_iters):

        # -----------------------------
        # Forward pass
        # -----------------------------
        Z, caches = forward(X, layers, parameters)

        # Loss 계산 (softmax + cross entropy)
        loss, probs = softmax_cross_entropy(Z, y)
        costs.append(loss)

        # -----------------------------
        # Best model 저장
        # -----------------------------
        # loss가 가장 낮을 때의 파라미터를 복사해서 저장
        if loss < best_cost:
            best_cost = loss
            best_parameters = {
                k: {
                    "W": v["W"].copy(),
                    "b": v["b"].copy()
                } for k, v in parameters.items()
            }

        # -----------------------------
        # Initial gradient (dZ)
        # -----------------------------
        # softmax + cross entropy의 미분 결과
        # dZ = probs - onehot(y)
        dZ = probs.copy()
        dZ[y, np.arange(y.shape[0])] -= 1

        # -----------------------------
        # Backward pass
        # -----------------------------
        grads = backward(caches, dZ)

        # -----------------------------
        # Parameter update (SGD)
        # -----------------------------
        update_params(parameters, grads, lr)

        print(f"Iteration {i+1}, Cost = {loss:.4f}")

    return best_parameters, costs