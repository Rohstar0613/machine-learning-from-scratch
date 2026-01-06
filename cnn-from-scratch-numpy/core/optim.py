import numpy as np

def update_params(parameters, grads, lr):
    """
    parameters: 모델 파라미터 딕셔너리
                {
                  "conv0": {"W": ..., "b": ...},
                  "fc0":   {"W": ..., "b": ...},
                  ...
                }
    grads: backward 단계에서 계산된 gradient 딕셔너리
           {
             "conv0": {"dW": ..., "db": ...},
             "fc0":   {"dW": ..., "db": ...},
             ...
           }
    lr: learning rate
    """

    # 모든 파라미터에 대해 gradient descent 수행
    for name in parameters:

        # 가중치 업데이트
        # W = W - lr * dW
        parameters[name]["W"] -= lr * grads[name]["dW"]

        # bias 업데이트
        # b = b - lr * db
        parameters[name]["b"] -= lr * grads[name]["db"]