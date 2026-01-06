from data.mnist_loader import mnist_loader
from core.train import train
from core.eval import (
    evaluate_best_model,
    plot_cost_curve,
    evaluate_classification_model
)

# 모델 설계도
layers = [
    ("conv", {"filters": 16, "kernel": 3, "stride": 1, "pad": 1}),
    ("relu", {}),

    ("conv", {"filters": 16, "kernel": 3, "stride": 1, "pad": 1}),
    ("relu", {}),

    ("pool", {"size": 2, "stride": 2}),

    ("conv", {"filters": 32, "kernel": 3, "stride": 1, "pad": 1}),
    ("relu", {}),

    ("conv", {"filters": 32, "kernel": 3, "stride": 1, "pad": 1}),
    ("relu", {}),
    
    ("pool", {"size": 2, "stride": 2}),

    ("flatten", {}),

    ("fc", {"out_dim": 128}),
    ("relu", {}),

    ("fc", {"out_dim": 10})
]

num_iters = 10
lr = 0.001

# 데이터 로드
X, y = mnist_loader()

X = X[:5]
y = y[:5]

# 모델 학습
best_parameters, costs = train(X, y, num_iters, lr, layers)

# 가장 작은 loss를 가진 파라미터로 모델 평가
Z_best = evaluate_best_model(X, y, layers, best_parameters)

# loss 결과 시각화
plot_cost_curve(costs)

# 예측 결과 및 혼동행렬 출력
evaluate_classification_model(Z_best, y)
