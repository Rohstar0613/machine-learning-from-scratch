import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from model.forward import forward

def plot_cost_curve(costs):
    a = np.min(costs)
    b = np.max(costs)
    pad = (b - a) * 0.01  # 10% padding

    plt.plot(costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Learning Curve (Cost vs Iteration)")
    plt.ylim(a - pad, b + pad)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def evaluate_classification_model(Z_best, y):
    # ----------------------------------
    # 예측값 생성 (softmax / logits 공용)
    # Z_best: (C, N)
    # ----------------------------------
    y_pred = np.argmax(Z_best, axis=0)   # (N,)
    y_true = y                           # (N,)

    # ----------------------------------
    # 정확도
    # ----------------------------------
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)

    # ----------------------------------
    # Classification Report (DataFrame)
    # ----------------------------------
    report_df = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True)
    ).T

    print("\nClassification Report:\n")
    print(report_df)

    # ----------------------------------
    # Confusion Matrix
    # ----------------------------------
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def evaluate_best_model(X, y, layers, best_parameters):
    # =============================
    # Evaluation (Best Model)
    # =============================

    # 학습 중 가장 낮은 loss를 기록한 파라미터로 최종 평가
    Z_best, _ = forward(X, layers, best_parameters)
    y_pred = np.argmax(Z_best, axis=0)
    accuracy = np.mean(y_pred == y)

    print("Final Accuracy :", accuracy)

    return Z_best