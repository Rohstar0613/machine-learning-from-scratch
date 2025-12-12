from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def save_parameters(best_parameters, filename="best_params.npz"):
    np.savez(filename, **best_parameters)
    print(f"Saved best parameters to {filename}")


def print_bestcost(best_cost):
    print(f"Best Cost: {best_cost:.6f}")


def plot_learning_curve(costs):
    plt.plot(costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Learning Curve (Cost vs Iteration)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def evaluate_model(A_L, Y):
    # 예측값 0/1 변환
    y_pred = (A_L >= 0.5).astype(int).reshape(-1)

    # 실제값 1차원 변환
    y_true = Y.reshape(-1)

    # 정확도
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)

    # 정밀도/재현율/F1/Support 모두 포함된 리포트
    report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T
    print("\nClassification Report:\n")
    print(report_df)

    # 혼동 행렬
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()