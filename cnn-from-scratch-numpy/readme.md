# CNN from Scratch (NumPy)

NumPy만을 사용해 Convolutional Neural Network(CNN)의
forward / backward 과정을 직접 구현하며,
합성곱과 역전파가 실제 코드에서 어떻게 동작하는지를 이해하기 위해 진행한 프로젝트입니다.

본 프로젝트의 명확한 목표는 MNIST 분류 문제에서 정확도 1.0에 도달하는 것입니다.

## 1. Project Overview & Motivation

이 프로젝트는 딥러닝 프레임워크를 사용해 모델을 빠르게 만드는 것보다,
CNN이 내부적으로 어떻게 학습되는지를 구조적으로 이해하는 것을 목표로 시작했습니다.

특히 이미지 분류 문제에서 가장 많이 사용되는 CNN 구조를 직접 구현하면서,
합성곱 연산이 입력 이미지의 어떤 특징을 추출하고,
그 결과가 역전파 과정에서 어떻게 다시 입력 방향으로 전달되는지를
코드 수준에서 확인하고 싶었습니다.

이를 위해 PyTorch나 TensorFlow와 같은 프레임워크는 사용하지 않고,
NumPy만을 활용해 CNN의 모든 연산을 직접 구현하였습니다.

이 프로젝트의 목표는 단순히 “잘 돌아가는 코드”를 만드는 것이 아니라,
모델의 내부 구조를 완전히 이해한 상태에서
MNIST 분류 문제를 정확도 1.0까지 수렴시키는 것입니다.

## 2. Model Architecture

본 프로젝트에서 사용한 CNN 모델 구조는 다음과 같습니다.
```
Input (1 × 28 × 28)
 ↓
Conv (16 filters, 3×3, stride=1, pad=1)
ReLU
Conv (16 filters, 3×3, stride=1, pad=1)
ReLU
MaxPool (2×2)

Conv (32 filters, 3×3, stride=1, pad=1)
ReLU
Conv (32 filters, 3×3, stride=1, pad=1)
ReLU
MaxPool (2×2)

Flatten
FC (128)
ReLU
FC (10)
```
설계 의도

VGG 구조에서 영감을 받아 mini VGG를 구현하였습니다.

## 3. Model & Implementation

각 레이어는 forward와 backward 연산을 모두 직접 구현하였으며,
모든 gradient는 Chain Rule을 기반으로 계산됩니다.

구현 과정에서 가장 어려웠던 부분은
convolution backward 단계에서 gradient의 shape을 일관되게 유지하는 것이었습니다.
출력의 한 위치가 입력 이미지의 어느 영역에 영향을 미치는지를 추적하며
gradient를 계산하는 과정에서 많은 차원 불일치 오류를 겪었습니다.

이를 해결하기 위해

작은 입력 예제를 활용한 단계별 검증

print 기반 디버깅

각 연산 단계의 tensor 흐름 직접 추적

과 같은 방법으로 문제를 하나씩 확인하며 구현을 진행했습니다.

## 4. Training Behavior (Loss Curve)

아래 그래프는 학습 과정에서 loss가 어떻게 변화하는지를 확인하기 위해 추가한 결과입니다.

이 그래프의 목적은 성능 비교가 아니라,
직접 구현한 forward와 backward 연산이 정상적으로 동작하며
모델이 실제로 학습되고 있는지를 검증하는 데 있습니다.

loss가 반복 학습을 통해 점진적으로 감소하는 모습을 통해,
gradient가 올바르게 계산되어 가중치 업데이트가 이루어지고 있음을
직관적으로 확인할 수 있었습니다.

![Training Loss Curve](plot/Learning_curve.png)

## 5. Project Structure

본 프로젝트의 전체 폴더 구조는 다음과 같습니다.
```
cnn-from-scratch-numpy/
├── main.py                     # 전체 학습 및 평가 실행 진입점
├── best_params.npz             # 최적 파라미터 저장
│
├── core/
│   ├── train.py                # 학습 루프
│   ├── eval.py                 # 평가 및 정확도 계산
│   ├── loss.py                 # Softmax + Cross Entropy loss
│   └── optim.py                # SGD 기반 파라미터 업데이트
│
├── data/
│   └── mnist_loader.py         # MNIST 데이터 로딩
│
├── layers/
│   ├── conv.py                 # Convolution layer
│   ├── relu.py                 # ReLU activation
│   ├── pool.py                 # Max Pooling layer
│   └── fc.py                   # Fully Connected layer
│
├── model/
│   ├── forward.py              # 설계도 기반 forward propagation
│   ├── backward.py             # cache 기반 backward propagation
│   └── initialize_parameters.py# 파라미터 초기화
│
├── notebooks/
│   ├── experiment.ipynb        # 실험 및 결과 분석
│   └── forward.ipynb           # forward 동작 검증
│
├── plot/
│   ├── Learning_curve.png      # loss 곡선
│   └── Confusion_Matrix.png    # 혼동 행렬
│
└── README.md
```

레이어 구현(layers)과
네트워크 흐름 제어(model, core)를 분리하여 구성함으로써,
각 연산이 전체 학습 과정에서 어떤 역할을 하는지
구조적으로 확인할 수 있도록 설계했습니다.

## 6. What I Learned & Limitations

이 프로젝트를 통해 CNN의 학습 과정이
단순한 함수 호출이 아니라,
여러 미분 연산이 연결된 계산 그래프라는 점을 체감하게 되었습니다.

특히 Chain Rule이 수식으로만 존재하는 개념이 아니라,
실제 코드에서 gradient가 어떻게 전달되는지를 이해하게 되었고,
프레임워크를 사용할 때 발생하는 오류의 원인을
구조적으로 추적할 수 있는 시각을 갖게 되었습니다.

다만 MNIST 데이터셋은 비교적 단순한 문제이기 때문에,
정확도 1.0 달성 자체가 모델의 일반화 성능을 보장하지는 않습니다.
이 프로젝트는 성능 경쟁이 아니라
학습 원리와 구현 정확성을 검증하기 위한 목적을 갖고 있습니다.

## 7. Future Work

향후에는 본 프로젝트에서 구현한 CNN 구조를 기반으로,

Pose Estimation 구조로 확장

Temporal Modeling을 통한 영상 분석

인간 동작 분석 문제로의 적용

과 같은 방향으로 연구를 확장하고자 합니다.
이를 통해 이미지 단위 분류를 넘어,
사람의 움직임을 이해하는 문제로 연결하고 싶습니다.

## 8. How to Run
```
python main.py
```

Environment

## Environment

### Core Implementation
- Python 3.10
- NumPy

### Data Loading & Analysis
- Matplotlib
- scikit-learn
- pandas
