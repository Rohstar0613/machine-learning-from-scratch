# DNN from Scratch with NumPy

## Overview

NumPy만을 사용하여 Deep Neural Network(DNN)를 처음부터 직접 구현한 프로젝트입니다.

순전파, 역전파, 파라미터 업데이트 전 과정을 직접 구현하며 딥러닝의 내부 동작 원리를 이해하는 것을 목표로 했습니다.

본 프로젝트에서는 train set 정확도만 계산하였으며, dev / test set 평가는 의도적으로 수행하지 않았습니다.

---

## 핵심 기능 및 학습 설정 (Key Features & Training Setup)

- **프레임워크** : NumPy만 사용 (GPU 미사용, 딥러닝 프레임워크 미사용)
- **활성화 함수** : ReLU (은닉층), Sigmoid (출력층)
- **손실 함수** : Binary Cross-Entropy
- **최적화 방법** : Gradient Descent
- **초기화 방법** : He initialization
- **데이터셋** : Forest Cover Type Dataset (이진 분류)
- **입력 데이터 크기** : (54, 495,141)
- **학습 방식** : CPU 환경에서 full-batch gradient descent

본 프로젝트는 성능을 최대한 끌어올리는 것이 목적이 아니라,
from-scratch로 구현한 신경망이 정상적으로 학습되는지와
학습 흐름(순전파–역전파–파라미터 업데이트)이 올바르게 동작하는지를
검증하는 것을 목표로 한다.

---

## Sanity Check 기준

- Train 데이터만 사용
- Train Accuracy ≥ 80%
- 목적: 모델 성능 평가가 아닌,
    
    구현 정확성과 학습 파이프라인의 정상 동작 여부 검증
    

대규모 데이터셋을 대상으로 NumPy 기반 full-batch 학습을 수행했기 때문에, 학습 시간은 비교적 오래 소요되었다.

이는 성능 비교를 위한 선택이 아니라, 구현 검증 목적에 따른 의도적인 설정이다.

---

## Results

- **Final Train Accuracy**: **77.65%**
- **Final Training Loss**: ~0.52
- **Network Structure**: [54, 32, 16, 1]

학습 과정에서 loss는 초기 1.04에서 약 0.52까지
단조 감소하며 안정적으로 수렴하는 양상을 보였다.

학습을 더 진행하면 추가적인 수렴이 가능할 것으로 판단되지만,
본 프로젝트는 성능 최적화가 목적이 아니기 때문에
해당 시점에서 학습을 종료하였다.
본 결과는 NumPy 기반 from-scratch DNN 구현이 정상적으로 학습되고
순전파–역전파–파라미터 업데이트 흐름이 올바르게 동작함을
검증하기 위한 sanity check 결과이다.

---

## Next Steps

- 다양한 정규화 기법 구현
- CNN from scratch 구현

---

## Blog

구현 과정과 시행착오, 설계 의도에 대한 상세한 설명은 아래 글에 정리되어 있습니다.

👉 https://rohstar.tistory.com/entry/Numpy%EB%A7%8C%EC%9C%BC%EB%A1%9C-DNN-%EA%B5%AC%ED%98%84%ED%95%B4%EB%B3%B4%EA%B8%B0

---

## Reference

- Andrew Ng, *Neural Networks and Deep Learning*
- Forest Cover Type Dataset

---

## 📁 Repository Structure

```
numpy_dnn_from_scratch/
├── notebooks/
│   ├── best_params.npz
│   └── experiment.ipynb
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── main.py
└── requirements.txt

```
