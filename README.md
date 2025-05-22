# Quantum Neural Network for Image Classification

## Project Overview

This project implements a Quantum Neural Network (qNN) to distinguish between diffusion-generated images and real images. By leveraging quantum computing's ability to process high-dimensional data, we aim to improve upon classical CNN-based approaches in detecting AI-generated images.

## Problem Statement

Traditional Convolutional Neural Networks (CNNs) face challenges in distinguishing between diffusion-generated images and real images. This project explores the potential of quantum computing to enhance this classification task by processing high-dimensional image data in quantum space.

## Features

- Quantum Neural Network implementation for image classification
- Comparison with classical CNN approaches
- High-dimensional data processing using quantum circuits
- Support for diffusion-generated and real image datasets

## Technical Stack

- Qiskit for quantum circuit implementation
- PyTorch for classical neural network components
- NumPy for numerical computations
- Matplotlib for visualization

## Project Structure

```
├── data/                  # Dataset directory
├── models/               # Model implementations
│   ├── quantum/         # Quantum neural network components
│   └── classical/       # Classical CNN components
├── utils/               # Utility functions
├── train.py            # Training script
├── evaluate.py         # Evaluation script
└── requirements.txt    # Project dependencies
```

## Getting Started

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare your dataset:

   - Place diffusion-generated images in `data/diffusion/`
   - Place real images in `data/real/`

3. Train the model:

```bash
python train.py
```

4. Evaluate the model:

```bash
python evaluate.py
```

## Results

[Results will be added after implementation]

## Future Work

- Optimization of quantum circuit architecture
- Integration with different quantum hardware
- Extension to other types of AI-generated images

---

# 양자 신경망을 이용한 이미지 구분

## 프로젝트 개요

이 프로젝트는 디퓨전 모델로 생성된 이미지와 실제 이미지를 구분하기 위한 양자 신경망(qNN)을 구현합니다. 양자 컴퓨팅의 고차원 데이터 처리 능력을 활용하여 AI 생성 이미지 탐지에서 기존 CNN 기반 접근 방식의 한계를 개선하고자 합니다.

## 문제 정의

기존의 컨볼루션 신경망(CNN)은 디퓨전 모델로 생성된 이미지와 실제 이미지를 구분하는 데 어려움을 겪고 있습니다. 이 프로젝트는 양자 공간에서 고차원 이미지 데이터를 처리함으로써 이러한 분류 작업의 성능을 향상시키는 양자 컴퓨팅의 잠재력을 탐구합니다.

## 주요 기능

- 이미지 분류를 위한 양자 신경망 구현
- 기존 CNN 접근 방식과의 비교
- 양자 회로를 이용한 고차원 데이터 처리
- 디퓨전 생성 이미지와 실제 이미지 데이터셋 지원

## 기술 스택

- 양자 회로 구현을 위한 Qiskit
- 고전 신경망 컴포넌트를 위한 PyTorch
- 수치 계산을 위한 NumPy
- 시각화를 위한 Matplotlib

## 프로젝트 구조

```
├── data/                  # 데이터셋 디렉토리
├── models/               # 모델 구현
│   ├── quantum/         # 양자 신경망 컴포넌트
│   └── classical/       # 고전 CNN 컴포넌트
├── utils/               # 유틸리티 함수
├── train.py            # 학습 스크립트
├── evaluate.py         # 평가 스크립트
└── requirements.txt    # 프로젝트 의존성
```

## 시작하기

1. 의존성 설치:

```bash
pip install -r requirements.txt
```

2. 데이터셋 준비:

   - 디퓨전 생성 이미지를 `data/diffusion/`에 배치
   - 실제 이미지를 `data/real/`에 배치

3. 모델 학습:

```bash
python train.py
```

4. 모델 평가:

```bash
python evaluate.py
```

## 결과

[구현 후 결과 추가 예정]

## 향후 계획

- 양자 회로 아키텍처 최적화
- 다양한 양자 하드웨어와의 통합
- 다른 유형의 AI 생성 이미지로의 확장
