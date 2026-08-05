---
title: "ResNet: Deep Residual Learning for Image Recognition"
date: 2026-04-24T10:00:00+09:00
draft: false
categories: ["Papers", "Computer Vision", "Deep Learning"]
tags: ["ResNet", "Residual Learning", "Skip Connection", "Image Classification", "CNN", "Microsoft Research", "ILSVRC 2015"]
year: 2015
references: []
---

## 💡 한 줄 요약
네트워크가 깊어질 때 발생하는 최적화 저하(Degradation Problem) 현상을 **Identity Shortcut Connection** 기반 잔차 학습 $\mathcal{F}(\mathbf{x}) + \mathbf{x}$ 구조로 완벽히 해소하여 152층 이상의 극단적 심층 신경망을 안정적으로 훈련시키고 ILSVRC 2015 1위를 달성했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)
- **발행년도**: 2015 (arXiv:1512.03385, CVPR 2016)
- **주요 기여점**:
  1. **잔차 학습 (Residual Learning) 정립**: 레이어가 매핑 함수 $\mathcal{H}(\mathbf{x})$를 직접 매핑하지 않고 잔차 함수 $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$만 모니터링하도록 수식 재정식화.
  2. **Identity Shortcut Connection**: 연산량 및 추가 파라미터가 0인 스킵 커넥션으로 역전파 기울기가 그라디언트 소실(Vanishing Gradient) 없이 전 계층으로 직통 전달.
  3. **Bottleneck 구조 제안**: $1 \times 1 \to 3 \times 3 \to 1 \times 1$ 컨볼루션 블록으로 연산 차원을 축소 후 복원하여 50층~152층 깊이의 컴퓨팅 효율 극대화.
  4. **ILSVRC & COCO 2015 전 부문 석권**: Top-5 오류율 3.57% 달성으로 분류, 탐지, 세그멘테이션 분야 백본의 글로벌 표준 등극.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **Plain CNN (VGG, GoogLeNet)**: 레이어를 20층 이상 쌓을 경우 과적합이 아님에도 불구하고 훈련 오차(Training Error)와 검증 오차가 동시에 폭증하는 Degradation 현상 발생.
2. **Highway Networks**: Shortcut에 데이터 의존적 Gating 파라미터를 추가하여 게이트가 닫힐 때 잔차 학습 손실.
3. **ResNet**: 아무런 파라미터가 없는 순수 Identity Shortcut을 유지하여 수백 레이어 단조 성능 향상.

---

## 📑 목차
- Chapter 1: Basic Residual Block & Identity Shortcut 수식
- Chapter 2: Projection Shortcut ($1 \times 1$ Conv) 수식
- Chapter 3: Bottleneck Block ($1 \times 1 \to 3 \times 3 \to 1 \times 1$) 연산
- Chapter 4: Global Average Pooling (GAP) & 백본 신경망
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: Basic Residual Block & Identity Shortcut 수식

### 1. 요약
목표 매핑 $\mathcal{H}(\mathbf{x})$를 $\mathcal{F}(\mathbf{x}) + \mathbf{x}$로 분해하여 잔차 $\mathcal{F}(\mathbf{x})$만 학습하도록 구성합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

$$\mathcal{F}(\mathbf{x}, \{W_i\}) = W_2 \cdot \text{ReLU}\big( \text{BN}(W_1 \mathbf{x}) \big)$$

- **$\mathbf{x}, \mathbf{y}$**: 잔차 블록의 입출력 텐서
- **$\mathcal{F}$**: 2개의 $3 \times 3$ Conv + BN + ReLU 레이어가 학습하는 잔차 함수

```python
import torch
import torch.nn as nn

class BasicResidualBlock(nn.Module):
    """
    ResNet-18 / ResNet-34용 기본 잔차 블록 (Basic Block)
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x # Identity Shortcut Connection
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # F(x) + x
        out += identity
        return self.relu(out)

# --- 사용 예시 ---
block = BasicResidualBlock(in_channels=64, out_channels=64)
x_in = torch.randn(2, 64, 56, 56)
print("Basic Residual Block 출력 Shape:", block(x_in).shape)
```

---

## 🛠️ Chapter 2: Projection Shortcut ($1 \times 1$ Conv) 수식

### 1. 요약
스트라이드 적용이나 채널 수 변경으로 인해 입력 $\mathbf{x}$와 잔차 $\mathcal{F}(\mathbf{x})$의 차원이 다를 때, $1 \times 1$ Conv $W_s$를 곱해 차원을 맞춥니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + W_s \mathbf{x}$$

```python
import torch
import torch.nn as nn

class ProjectionResidualBlock(nn.Module):
    """
    차원/해상도가 변경될 때 Projection Shortcut (W_s)을 포함하는 잔차 블록
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # W_s 1x1 Conv Projection Shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x) # W_s * x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        return self.relu(out)

# --- 사용 예시 ---
proj_block = ProjectionResidualBlock(in_channels=64, out_channels=128, stride=2)
x_in = torch.randn(2, 64, 56, 56)
print("Projection Residual Block 출력 Shape:", proj_block(x_in).shape)
```

---

## 🛠️ Chapter 3: Bottleneck Block ($1 \times 1 \to 3 \times 3 \to 1 \times 1$) 연산

### 1. 요약
ResNet-50/101/152 등 깊은 아키텍처의 연산 비용을 줄이기 위해 $1 \times 1$ Conv로 채널 축소 후 $3 \times 3$ Conv 연산을 수행하고 다시 $1 \times 1$ Conv로 4배 확충합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{F}_{\text{bottleneck}}(\mathbf{x}) = W_3 \cdot \text{ReLU}\Big( \text{BN}\big( W_2 \cdot \text{ReLU}( \text{BN}(W_1 \mathbf{x}) ) \big) \Big)$$

$$\mathbf{y} = \text{ReLU}\big( \mathcal{F}_{\text{bottleneck}}(\mathbf{x}) + W_s \mathbf{x} \big)$$

```python
import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    """
    ResNet-50 / 101 / 152용 병목 (Bottleneck) 잔차 블록
    """
    expansion: int = 4

    def __init__(self, in_channels: int, base_channels: int, stride: int = 1):
        super().__init__()
        out_channels = base_channels * self.expansion
        
        # 1x1 Conv (차원 축소)
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # 3x3 Conv (특징 추출)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels)
        
        # 1x1 Conv (차원 4배 복원)
        self.conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += identity
        return self.relu(out)

# --- 사용 예시 ---
bottleneck = BottleneckBlock(in_channels=256, base_channels=64, stride=1)
x_in = torch.randn(2, 256, 56, 56)
print("Bottleneck Block 출력 Shape:", bottleneck(x_in).shape)
```

---

## 🛠️ Chapter 4: Global Average Pooling (GAP) & 백본 신경망

### 1. 요약
공간 해상도 $(H \times W)$ 전체를 평균으로 압축하는 Global Average Pooling을 통해 파라미터를 획기적으로 줄이고 분류기로 전파합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{y}_{\text{gap}} = \frac{1}{H \times W} \sum_{h=1}^H \sum_{w=1}^W \mathbf{f}(h, w) \in \mathbb{R}^C$$

```python
import torch
import torch.nn as nn

class ResNet50BackboneClassifier(nn.Module):
    """
    ResNet-50 구조의 GAP 및 선형 분류 헤드 파이프라인
    """
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, last_stage_feature: torch.Tensor) -> torch.Tensor:
        """
        last_stage_feature: (B, 2048, H, W)
        """
        pooled = self.gap(last_stage_feature) # (B, 2048, 1, 1)
        flat = torch.flatten(pooled, 1)        # (B, 2048)
        logits = self.fc(flat)                 # (B, Num_classes)
        return logits

# --- 사용 예시 ---
backbone_head = ResNet50BackboneClassifier(num_classes=1000)
feat_in = torch.randn(2, 2048, 7, 7)
print("ResNet GAP 분류 Logits Shape:", backbone_head(feat_in).shape)
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. ImageNet-1K 검증 세트 리더보드 성능 비교

| 알고리즘 (Method) | 신경망 깊이 (Depth) | Top-1 오류율 ↓ | Top-5 오류율 ↓ | 특징 |
|---|---|---|---|---|
| **VGG-16** | 16-Layers | 28.07% | 9.33% | 파라미터 138M (비대함) |
| **Plain-34** | 34-Layers | 28.54% | 10.02% | Degradation 최적화 실패 |
| **ResNet-34** | 34-Layers | 24.52% | 7.46% | 34층 정상 최적화 성공 |
| **ResNet-152 (Ours)** | **152-Layers** | **21.43%** | **5.71% (SOTA)** | **ILSVRC 2015 1위 석권** |

- **결과**: 잔차 학습 구조 도입으로 152층까지 네트워크 깊이를 늘려도 **Top-5 오차율 3.57%**로 압도적 SOTA 갱신.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
ResNet은 잔차 학습(Residual Learning)이라는 혁신적 아이디어로 딥러닝 역사상 가장 위대한 컴퓨터 비전 백본 아키텍처를 정립했습니다.

### 2. 한계점 및 아쉬운 점
- 1202층 실험에서 보듯 과도하게 깊은 네트워크는 작은 데이터셋에서 과적합될 수 있어, "깊이가 항상 유리하다"는 결론은 데이터 규모에 의존적이다.
- Identity mapping이 왜 최적화에 유리한지에 대한 이론적 설명은 실험적 관찰(잔차 반응이 작다)에 그치며, 이후 연구(예: Identity Mappings in Deep Residual Networks)에서 더 정교하게 분석됨.
- Bottleneck 설계의 채널 축소·복원 비율 등 세부 하이퍼파라미터 선택 근거가 충분히 설명되지 않은 점은 아쉽다.

---

## 참고 자료
- [Microsoft ResNet 공식 GitHub 저장소](https://github.com/KaimingHe/deep-residual-networks)
- [CVPR 2016 논문 (arXiv:1512.03385)](https://arxiv.org/abs/1512.03385)

*관련 논문: [Attention Is All You Need](/posts/papers/attention-is-all-you-need/), [ViT](/posts/papers/vit-an-image-is-worth-16x16-words/), [DETR](/posts/papers/detr-end-to-end-object-detection-with-transformers/)*
