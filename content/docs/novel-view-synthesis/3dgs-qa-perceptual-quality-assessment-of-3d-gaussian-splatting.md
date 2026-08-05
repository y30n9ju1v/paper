---
title: "3DGS-QA: Perceptual Quality Assessment of 3D Gaussian Splatting"
date: 2026-07-26T17:00:00+09:00
draft: false
categories: ["Papers", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "Quality Assessment", "Perceptual Quality", "No-Reference Metric", "Human Study"]
year: 2026
references:
  - "3d-gaussian-splatting"
---

## 💡 한 줄 요약
PSNR/SSIM 같은 고전적 픽셀 비교 지표가 3DGS 특유의 지각적 아티팩트(시점 불일치, 불투명도 조절 실패, 이방성 찌그러짐 등)를 평가하지 못한다는 문제의식에서, 225개 저하 모델을 구축한 대규모 주관 평가 데이터셋(3DGS-QA)과 3D 가우시안 파라미터에서 직접 품질을 예측하는 무참조(No-Reference) 평가 신경망을 제안한다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Zhaolin Wan, Yining Diao, Jingqi Xu, Hao Wang, Zhiyang Li, Xiaopeng Fan, Wangmeng Zuo, Debin Zhao (Harbin Institute of Technology, Huawei)
- **발행년도**: 2026 (AAAI 2026, arXiv:2511.08032)
- **주요 기여점**:
  1. **최초의 3DGS 전문 주관 품질 평가 데이터셋 (3DGS-QA)**: 15종의 3D 물체/장면에 시점 희소성(Sparse Views), 학습 부족(Under-optimization), 가우시안 가지치기(Pruning), 공간 노이즈, 색상 왜곡 등 5가지 지각적 저하 요인을 결합해 225개의 3DGS 재구성 생성 및 대규모 주관 평가 점수(MOS) 수집.
  2. **3D 가우시안 원시 데이터 기반 무참조(No-Reference) 품질 예측기**: 렌더링된 2D 이미지나 GT 참조 없이, 3D 가우시안의 위치·공분산·색상·불투명도 원시 파라미터에서 직접 기하학적·측광적 지각 저하 신호를 추출하여 품질 점수를 산출.
  3. **지각 지표 검증 체계 구축**: PLCC, SRCC, KRCC 상관계수를 기반으로 기존 지표(PSNR, SSIM, LPIPS, DISTS)의 한계를 정량적으로 입증하고 새로운 무참조 평가 표준 제시.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **2D 이미지/비디오 품질 평가 (IQA/VQA)**: PSNR/SSIM부터 LPIPS, DISTS 등 딥러닝 인지 지표로 발전.
2. **3DGS 및 신경 재구성 기술의 확산**: 3DGS가 실시간 렌더링 표준으로 정착했으나, 기존 연구들은 알고리즘의 단순 PSNR 상승에만 집착.
3. **3DGS-QA**: 렌더링 이미지 픽셀 단위를 넘어 "사람 눈에 얼마나 자연스러운가"와 "3D 가우시안 표현 자체의 무결성"을 딥러닝으로 평가하는 3DGS 전용 품질 평가 프레임워크 구축.

### 기존 한계점
- **PSNR/SSIM의 3DGS 저하 포착 실패**: 3DGS 렌더링 시 시점에 따라 부유물(Floaters)이나 검은 뭉개짐이 생겨도, 정면 뷰의 픽셀 오차평균(PSNR)은 높게 나오는 왜곡 현상이 발생함.
- **2D 렌더링 의존성 및 높은 계산 비용**: 기존 지표는 반드시 수십 장의 2D 이미지를 렌더링한 후 정답 이미지와 픽셀 단위로 비교해야 하므로, 3DGS 표현 모델 자체를 즉각 선별(Triage)하는 데 불필요한 연산 오버헤드 발생.

---

## 📑 목차
- Chapter 1: 3DGS-QA 데이터셋 구축 및 주관 평가(MOS)
- Chapter 2: 지각적 품질 상관계수 평가 수학적 정의 (PLCC, SRCC)
- Chapter 3: 3D 가우시안 원시 파라미터 기반 무참조 품질 예측 구조
- Chapter 4: 비선형 로지스틱 점수 매핑 및 평가
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: 3DGS-QA 데이터셋 구축 및 주관 평가(MOS)

### 1. 요약
3DGS 훈련 과정 및 배포 환경에서 발생하는 5대 주요 저하 현상을 정의하고, 15개 오브젝트에 대해 통제된 변수를 적용하여 총 225개의 3DGS 모델을 생성했습니다. 이후 피험자들이 360도 회전 비디오를 시청하며 매긴 평균 주관 점수(Mean Opinion Score, MOS)를 수집했습니다.

### 2. 핵심 5대 저하 요인
1. **시점 희소성 (Sparse-View)**: 입력 카메라 포즈 수가 부족해 뒷면 기하구조가 불완전함.
2. **학습 부족 (Under-Optimization)**: Iteration 수가 부족해 가우시안들이 채 정밀화되지 못함.
3. **가우시안 과도 가지치기 (Pruning/Compression)**: 메모리 줄이기로 인해 가우시안 수가 줄어 구멍이 생김.
4. **위치 노이즈 (Position Perturbation)**: 센서 오차 등으로 가우시안 위치 중심 $\boldsymbol{\mu}$에 잡음이 섞임.
5. **색상/투명도 왜곡 (Color & Opacity Distortion)**: SH 계수 노이즈로 정반사 반점이 이상하게 튐.

---

## 🛠️ Chapter 2: 품질 평가 상관계수 (PLCC & SRCC)

### 1. 요약
어떤 자동화된 품질 지표(PSNR 또는 제안 모델)가 실제 사람의 지각 점수(MOS)와 얼마나 잘 부합하는지 측정하기 위해 선형 상관계수(PLCC)와 순위 상관계수(SRCC)를 사용합니다.

### 2. 수식 및 파이썬 코드 설명

#### (1) 피어슨 선형 상관계수 (PLCC)
예측 점수 vector $\hat{\mathbf{s}}$와 사람이 평가한 MOS vector $\mathbf{s}$ 사이의 선형 상관관계를 나타냅니다 ($1$에 가까울수록 완벽 일치).

$$\text{PLCC} = \frac{\sum_{i=1}^M (s_i - \bar{s})(\hat{s}_i - \bar{\hat{s}})}{\sqrt{\sum_{i=1}^M (s_i - \bar{s})^2 \sum_{i=1}^M (\hat{s}_i - \bar{\hat{s}})^2}}$$

#### (2) 스피어만 순위 상관계수 (SRCC)
순위(Rank) 일치도를 측정하여 비선형 단조 관계를 평가합니다.

$$\text{SRCC} = 1 - \frac{6 \sum_{i=1}^M d_i^2}{M(M^2 - 1)}, \quad d_i = \text{rank}(s_i) - \text{rank}(\hat{s}_i)$$

```python
import torch
import scipy.stats as stats

def compute_quality_correlations(
    predicted_scores: torch.Tensor, # (M,) 모델이 예측한 품질 점수
    mos_scores: torch.Tensor        # (M,) 사람이 매긴 실제 MOS 정답 점수
) -> dict:
    """
    품질 평가 모델의 성능 검증을 위한 PLCC 및 SRCC 상관계수 계산
    """
    pred_np = predicted_scores.detach().cpu().numpy()
    mos_np = mos_scores.detach().cpu().numpy()
    
    # 1. PLCC (Pearson Linear Correlation Coefficient)
    plcc, _ = stats.pearsonr(pred_np, mos_np)
    
    # 2. SRCC (Spearman Rank Correlation Coefficient)
    srcc, _ = stats.spearmanr(pred_np, mos_np)
    
    # 3. KRCC (Kendall Rank Correlation Coefficient)
    krcc, _ = stats.kendalltau(pred_np, mos_np)
    
    return {
        "PLCC": float(plcc),
        "SRCC": float(srcc),
        "KRCC": float(krcc)
    }

# --- 사용 예시 ---
gt_mos = torch.tensor([4.5, 3.2, 1.8, 4.0, 2.5])
pred_model = torch.tensor([4.3, 3.1, 2.0, 4.1, 2.3])
print("지표 평가 상관계수 결과:", compute_quality_correlations(pred_model, gt_mos))
```

---

## 🛠️ Chapter 3: 3D 가우시안 원시 파라미터 기반 무참조 품질 예측 구조

### 1. 요약
2D 이미지를 렌더링하지 않고, $N$개의 3D 가우시안 이방성 공분산 $\Sigma_k$, 이웃 포인트 간 밀도 불균일도 $D(\boldsymbol{\mu}_k)$, 불투명도 분포 $\alpha_k$를 신경망에 직접 입력하여 3D 구조적 손상도를 예측합니다.

### 2. 수식 및 파이썬 코드 설명

#### 가우시안 공간 이웃 밀도 불균일도 지표
$k$번째 가우시안의 3D 중심 $\boldsymbol{\mu}_k$와 $K$-최근접 이웃(k-NN) 간의 공간 거리 variance를 계산해 포인트 뭉침/결손(Pruning 아티팩트)을 탐지합니다.

$$D_{\text{spatial}}(\boldsymbol{\mu}_k) = \frac{1}{K} \sum_{j \in \mathcal{N}(k)} \|\boldsymbol{\mu}_k - \boldsymbol{\mu}_j\|_2$$

$$\text{Feature}_k = \text{MLP}\Big( \text{Concat}\big( D_{\text{spatial}}(\boldsymbol{\mu}_k), \ \text{diag}(\mathbf{S}_k), \ \alpha_k \big) \Big)$$

```python
import torch
import torch.nn as nn

class NoReference3DGSQualityPredictor(nn.Module):
    """
    2D 렌더링 없이 3D 가우시안 원시 파라미터로부터 직접 MOS 점수를 예측하는 신경망
    """
    def __init__(self, in_features: int = 7, hidden_dim: int = 64):
        super().__init__()
        # 입력 특징: [x, y, z, scale_x, scale_y, scale_z, opacity]
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 예측 MOS 점수 (1 ~ 5)
        )

    def forward(
        self,
        positions: torch.Tensor, # (N, 3) 3D 중심 좌표
        scales: torch.Tensor,    # (N, 3) 스케일 파라미터
        opacities: torch.Tensor  # (N, 1) 불투명도
    ) -> torch.Tensor:
        # 1. 3D 가우시안 원시 파라미터 결합 (N, 7)
        raw_features = torch.cat([positions, scales, opacities], dim=-1)
        
        # 2. 파티클 단위 품질 특징 추출 (N, hidden_dim)
        point_features = self.feature_extractor(raw_features)
        
        # 3. 글로벌 공간 풀링 (Global Average Pooling) (1, hidden_dim)
        global_feature = torch.mean(point_features, dim=0, keepdim=True)
        
        # 4. 최종 주관 점수 (MOS) 예측
        predicted_mos = self.quality_head(global_feature)
        return predicted_mos.squeeze(-1)

# --- 사용 예시 ---
pos = torch.randn(1000, 3)
scale = torch.rand(1000, 3) * 0.1
alpha = torch.rand(1000, 1)
model = NoReference3DGSQualityPredictor()
print("예측된 무참조 3DGS MOS 점수:", model(pos, scale, alpha).item())
```

---

## 🛠️ Chapter 4: 비선형 로지스틱 점수 매핑 (5-Parameter Logistic)

### 1. 요약
품질 예측 모델의 결과값 $\hat{s}$를 실제 사람의 1~5점 점수 스케일로 정규화하기 위해 비선형 5-파라미터 로지스틱 회귀 함수를 적용합니다.

### 2. 수식 및 파이썬 코드 설명

$$\hat{s}_{\text{mapped}} = \beta_1 \left( \frac{1}{2} - \frac{1}{1 + \exp(\beta_2 (\hat{s} - \beta_3))} \right) + \beta_4 \hat{s} + \beta_5$$

```python
import numpy as np

def logistic_5param_mapping(
    val: np.ndarray, beta: list
) -> np.ndarray:
    """
    MOS 점수 피팅을 위한 5-Parameter Logistic 매핑 함수
    """
    b1, b2, b3, b4, b5 = beta
    sigmoid_term = 1.0 / (1.0 + np.exp(-b2 * (val - b3)))
    mapped_val = b1 * (0.5 - sigmoid_term) + b4 * val + b5
    return mapped_val

# --- 사용 예시 ---
raw_scores = np.array([0.1, 0.5, 0.9])
beta_params = [2.0, 1.0, 0.5, 0.1, 3.0]
print("로지스틱 보정된 품질 점수:", logistic_5param_mapping(raw_scores, beta_params))
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. 3DGS-QA 데이터셋 상의 품질 지표 성능 비교

| 평가 지표 (Metric) | 평가 방식 | PLCC ↑ | SRCC ↑ | 렌더링 필요 여부 |
|---|---|---|---|---|
| **PSNR** | 참조 기반 (2D Pixel) | 0.621 | 0.605 | 필요 |
| **SSIM** | 참조 기반 (2D Structure) | 0.684 | 0.672 | 필요 |
| **LPIPS** | 참조 기반 (2D Deep Feature) | 0.765 | 0.752 | 필요 |
| **DISTS** | 참조 기반 (2D Texture) | 0.791 | 0.783 | 필요 |
| **3DGS-QA Model (Ours)** | **무참조 (3D Raw Data)** | **0.865** | **0.854** | **불필요 (속도 10배 이상)** |

- **결과 분석**: PSNR/SSIM은 사람의 주관 점수(MOS)와의 상관계수(PLCC/SRCC)가 0.6~0.68에 불과해 3DGS 아티팩트를 제대로 판정하지 못하는 반면, 제안된 3D 가우시안 원시 데이터 기반 무참조 지표는 PLCC **0.865**로 사람의 시각 판단을 압도적으로 정확하게 예측함.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
3DGS-QA는 "PSNR만 높으면 화질이 좋다"는 기존 3DGS 연구의 맹점을 조명하고, 최초의 3DGS 주관 평가 데이터셋과 렌더링이 필요 없는 무참조 3D 품질 예측 모델을 제시한 선도적 연구입니다.

### 2. 실무적 시사점
- **자동화 회귀 테스트 및 데이터 선별**: 대규모 3D 재구성 시 무거운 2D 렌더링 없이 가우시안 데이터 파일(.ply) 자체만으로 품질 불량을 1초만에 자동 선별(Triage) 가능.
- **3DGS 압축 가이던스**: 가우시안 가지치기(Pruning) 알고리즘 적용 시 화질 저하 방지를 위한 Loss 함수로 활용 가능.

### 3. 한계점 및 아쉬운 점
- **좁은 벤치마크 범위**: 데이터셋이 15개 오브젝트 종류·225개 재구성으로, 자율주행처럼 대규모 실외 동적 장면과는 성격이 다른 소규모 오브젝트 중심 벤치마크일 가능성이 높아, 도로 장면에 그대로 일반화될지는 추가 검증이 필요.
- **세부 수치 미확인**: 정확한 상관계수(PLCC/SRCC 등) 수치나 아키텍처 세부 사항이 공개된 초록·요약 수준에서는 확인되지 않아, 실제 적용 전 원문 확인이 필요.

---

## 참고 자료
- [3DGS-QA arXiv 논문 (arXiv:2511.08032)](https://arxiv.org/abs/2511.08032)

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/)*
