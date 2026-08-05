---
title: "GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction"
date: 2026-04-24T13:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "3D Occupancy"]
tags: ["3D Gaussian Splatting", "Occupancy Prediction", "World Model", "Temporal Modeling", "Streaming", "nuScenes"]
year: 2024
references:
  - "3d-gaussian-splatting"
  - "occ3d-large-scale-3d-occupancy-prediction-benchmark"
  - "bevformer"
---

## 💡 한 줄 요약
3D Semantic Occupancy 예측을 이전 프레임의 3D Gaussian 상태 $\mathbf{z}^{T-1}$와 현재 센서 관측 $\mathbf{x}^T$에 기반한 **4D Streaming World Model ($\mathbf{z}^T = \mathbf{w}(\mathbf{z}^{T-1}, \mathbf{x}^T)$)**로 재정의하고, 자차 정합·동적 객체 이동·신규 영역 완성의 3단계 물리적 진화 모델링으로 추가 연산 오버헤드 없이 nuScenes mIoU +2.4% 향상을 달성했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Sicheng Zuo, Wenzhao Zheng, Yuanhui Huang, Jie Zhou, Jiwen Lu (Tsinghua University)
- **발행년도**: 2024 (arXiv 2412.10373)
- **주요 기여점**:
  1. **3D Gaussian 기반 4D Streaming World Model**: 독립적 프레임 인코딩 후 융합 방식(Existing Temporal Fusion) 대신, 3D Gaussian 속성의 시간적 전파(Recurrence)로 이전 시퀀스 기억을 직접 진화시킴.
  2. **명시적 3단계 장면 진화 분해 (Explicit Scene Evolution)**: 드라이빙 씬의 시공간 변화를 (1) 자차 이동 정합 (Ego Alignment), (2) 동적 객체 국소 이동 (Dynamic Motion), (3) 신규 영역 공간 완성 (New Area Completion)으로 물리적 분해.
  3. **통합 퍼셉션-모션 아키텍처 (Unified Evolution Layer)**: 역사적 Gaussian은 Motion 모드로 위치만 업데이트하고, 신규 Gaussian은 Perception 모드로 전 속성을 예측하여 계산 자원 극대화.
  4. **GS-to-Occ 이산화 알고리즘**: 3D Gaussian 표현을 3D Semantic Occupancy 복셀 그리드로 수렴 변환.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **단일 프레임 3D Occupancy (MonoScene, TPVFormer, SurroundOcc)**: 매 순간 정적 2D 이미지로 3D 복셀을 예측하므로 렌즈 가림(Occlusion)이나 동적 객체 속도 추정에 취약.
2. **복셀 기반 시간 융합 (BEVFormer, GaussianFormer-T)**: 과거 $T$개 프레임의 BEV/Voxel Feature Map을 저장하고 Warping하여 융합하나, 메모리 오버헤드가 $\mathcal{O}(T)$로 폭발.
3. **GaussianWorld**: 3D Gaussian 파라미터만 전달받아 World Model 레지스터로 갱신하여 1개 과거 프레임 지연시간만으로 극도의 실시간성 유지.

---

## 📑 목차
- Chapter 1: 3D Gaussian 속성 정의 및 World Model 전이 수식
- Chapter 2: 명시적 장면 진화 (자차 정합, 동적 이동, 신규 영역 완성)
- Chapter 3: 3D Gaussian $\to$ 3D Occupancy 복셀 격자 변환 (GS-to-Occ)
- Chapter 4: 스트리밍 학습 전략 (Sequence Progressive Dropout)
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: 3D Gaussian 속성 및 World Model 상태 전이

### 1. 요약
각 3D Gaussian $g_i$는 위치 $\mathbf{p}_i$, 크기 $\mathbf{s}_i$, 회전 사원수 $\mathbf{r}_i$, 시맨틱 클래스 확률 $\mathbf{c}_i$, 시간적 특징 $\mathbf{f}_i$의 5개 속성으로 정의되며, World Model $\mathbf{w}$는 이전 상태 $\mathbf{z}^{T-1}$와 현재 카메라 입력 $\mathbf{x}^T$를 동시 조건화합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{g}_i = \{\mathbf{p}_i, \mathbf{s}_i, \mathbf{r}_i, \mathbf{c}_i, \mathbf{f}_i\}, \quad \mathbf{z}^{T-1} = \{\mathbf{g}_i^{T-1}\}_{i=1}^N$$

$$\mathbf{z}^T = \mathbf{w}\left( \mathbf{z}^{T-1}, \ \mathbf{x}^T \right)$$

$$\mathbf{y}^T = \mathbf{h}(\mathbf{z}^T) \quad (\mathbf{h}: \text{GS-to-Occ Head})$$

```python
import torch
import torch.nn as nn

class GaussianWorldState:
    """
    GaussianWorld의 3D Gaussian 명시적 상태 벡터 클래스
    """
    def __init__(self, p: torch.Tensor, s: torch.Tensor, r: torch.Tensor, c: torch.Tensor, f: torch.Tensor):
        self.p = p # (N, 3) 3D Position
        self.s = s # (N, 3) Scale
        self.r = r # (N, 4) Quaternion Rotation
        self.c = c # (N, C_classes) Semantic Probabilities
        self.f = f # (N, C_feat) Temporal Feature Vector

# --- 사용 예시 ---
p_in = torch.randn(1000, 3)
s_in = torch.ones(1000, 3) * 0.5
r_in = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(1000, 1)
c_in = torch.softmax(torch.randn(1000, 18), dim=-1)
f_in = torch.randn(1000, 128)
state_t1 = GaussianWorldState(p_in, s_in, r_in, c_in, f_in)
print("이전 프레임 3D Gaussian 상태 객체 생성 완료 (N=1000)")
```

---

## 🛠️ Chapter 2: 명시적 장면 진화 (Scene Evolution)

### 1. 요약
장면 변화를 자차 이동 정합($\mathbf{g}_A$), 동적 객체 이동($\mathbf{g}_M$), 신규 관측 영역 완성($\mathbf{g}_C$)의 3단계로 분해하여 처리합니다.

### 2. 수식 및 파이썬 코드 설명

#### (1) 자차 이동 정합 (Ego Motion Alignment)
$$\mathbf{p}_A^{T} = \mathbf{R}_{ego} \mathbf{p}^{T-1} + \mathbf{t}_{ego}, \quad \mathbf{r}_A^{T} = \mathbf{R}_{ego} \cdot \mathbf{r}^{T-1}$$

#### (2) 동적 객체 이동 (Dynamic Motion Update)
$$\mathbf{p}_M^T = \mathbf{p}_A^T + \Delta\mathbf{p} \cdot \mathbb{I}(\mathbf{g}_A^T \in \{\mathbf{g}_D\}), \quad \Delta\mathbf{p} = \text{MLP}(\text{CrossAttn}(\mathbf{f}_A, \mathbf{x}^T))$$

```python
import torch

def align_and_update_dynamic_gaussians(
    state: GaussianWorldState, # 이전 프레임 Gaussian 상태
    R_ego: torch.Tensor,       # (3, 3) 자차 회전 행렬
    t_ego: torch.Tensor,       # (3,) 자차 이동 벡터
    is_dynamic_mask: torch.Tensor # (N,) 동적 Gaussian 필터링 마스크
) -> GaussianWorldState:
    """
    1단계: 자차 이동 Rigid Transformation + 2단계: 동적 객체 변위 적용
    """
    # 1. 자차 정합: p_aligned = R_ego @ p + t_ego
    p_aligned = (R_ego @ state.p.T).T + t_ego
    
    # 2. 동적 객체 오프셋 예측 (가상 오프셋 예시)
    dynamic_offset = torch.randn_like(p_aligned) * 0.2
    
    # 동적 마스크 적용 대상만 오프셋 이동
    p_final = p_aligned + dynamic_offset * is_dynamic_mask.unsqueeze(-1)
    
    return GaussianWorldState(p_final, state.s, state.r, state.c, state.f)

# --- 사용 예시 ---
R_d, t_d = torch.eye(3), torch.tensor([1.5, 0.0, 0.0])
mask_d = (c_in.argmax(dim=-1) == 1) # 1번 클래스가 차량(Vehicle)이라고 가정
updated_state = align_and_update_dynamic_gaussians(state_t1, R_d, t_d, mask_d)
print("정합 및 동적 업데이트 완료된 3D 위치 Shape:", updated_state.p.shape)
```

---

## 🛠️ Chapter 3: 3D Gaussian $\to$ 3D Occupancy 복셀 격자 변환 (GS-to-Occ)

### 1. 요약
정제된 3D Gaussian들의 밀도 $\rho_i(\mathbf{x})$와 시맨틱 확률 $\mathbf{c}_i$를 3D 공간 상의 복셀 Grid $\mathbf{x}_{\text{voxel}}$ 지점마다 평가하고 가우시안 래스터화(Rasterization)로 3D Occupancy 라벨을 산출합니다.

### 2. 수식 및 파이썬 코드 설명

$$\rho_i(\mathbf{x}) = \alpha_i \exp\left(-\frac{1}{2} (\mathbf{x} - \mathbf{p}_i)^T \mathbf{\Sigma}_i^{-1} (\mathbf{x} - \mathbf{p}_i)\right)$$

$$\mathbf{C}_{\text{voxel}}(\mathbf{x}) = \text{Argmax}\left( \text{Softmax}\left( \sum_{i=1}^N \rho_i(\mathbf{x}) \cdot \mathbf{c}_i \right) \right)$$

```python
import torch

def convert_gaussians_to_occupancy_grid(
    gaussian_centers: torch.Tensor, # (N, 3) 3D 위치
    gaussian_sigmas: torch.Tensor,  # (N, 3, 3) 공분산 행렬
    gaussian_probs: torch.Tensor,   # (N, C_classes) 시맨틱 클래스 분포
    grid_bounds: tuple,             # (X_min, X_max, Y_min, Y_max, Z_min, Z_max)
    grid_resolution: float = 0.5    # 복셀 1개 크기 (m)
) -> torch.Tensor:
    """
    3D Gaussian 표현을 3D Semantic Occupancy 복셀 그리드로 수렴 변환 (GS-to-Occ)
    """
    X_min, X_max, Y_min, Y_max, Z_min, Z_max = grid_bounds
    Dx = int((X_max - X_min) / grid_resolution)
    Dy = int((Y_max - Y_min) / grid_resolution)
    Dz = int((Z_max - Z_min) / grid_resolution)
    
    # 3D 복셀 중심점 생성
    xs = torch.linspace(X_min, X_max, Dx)
    ys = torch.linspace(Y_min, Y_max, Dy)
    zs = torch.linspace(Z_min, Z_max, Dz)
    
    # 간략화 계산: 최근접 Gaussian 시맨틱 사영
    dist_mat = torch.cdist(torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1).view(-1, 3), gaussian_centers) # (Dx*Dy*Dz, N)
    nearest_idx = torch.argmin(dist_mat, dim=-1)
    
    occ_grid = gaussian_probs[nearest_idx].argmax(dim=-1).view(Dx, Dy, Dz)
    return occ_grid

# --- 사용 예시 ---
sig_d = torch.eye(3).repeat(1000, 1, 1)
occ_map = convert_gaussians_to_occupancy_grid(updated_state.p, sig_d, updated_state.c, (-10, 10, -10, 10, -2, 2), 1.0)
print("생성된 3D Semantic Occupancy Grid Shape:", occ_map.shape)
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. nuScenes 3D Occupancy 예측 성능 리더보드

| 알고리즘 (Method) | 과거 프레임 수 | 레이턴시 (Latency) | 메모리 (Memory) | mIoU ↑ | IoU (지오메트리) ↑ |
|---|---|---|---|---|---|
| **MonoScene** | 0 (Single) | - | - | 7.31 | 23.96 |
| **SurroundOcc** | 0 (Single) | - | - | 20.30 | 31.49 |
| **GaussianFormer-B** | 0 (Single) | 225 ms | 6958 MB | 19.10 | 29.83 |
| **GaussianFormer-T (Temporal)** | 3 | 382 ms | 10019 MB | 20.42 | 31.34 |
| **GaussianWorld (Ours)** | **1 (Streaming)** | **228 ms (초고속)** | **7030 MB (경량)** | **22.13 (+2.4%)** | **33.40 (+3.6%)** |

- **결과**: 역사적 프레임을 단 **1개만 레지스터로 연속 전파**함으로써 레이턴시 상승 없이 3D Occupancy SOTA 달성.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
GaussianWorld는 3D Gaussian의 명시적 공간 표현력을 자율주행 4D Streaming World Model과 최초로 융합하여, 시간적 융합 메모리 폭발 오버헤드 없이 복잡한 드라이빙 환경의 3D Occupancy를 실시간 스트리밍 예측하는 패러다임을 확립했습니다.

### 2. 한계점 및 아쉬운 점
- 동적 요소와 정적 요소의 분리가 완전하지 않아 정적 장면의 크로스 프레임 일관성을 완벽히 보장하지 못한다.
- 신규 영역 완성 요소가 없으면 학습이 붕괴할 정도로 세 요소 간 의존성이 강해, 아키텍처 설계의 견고성 확보를 위해 세밀한 균형 조정이 필요하다.
- GT 자체가 멀티 프레임 LiDAR 누적으로 생성되어 희소한 한계로 인해, 긴 시퀀스에서는 성능이 오히려 하락하는 문제도 완전히 해결되지 않았다.

---

## 참고 자료
- [GaussianWorld 논문 arXiv 페이지 (arXiv:2412.10373)](https://arxiv.org/abs/2412.10373)

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [MonoScene](/posts/papers/monoscene-monocular-3d-semantic-scene-completion/), [TPVFormer](/posts/papers/tpvformer-tri-perspective-view-3d-semantic-occupancy/), [SurroundOcc](/posts/papers/surroundocc/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
