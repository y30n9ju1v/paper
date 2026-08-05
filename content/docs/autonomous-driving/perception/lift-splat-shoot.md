---
title: "Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D"
date: 2026-04-17T08:30:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving"]
tags: ["Autonomous Driving", "BEV", "Multi-Camera", "3D Object Detection"]
year: 2020
references:
  - "resnet-deep-residual-learning-for-image-recognition"
  - "nuscenes-multimodal-dataset-autonomous-driving"
---

## 💡 한 줄 요약
임의의 카메라 파라미터 구조를 가진 멀티뷰 RGB 이미지에서 이산 깊이 범주 분포(Categorical Depth Distribution)를 예측하여 3D Frustum으로 "Lift"하고, 자차 기준 BEV 평면상에 "Splat"한 뒤, 템플릿 궤적 비용을 최적화하는 "Shoot" 구조로 End-to-End 카메라 BEV 표현 파이프라인의 시초를 열었다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Jonah Philion, Sanja Fidler (NVIDIA, University of Toronto, Vector Institute)
- **발행년도**: 2020 (arXiv:2008.05711, ECCV 2020)
- **주요 기여점**:
  1. **Lift (3D Unprojection)**: 단안 픽셀의 깊이 모호성을 이산 확률 분포 $\alpha$로 모델링하고, 2D 컨텍스트 특징 $\mathbf{c}$와의 외적(Outer Product)으로 3D Frustum 특징 생성.
  2. **Splat (BEV Pillar Pooling)**: 카메라 뷰별 Frustum 포인트들을 자차(Ego) 기준 2D BEV 그리드(Pillar)에 집계(Sum Pooling).
  3. **Cumulative Sum (CumSum) Trick**: GPU 메모리 할당 및 패딩 오버헤드 없이 정렬과 누적합 연산만으로 Pooling 역전파 속도를 2배 가속.
  4. **Shoot (Motion Planning)**: 예측된 BEV Cost Map 상에 템플릿 궤적들을 투사(Shoot)하여 Boltzmann 확률 분포로 종단간 모션 플래닝 통합.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **OFT (Orthographic Feature Transform)**: 3D 복셀을 이미지에 투영해 특징을 모으나 깊이 분포와 상관없이 동일 특징을 기여함.
2. **Pseudo-LiDAR**: 단안 깊이를 포인트 클라우드로 변환 후 3D 탐지기를 연결하나 End-to-End 미분 불가능.
3. **Lift-Splat-Shoot**: 카메라 파라미터가 무작위로 변경(Arbitrary Camera Rigs)되어도 유연하게 대응하는 최초의 완전 미분 가능한 LSS 프레임워크.

---

## 📑 목차
- Chapter 1: Lift — 범주형 깊이 확률 분포 및 Frustum 특징 생성
- Chapter 2: Splat — 3D 역투영 및 BEV Pillar Sum Pooling
- Chapter 3: Cumulative Sum (CumSum) Trick 고속 풀링 구현
- Chapter 4: Shoot — Boltzmann 궤적 평가 및 모션 플래닝
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: Lift — 범주형 깊이 확률 분포 및 Frustum 특징 생성

### 1. 요약
각 카메라 이미지 픽셀 $(u, v)$에 대해 이산 깊이 Bin $D = \{d_0, d_1, \dots, d_{|D|-1}\}$에 속할 확률 $\alpha_d(u, v)$와 컨텍스트 특징 $\mathbf{c}(u, v)$를 예측한 뒤, 두 벡터의 외적으로 3D Frustum 특징 $\mathbf{c}_d(u, v)$를 생성합니다.

### 2. 수식 및 파이썬 코드 설명

$$\boldsymbol{\alpha}(u, v) = \text{Softmax}\left( \text{Conv2D}(F_{2D}) \right) \in \mathbb{R}^{|D|}, \quad \mathbf{c}(u, v) = \text{Conv2D}(F_{2D}) \in \mathbb{R}^{C}$$

$$\mathbf{c}_d(u, v) = \alpha_d(u, v) \cdot \mathbf{c}(u, v) \in \mathbb{R}^{C} \quad (d \in D)$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSSLiftModule(nn.Module):
    """
    LSS Lift 단계: 2D 이미지 특징에서 이산 깊이 확률 분포 alpha와 Context feature c 생성
    """
    def __init__(self, in_channels: int = 256, num_depth_bins: int = 41, context_dim: int = 64):
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.context_dim = context_dim
        
        # Depth + Context 합성곱 헤드
        self.depth_context_head = nn.Conv2d(in_channels, num_depth_bins + context_dim, kernel_size=1)

    def forward(self, img_feat: torch.Tensor) -> torch.Tensor:
        """
        img_feat: (B, C_in, H, W)
        Return Frustum Feat: (B, D, C_ctx, H, W)
        """
        B, _, H, W = img_feat.shape
        out = self.depth_context_head(img_feat)
        
        # 1. 깊이 확률 분포 (Softmax) 및 Context 분리
        depth_logits = out[:, :self.num_depth_bins, :, :]
        depth_prob = F.softmax(depth_logits, dim=1) # (B, D, H, W)
        
        context = out[:, self.num_depth_bins:, :, :] # (B, C_ctx, H, W)
        
        # 2. Outer Product: Frustum = Depth_prob (B, D, 1, H, W) * Context (B, 1, C_ctx, H, W)
        frustum_feat = depth_prob.unsqueeze(2) * context.unsqueeze(1)
        return frustum_feat # (B, D, C_ctx, H, W)

# --- 사용 예시 ---
f_2d = torch.randn(1, 256, 16, 44)
lift = LSSLiftModule()
print("Lift Frustum Feature Shape:", lift(f_2d).shape)
```

---

## 🛠️ Chapter 2: Splat — 3D 역투영 및 BEV Pillar Sum Pooling

### 1. 요약
2D 픽셀 $(u, v)$ 및 이산 깊이 $d$로부터 카메라 좌표계 3D 좌표 $\mathbf{p}_{3D}$를 계산하고, 외재 파라미터 $(R, t)$를 이용해 자차 좌표계 3D 좌표 $\mathbf{p}_{BEV}$로 변환 후 지정된 Pillar 셀에 특징들을 Sum Pooling합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{p}_{3D}(u, v, d) = d \cdot \mathbf{K}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

$$\mathbf{p}_{BEV}(u, v, d) = \mathbf{R} \cdot \mathbf{p}_{3D}(u, v, d) + \mathbf{t}$$

$$\text{BEV}(x, y) = \sum_{(u, v, d) \in \text{Pillar}(x, y)} \mathbf{c}_d(u, v)$$

```python
import torch

def unproject_frustum_to_bev_coords(
    depth_bins: torch.Tensor, # (D,) 이산 깊이값들 (m)
    K_inv: torch.Tensor,       # (3, 3) 카메라 역내재 행렬
    R_c2e: torch.Tensor,       # (3, 3) 카메라->Ego 회전
    t_c2e: torch.Tensor,       # (3,) 카메라->Ego 이동
    img_shape: tuple           # (H, W)
) -> torch.Tensor:
    """
    Frustum 각 픽셀(u, v, d)을 3D Ego 좌표계 좌표로 역투영
    """
    H, W = img_shape
    D = len(depth_bins)
    
    # 2D Grid (u, v, 1)
    vs = torch.arange(0, H).float()
    us = torch.arange(0, W).float()
    grid_v, grid_u = torch.meshgrid(vs, us, indexing='ij')
    
    # (3, H*W)
    pts_2d = torch.stack([grid_u.flatten(), grid_v.flatten(), torch.ones(H*W)], dim=0)
    
    # 3D normalized rays in camera frame
    rays_cam = K_inv @ pts_2d # (3, H*W)
    
    # (D, 3, H*W)
    pts_cam_3d = depth_bins.view(D, 1, 1) * rays_cam.unsqueeze(0)
    
    # Ego 좌표계 변환: p_ego = R @ p_cam + t
    # (D, H*W, 3)
    pts_cam_flat = pts_cam_3d.permute(0, 2, 1).reshape(-1, 3)
    pts_ego = (R_c2e @ pts_cam_flat.T).T + t_c2e
    
    return pts_ego.view(D, H, W, 3)

# --- 사용 예시 ---
bins = torch.linspace(4.0, 45.0, 41)
K_i = torch.eye(3)
R_e, t_e = torch.eye(3), torch.zeros(3)
pts_ego_res = unproject_frustum_to_bev_coords(bins, K_i, R_e, t_e, (16, 44))
print("역투영된 3D Ego 좌표 Shape:", pts_ego_res.shape)
```

---

## 🛠️ Chapter 3: Cumulative Sum (CumSum) Trick 구현

### 1. 요약
GPU 메모리 패딩이나 불필요한 스레드 대기 없이, 포인트들을 BEV Bin 인덱스 순서로 정렬(Sort)한 후 `torch.cumsum`과 구간 경계 차분으로 BEV Sum Pooling을 고속 계산합니다.

### 2. 수식 및 파이썬 코드 설명

$$\text{CumSum}(k) = \sum_{j=1}^k \mathbf{c}_{d, j}$$

$$\text{PillarSum}(m) = \text{CumSum}(\text{end}_m) - \text{CumSum}(\text{start}_m)$$

```python
import torch

def cumsum_trick_bev_pooling(
    feats: torch.Tensor,      # (N, C) Frustum 특징들
    bin_ids: torch.Tensor,    # (N,) 정수형 BEV 셀 Bin 인덱스
    num_bins: int             # 총 BEV 셀 개수 H * W
) -> torch.Tensor:
    """
    LSS의 핵심: Cumulative Sum Trick 기반 고속 Pillar Sum Pooling
    """
    N, C = feats.shape
    
    # 1. Bin ID 순서로 정렬
    ranks = torch.argsort(bin_ids)
    sorted_feats = feats[ranks]
    sorted_bins = bin_ids[ranks]
    
    # 2. Cumulative Sum 계산
    cum_sum = torch.cumsum(sorted_feats, dim=0)
    
    # 3. Bin 경계 지점 (Section Boundaries) 검출
    kept = torch.ones(N, dtype=torch.bool, device=feats.device)
    kept[:-1] = sorted_bins[:-1] != sorted_bins[1:]
    
    end_indices = torch.where(kept)[0]
    
    # 구간 차분으로 Bin별 Sum 추출
    slice_sum = cum_sum[end_indices]
    prev_sum = torch.cat([torch.zeros((1, C), device=feats.device), slice_sum[:-1]], dim=0)
    final_bin_sum = slice_sum - prev_sum
    
    unique_bins = sorted_bins[end_indices]
    
    bev_flat = torch.zeros((num_bins, C), device=feats.device)
    bev_flat[unique_bins] = final_bin_sum
    return bev_flat

# --- 사용 예시 ---
f_dummy = torch.randn(1000, 64)
b_dummy = torch.randint(0, 100, (1000,))
bev_res = cumsum_trick_bev_pooling(f_dummy, b_dummy, 100)
print("CumSum Trick Pooling 결과 Shape:", bev_res.shape)
```

---

## 🛠️ Chapter 4: Shoot — Boltzmann 궤적 평가 및 모션 플래닝

### 1. 요약
생성된 BEV 특징 위에서 학습된 Cost Map $c_o(x, y)$을 추출하고, 사전 정의된 템플릿 궤적 $\tau_i$ 상의 누적 비용에 대한 Boltzmann 확률 분포를 계산하여 최선의 주행 궤적을 선택합니다.

### 2. 수식 및 파이썬 코드 설명

$$p(\tau_i \mid o) = \frac{\exp\left(-\sum_{(x, y) \in \tau_i} c_o(x, y)\right)}{\sum_{\tau \in \mathcal{T}} \exp\left(-\sum_{(x', y') \in \tau} c_o(x', y')\right)}$$

```python
import torch
import torch.nn.functional as F

def shoot_template_trajectories(
    bev_cost_map: torch.Tensor, # (H, W) 예측된 BEV 공간 주행 위험도 Cost Map
    trajectories: torch.Tensor  # (K_templates, T_steps, 2) 궤적 템플릿 픽셀 좌표 (x, y)
) -> torch.Tensor:
    """
    Shoot 단계: 템플릿 궤적들에 대한 Boltzmann 확률 분포 평가
    """
    K, T, _ = trajectories.shape
    costs = []
    
    for i in range(K):
        traj = trajectories[i] # (T, 2)
        xs = traj[:, 0].long().clamp(0, bev_cost_map.shape[1]-1)
        ys = traj[:, 1].long().clamp(0, bev_cost_map.shape[0]-1)
        
        # 궤적 누적 비용
        traj_cost = bev_cost_map[ys, xs].sum()
        costs.append(traj_cost)
        
    costs_tensor = torch.stack(costs)
    # Boltzmann Probability (비용이 낮을수록 높은 확률)
    probs = F.softmax(-costs_tensor, dim=0)
    return probs

# --- 사용 예시 ---
c_map = torch.rand(200, 200)
trajs = torch.randint(0, 200, (10, 20, 2)).float()
print("최적 궤적 선택 확률:", shoot_template_trajectories(c_map, trajs))
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. nuScenes & Lyft 3D 분할 및 모션 플래닝 성과

| 알고리즘 (Method) | 사용 카메라 수 | nuScenes Car mIoU ↑ | Lyft Car mIoU ↑ | Zero-Shot Transfer mIoU ↑ |
|---|---|---|---|---|
| **CNN Baseline** | 6 Cameras | 22.78 | 30.71 | 7.00 |
| **OFT** | 6 Cameras | 29.72 | 39.48 | 16.25 |
| **Lift-Splat (Ours)** | **6 Cameras** | **32.06 (+2.3%)** | **43.09 (+3.6%)** | **21.35 (+5.1%)** |

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
Lift-Splat-Shoot은 카메라 파라미터를 불변(Implicit Unprojection)으로 수용하여 단안 깊이의 범주형 확률 분포를 3D BEV 공간으로 사영하고 모션 플래닝까지 잇는 현대 카메라 기반 3D Perception의 기초 석조입니다.

### 2. 한계점 및 아쉬운 점
- 야간이나 장거리에서 LiDAR 대비 성능 격차가 존재한다.
- 단일 타임스텝만 처리하므로 temporal 정보를 전혀 활용하지 못한다 (이는 이후 BEVDet4D, BEVFormer 등이 해결).
- 모션 플래닝 성능도 LiDAR 기반보다 명확히 낮아, 실제 배포에는 추가 검증이 필요하다.

---

## 참고 자료
- [Lift-Splat-Shoot 공식 GitHub 저장소](https://github.com/nv-tlabs/lift-splat-shoot)
- [ECCV 2020 논문 (arXiv:2008.05711)](https://arxiv.org/abs/2008.05711)

*관련 논문: [BEVDepth](/posts/papers/bevdepth/), [BEVFormer](/posts/papers/bevformer/), [BEVFusion](/posts/papers/bevfusion-multi-task-multi-sensor-fusion/), [DETR3D](/posts/papers/detr3d-3d-object-detection-multi-view-images/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
