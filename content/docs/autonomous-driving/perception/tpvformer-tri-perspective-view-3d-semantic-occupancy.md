---
title: "TPVFormer: Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction"
date: 2026-04-19T14:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "3D Occupancy"]
tags: ["3D Occupancy", "BEV", "Autonomous Driving", "Transformer", "Semantic Scene Completion", "nuScenes"]
year: 2023
references:
  - "bevformer"
  - "nuscenes-multimodal-dataset-autonomous-driving"
---

## 💡 한 줄 요약
BEV의 z축(높이) 정보 손실과 3D Voxel의 $\mathcal{O}(H W D)$ 연산 폭발 한계를 극복하기 위해 3D 공간을 세 직교 평면 **Tri-Perspective View ($\mathbf{T}^{HW}, \mathbf{T}^{DH}, \mathbf{T}^{WD}$)**으로 분해하고, 3D Point를 세 평면 사영 특징의 합산으로 표현하여 카메라만으로 LiDAR 분할 SOTA의 70% 수준인 69.4 mIoU를 달성했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, Jiwen Lu (Tsinghua Univ., PhiGent Robotics)
- **발행년도**: 2023 (arXiv:2302.07817, CVPR 2023)
- **주요 기여점**:
  1. **Tri-Perspective View (TPV) 3D 공간 분해**: Top $(H \times W)$, Side $(D \times H)$, Front $(W \times D)$ 세 평면으로 3D 표현을 일반화하여 $\mathcal{O}(HW + DH + WD)$의 효율적 복잡도 달성.
  2. **Image Cross-Attention (ICA)**: TPV 쿼리를 3D 시선 상의 $N_{ref}$개 픽셀 좌표로 투영하여 2D 멀티뷰 특징 수집.
  3. **Cross-View Hybrid-Attention (CVHA)**: 직교하는 세 TPV 평면 간에 디포머블 어텐션을 수행하여 높이 및 상호 시공간 문맥 융합.
  4. **희소 LiDAR 감독 (Sparse Supervision) 및 Voxel-Point Joint Loss**: 훈련 시 희소한 LiDAR 레이블만으로 최적화하되, 추론 시 완전한 3D 밀집 Occupancy 복원.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **3D Voxel (MonoScene)**: 3D 복셀 메모리 및 연산량이 $\mathcal{O}(N^3)$으로 폭발하여 고해상도 처리가 어려움.
2. **BEV (BEVFormer)**: z축 높이 정보를 1개 2D 평면으로 압축하여 차량 위상이나 세로로 긴 장애물 인식 한계.
3. **TPVFormer**: 세 직교 평면(Top, Side, Front)의 덧셈 집합으로 BEV와 Voxel의 트레이드오프 완벽 해결.

---

## 📑 목차
- Chapter 1: TPV (Tri-Perspective View) 3D Point Feature Summation
- Chapter 2: Image Cross-Attention (ICA) 투영 어텐션
- Chapter 3: Cross-View Hybrid Attention (CVHA) 수식
- Chapter 4: TPV Broadcast 3D Voxel 복원 & Joint Loss
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: TPV (Tri-Perspective View) 3D Point Feature Summation

### 1. 요약
3D 좌표 $(x, y, z)$를 Top 평면 $\mathbf{T}^{HW}$, Side 평면 $\mathbf{T}^{DH}$, Front 평면 $\mathbf{T}^{WD}$으로 사영하여 인포메이션을 빌리니어 샘플링한 후 합산하여 점 특징 $\mathbf{f}_{x, y, z}$를 생성합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{t}_{h,w} = \text{BilinearSample}(\mathbf{T}^{HW}, \mathcal{P}_{hw}(x,y))$$

$$\mathbf{t}_{d,h} = \text{BilinearSample}(\mathbf{T}^{DH}, \mathcal{P}_{dh}(z,x))$$

$$\mathbf{t}_{w,d} = \text{BilinearSample}(\mathbf{T}^{WD}, \mathcal{P}_{wd}(y,z))$$

$$\mathbf{f}_{x,y,z} = \mathbf{t}_{h,w} + \mathbf{t}_{d,h} + \mathbf{t}_{w,d} \in \mathbb{R}^C$$

```python
import torch
import torch.nn.functional as F

def sample_tpv_point_features(
    T_hw: torch.Tensor, # (B, C, H, W) Top Plane Feature
    T_dh: torch.Tensor, # (B, C, D, H) Side Plane Feature
    T_wd: torch.Tensor, # (B, C, W, D) Front Plane Feature
    pts_3d: torch.Tensor # (B, N_points, 3) -> (x, y, z) [normalized in -1 ~ 1]
) -> torch.Tensor:
    """
    3D 점 (x, y, z)을 세 TPV 평면에 각각 사영 후 특징 합산 (Summation)
    """
    B, C, H, W = T_hw.shape
    N_pts = pts_3d.shape[1]
    
    # 1. 2D 좌표 분할
    xy = pts_3d[..., [0, 1]].unsqueeze(2) # (B, N_pts, 1, 2) Top: (x, y)
    zx = pts_3d[..., [2, 0]].unsqueeze(2) # (B, N_pts, 1, 2) Side: (z, x)
    yz = pts_3d[..., [1, 2]].unsqueeze(2) # (B, N_pts, 1, 2) Front: (y, z)
    
    # 2. Bilinear Grid Sampling
    t_hw = F.grid_sample(T_hw, xy, mode='bilinear', align_corners=True).squeeze(-1).permute(0, 2, 1) # (B, N_pts, C)
    t_dh = F.grid_sample(T_dh, zx, mode='bilinear', align_corners=True).squeeze(-1).permute(0, 2, 1)
    t_wd = F.grid_sample(T_wd, yz, mode='bilinear', align_corners=True).squeeze(-1).permute(0, 2, 1)
    
    # 3. Summation
    f_3d = t_hw + t_dh + t_wd # (B, N_pts, C)
    return f_3d

# --- 사용 예시 ---
t_top = torch.randn(1, 128, 100, 100)
t_side = torch.randn(1, 128, 16, 100)
t_front = torch.randn(1, 128, 100, 16)
p3d = torch.rand(1, 500, 3) * 2.0 - 1.0
print("TPV 점 특징 합산 결과 Shape:", sample_tpv_point_features(t_top, t_side, t_front, p3d).shape)
```

---

## 🛠️ Chapter 2: Image Cross-Attention (ICA) 투영 수식

### 1. 요약
Top, Side, Front 평면 쿼리가 3D 공간 상의 수직 광선을 따라 $N_{ref}$개 픽셀로 사영되어 2D 멀티뷰 카메라 특징으로부터 시각 가중치를 흡수합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{Ref}_{h,w}^{world} = \{(x, y, z_i)\}_{i=1}^{N_{ref}^{HW}}$$

$$\mathbf{Ref}_{h,w}^{pix} = \mathbf{K} \left( \mathbf{R} \cdot \mathbf{Ref}_{h,w}^{world} + \mathbf{t} \right)$$

$$\text{ICA}(\mathbf{t}_{h,w}, \mathbf{I}) = \frac{1}{|N^{val}|} \sum_{j \in N^{val}} \text{DeformAttn}\left( \mathbf{t}_{h,w}, \ \mathbf{Ref}_{h,w}^{pix, j}, \ \mathbf{I}_j \right)$$

```python
import torch

def generate_tpv_ray_reference_points(
    tpv_hw_coords: torch.Tensor, # (H, W, 2) Top 평면 (x, y) 좌표
    z_samples: torch.Tensor      # (N_z,) z축 샘플 고도값들
) -> torch.Tensor:
    """
    Top 평면 쿼리를 위해 z축 방향으로 N_z개 3D Ray Reference Points 확장
    """
    H, W, _ = tpv_hw_coords.shape
    N_z = len(z_samples)
    
    xs = tpv_hw_coords[..., 0].unsqueeze(-1).repeat(1, 1, N_z)
    ys = tpv_hw_coords[..., 1].unsqueeze(-1).repeat(1, 1, N_z)
    zs = z_samples.view(1, 1, N_z).repeat(H, W, 1)
    
    ref_pts_3d = torch.stack([xs, ys, zs], dim=-1) # (H, W, N_z, 3)
    return ref_pts_3d

# --- 사용 예시 ---
hw_grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 10), torch.linspace(-1, 1, 10), indexing='ij'), dim=-1)
z_bins = torch.linspace(-2, 2, 8)
print("생성된 TPV Ray Reference Points Shape:", generate_tpv_ray_reference_points(hw_grid, z_bins).shape)
```

---

## 🛠️ Chapter 3: TPV Broadcast 3D Voxel 복원 & Joint Loss

### 1. 요약
세 직교 TPV 평면을 브로드캐스팅(Broadcast)하여 $(H \times W \times D)$ 3D Voxel Tensor를 복원하고, Point Prediction과 Voxel Prediction 손실을 교차 최적화합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{V}(x, y, z) = \mathbf{T}^{HW}(x, y) + \mathbf{T}^{DH}(z, x) + \mathbf{T}^{WD}(y, z) \in \mathbb{R}^{H \times W \times D \times C}$$

$$\mathcal{L}_{total} = \mathcal{L}_{ce}(\text{PointPred}, \text{GT}_{point}) + \mathcal{L}_{ce}(\text{VoxelPred}, \text{GT}_{voxel})$$

```python
import torch
import torch.nn.functional as F

def reconstruct_full_3d_voxel_from_tpv(
    T_hw: torch.Tensor, # (B, C, H, W)
    T_dh: torch.Tensor, # (B, C, D, H)
    T_wd: torch.Tensor  # (B, C, W, D)
) -> torch.Tensor:
    """
    3개 TPV 평면의 Broadcast Summation을 통한 (H, W, D) 3D Voxel Tensor 완전 복원
    """
    B, C, H, W = T_hw.shape
    D = T_dh.shape[2]
    
    # Broadcast 준비
    hw_bcast = T_hw.unsqueeze(-1).repeat(1, 1, 1, 1, D)             # (B, C, H, W, D)
    dh_bcast = T_dh.permute(0, 1, 3, 2).unsqueeze(3).repeat(1, 1, 1, W, 1) # (B, C, H, W, D)
    wd_bcast = T_wd.permute(0, 1, 2, 3).unsqueeze(2).repeat(1, 1, H, 1, 1) # (B, C, H, W, D)
    
    voxel_tensor = hw_bcast + dh_bcast + wd_bcast
    return voxel_tensor

# --- 사용 예시 ---
th = torch.randn(1, 64, 50, 50)
td = torch.randn(1, 64, 10, 50)
tw = torch.randn(1, 64, 50, 10)
print("복원된 3D Voxel Tensor Shape:", reconstruct_full_3d_voxel_from_tpv(th, td, tw).shape)
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. nuScenes LiDAR Semantic Segmentation 벤치마크 (Camera-Only)

| 알고리즘 (Method) | 입력 모달리티 | 3D 공간 표현 | mIoU ↑ |
|---|---|---|---|
| **Cylinder3D++** | LiDAR (Super SOTA) | 3D Voxel | 77.9 |
| **BEVFormer-Base** | Camera | 2D BEV Grid | 56.2 |
| **TPVFormer-Small (Ours)** | **Camera** | **Tri-Perspective View** | **59.2 (+3.0%)** |
| **TPVFormer-Base (Ours)** | **Camera** | **Tri-Perspective View** | **69.4 (+13.2%)** |

- **결과**: 세 직교 평면 융합 덕분에 BEVFormer 대비 **mIoU +13.2%p 폭발적 향상**으로 LiDAR 성능의 70% 초과 달성.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
TPVFormer는 BEV의 z축 정보 손실과 3D Voxel의 연산량 폭발 문제를 3개 직교 평면 분해(Tri-Perspective View)로 명쾌하게 극복한 카메라이 기반 3D Occupancy 분야의 세미날 논문입니다.

### 2. 한계점 및 아쉬운 점
- 여전히 LiDAR SOTA(77.9 mIoU) 대비 약 8.5%p의 격차가 남아 있으며, 희소 LiDAR supervision에 의존하므로 완전히 LiDAR-free한 학습 파이프라인은 아니다.
- 세 평면의 단순 합산(summation)이 항상 최적의 융합 방식인지에 대한 이론적 분석은 제한적이다.

---

## 참고 자료
- [TPVFormer 공식 GitHub 저장소](https://github.com/wzzheng/TPVFormer)
- [CVPR 2023 논문 (arXiv:2302.07817)](https://arxiv.org/abs/2302.07817)

*관련 논문: [MonoScene](/posts/papers/monoscene-monocular-3d-semantic-scene-completion/), [BEVFormer](/posts/papers/bevformer/), [SurroundOcc](/posts/papers/surroundocc/), [Occ3D](/posts/papers/occ3d-large-scale-3d-occupancy-prediction-benchmark/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
