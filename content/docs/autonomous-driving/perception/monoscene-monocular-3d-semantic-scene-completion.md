---
title: "MonoScene: Monocular 3D Semantic Scene Completion"
date: 2026-04-19T21:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "3D Scene Understanding"]
tags: ["Occupancy Prediction", "Semantic Scene Completion", "NeRF", "BEV", "3D Understanding"]
year: 2022
references:
  - "resnet-deep-residual-learning-for-image-recognition"
---

## 💡 한 줄 요약
단일 RGB 이미지 입력만으로 3D 공간의 기하(Geometry)와 시맨틱(Semantic)을 동시 완료(Completion)하기 위해 FLoSP(시선 기반 2D-3D 피처 투영), 3D CRP(공간-의미 맥락 사전), Scene-Class Affinity & Frustum Proportion Loss를 통합한 최초의 단안 3D Semantic Scene Completion(SSC) 아키텍처다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Anh-Quan Cao, Raoul de Charette (Inria)
- **발행년도**: 2022 (arXiv:2112.00726, CVPR 2022)
- **주요 기여점**:
  1. **FLoSP (Features Line of Sight Projection)**: 별도의 깊이 추정기 없이 카메라 시선(Ray)을 따라 2D 픽셀 멀티스케일 특징을 3D 복셀 그리드로 연속 사영.
  2. **3D CRP (3D Context Relation Prior)**: Supervoxel 간 4가지 전역 공간 관계(Free/Occupied $\times$ Same/Diff Class)를 학습하는 3D UNet 병목 레이어.
  3. **Scene-Class Affinity Loss ($\mathcal{L}_{scal}$)**: 클래스별 정밀도(Precision), 재현율(Recall), 특이도(Specificity)를 직접 최적화하여 Cross-Entropy의 전역 맥락 무시 극복.
  4. **Frustum Proportion Loss ($\mathcal{L}_{fp}$)**: 이미지 패치별 3D Frustum 가림(Occlusion) 영역의 클래스 확률 분포 KL 발산을 축소.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **3D SSC (SSCNet, LMSCNet, 3DSketch)**: LiDAR, Depth 센서, TSDF 3D 입력을 필수적으로 요구하여 고비용 센서 의존성 존재.
2. **MonoScene**: 단안 RGB 이미지만으로 깊이 센서 기반 SOTA 기법들을 mIoU 면에서 추월하는 패러다임 전환.

---

## 📑 목차
- Chapter 1: FLoSP (Features Line of Sight Projection) 투영 수식
- Chapter 2: 3D CRP (Context Relation Prior) 및 관계 손실 함수
- Chapter 3: Scene-Class Affinity Loss ($\mathcal{L}_{scal}$)
- Chapter 4: Frustum Proportion KL-Divergence Loss ($\mathcal{L}_{fp}$)
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: FLoSP (Features Line of Sight Projection) 투영 수식

### 1. 요약
3D 복셀 중심점 $x^c$를 카메라 내재 행렬로 2D 픽셀 평면에 사영하여, 시선(Line of Sight) 상의 2D 멀티스케일 특징 $F_{2D}^{1:s}$를 Bilinear Interpolation으로 샘플링 후 합산합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{F}_{3D}(x^c) = \sum_{s \in S} \text{BilinearSample}\left( F_{2D}^{1:s}, \ \rho(x^c) \right)$$

$$\rho(x^c) = \mathbf{K} \left( \mathbf{R} x^c + \mathbf{t} \right)$$

```python
import torch
import torch.nn.functional as F

def flosp_2d_to_3d_line_of_sight_projection(
    feat_2d: torch.Tensor,       # (B, C, H_2d, W_2d) 2D 이미지 특징
    voxel_centers_3d: torch.Tensor, # (Dx, Dy, Dz, 3) 3D 복셀 중심 좌표
    K_intrinsics: torch.Tensor   # (3, 3) 카메라 내재 행렬
) -> torch.Tensor:
    """
    MonoScene FLoSP: 3D 복셀 중심점을 2D 이미지 좌표로 사영하여 3D 피처 맵으로 복원
    """
    Dx, Dy, Dz, _ = voxel_centers_3d.shape
    B, C, H_2d, W_2d = feat_2d.shape
    
    # 1. 3D 복셀 중심점 2D 투영: p_2d = K @ x_3d
    pts_flat = voxel_centers_3d.view(-1, 3).T # (3, N_voxels)
    pts_2d_homo = (K_intrinsics @ pts_flat).T # (N_voxels, 3)
    
    depths = pts_2d_homo[:, 2]
    u = pts_2d_homo[:, 0] / (depths + 1e-6)
    v = pts_2d_homo[:, 1] / (depths + 1e-6)
    
    # 2. [-1, 1] 범위로 Grid Sample 정규화
    norm_u = (u / W_2d) * 2.0 - 1.0
    norm_v = (v / H_2d) * 2.0 - 1.0
    grid = torch.stack([norm_u, norm_v], dim=-1).view(1, Dx, Dy * Dz, 2).repeat(B, 1, 1, 1)
    
    # 3. Bilinear Interpolation 특징 샘플링
    sampled_feat = F.grid_sample(feat_2d, grid, mode='bilinear', align_corners=True) # (B, C, Dx, Dy*Dz)
    feat_3d = sampled_feat.view(B, C, Dx, Dy, Dz)
    
    return feat_3d

# --- 사용 예시 ---
f2d_in = torch.randn(1, 64, 60, 80)
vox_3d = torch.randn(30, 30, 10, 3) + torch.tensor([0.0, 0.0, 5.0])
K_in = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
print("생성된 3D Feature Map Shape:", flosp_2d_to_3d_line_of_sight_projection(f2d_in, vox_3d, K_in).shape)
```

---

## 🛠️ Chapter 2: Scene-Class Affinity Loss ($\mathcal{L}_{scal}$)

### 1. 요약
각 시맨틱 클래스 $c$에 대해 전역 복셀 픽셀들의 정밀도 $P_c$, 재현율 $R_c$, 특이도 $S_c$를 직접 계산하여 클래스 불균형을 해결하는 손실 함수입니다.

### 2. 수식 및 파이썬 코드 설명

$$P_c = \log \frac{\sum_i \hat{p}_{i, c} \cdot \mathbb{I}(p_i = c)}{\sum_i \hat{p}_{i, c} + 1e-6}, \quad R_c = \log \frac{\sum_i \hat{p}_{i, c} \cdot \mathbb{I}(p_i = c)}{\sum_i \mathbb{I}(p_i = c) + 1e-6}$$

$$\mathcal{L}_{scal} = -\frac{1}{C} \sum_{c=1}^{C} \left( P_c + R_c + S_c \right)$$

```python
import torch

def compute_scene_class_affinity_loss(
    pred_probs: torch.Tensor, # (N_voxels, C_classes) 예측 3D 복셀 클래스 확률
    gt_labels: torch.Tensor   # (N_voxels,) GT 3D 복셀 라벨
) -> torch.Tensor:
    """
    MonoScene 전역 복셀 클래스 Affinity Loss (Precision + Recall + Specificity)
    """
    N_voxels, C = pred_probs.shape
    total_loss = 0.0
    
    for c in range(C):
        gt_c = (gt_labels == c).float()
        pred_c = pred_probs[:, c]
        
        tp = (pred_c * gt_c).sum()
        fp = (pred_c * (1.0 - gt_c)).sum()
        fn = ((1.0 - pred_c) * gt_c).sum()
        tn = ((1.0 - pred_c) * (1.0 - gt_c)).sum()
        
        precision = torch.log((tp + 1e-6) / (tp + fp + 1e-6))
        recall = torch.log((tp + 1e-6) / (tp + fn + 1e-6))
        specificity = torch.log((tn + 1e-6) / (tn + fp + 1e-6))
        
        total_loss -= (precision + recall + specificity)
        
    return total_loss / C

# --- 사용 예시 ---
p_voxel = torch.softmax(torch.randn(1000, 12), dim=-1)
g_voxel = torch.randint(0, 12, (1000,))
print("Scene-Class Affinity Loss:", compute_scene_class_affinity_loss(p_voxel, g_voxel).item())
```

---

## 🛠️ Chapter 3: Frustum Proportion Loss ($\mathcal{L}_{fp}$)

### 1. 요약
이미지 패치별 3D Frustum 영역 내 시맨틱 클래스 비율 $P_k(c)$와 예측 비율 $\hat{P}_k(c)$ 간의 KL Divergence를 최소화하여 가림(Occlusion) 영역을 정정합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{L}_{fp} = \sum_{k=1}^{\ell^2} D_{KL}(P_k \parallel \hat{P}_k) = \sum_{k=1}^{\ell^2} \sum_{c \in C_k} P_k(c) \log \frac{P_k(c)}{\hat{P}_k(c)}$$

```python
import torch
import torch.nn.functional as F

def compute_frustum_proportion_kl_loss(
    pred_frustum_dist: torch.Tensor, # (N_frustums, C_classes) Frustum별 예측 클래스 비율
    gt_frustum_dist: torch.Tensor    # (N_frustums, C_classes) Frustum별 GT 클래스 비율
) -> torch.Tensor:
    """
    Frustum 내 3D 시맨틱 확률 분포의 KL-Divergence 손실
    """
    # PyTorch kl_div(input=log_pred, target=gt)
    log_pred = torch.log(pred_frustum_dist + 1e-6)
    kl_loss = F.kl_div(log_pred, gt_frustum_dist, reduction='batchmean')
    return kl_loss

# --- 사용 예시 ---
p_dist = torch.softmax(torch.randn(64, 12), dim=-1)
g_dist = torch.softmax(torch.randn(64, 12), dim=-1)
print("Frustum Proportion KL Loss:", compute_frustum_proportion_kl_loss(p_dist, g_dist).item())
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. SemanticKITTI & NYUv2 3D SSC 벤치마크 성능 비교

| 데이터셋 (Dataset) | 모델 (Method) | 입력 모달리티 | IoU (지오메트리) ↑ | mIoU (시맨틱) ↑ |
|---|---|---|---|---|
| **SemanticKITTI** | LMSCNet | Occupancy | 31.38 | 7.07 |
| **SemanticKITTI** | AICNet | RGB + Depth | 23.93 | 7.09 |
| **SemanticKITTI** | **MonoScene (Ours)** | **RGB Only (단안)** | **34.16 (+2.78%)** | **11.08 (+3.99%)** |
| **NYUv2 (Indoor)** | **MonoScene (Ours)** | **RGB Only (단안)** | **42.51** | **26.94** |

- **결과**: 깊이 센서(Depth/LiDAR)가 없는 순수 RGB 단안 입력만으로 기존 깊이 기반 SOTA 기법들을 **mIoU 면에서 대폭 초과 달성**.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
MonoScene은 FLoSP 시선 투영 메커니즘과 3D CRP, 전역 손실 함수를 통해 깊이 센서가 결여된 2D RGB 이미지만으로 밀집한 3D Semantic Occupancy를 복원한 단안 3D 인지의 선구적 논문입니다.

### 2. 한계점 및 아쉬운 점
- 세밀한 기하(fine-grained geometry) 추론에 어려움이 있어 작은 물체나 얇은 구조물을 놓치기 쉽다.
- 의미적으로 유사한 클래스(의자/소파, 창문/물체 등)를 혼동하는 경우도 있다.
- 실외 장면에서는 시선 방향을 따라 아티팩트가 발생하는 single viewpoint의 근본적 한계가 있고, 카메라 FOV가 훈련 데이터 설정과 다를 때 성능이 저하된다.

---

## 참고 자료
- [MonoScene 공식 GitHub 저장소](https://github.com/astra-vision/MonoScene)
- [CVPR 2022 논문 (arXiv:2112.00726)](https://arxiv.org/abs/2112.00726)

*관련 논문: [TPVFormer](/posts/papers/tpvformer-tri-perspective-view-3d-semantic-occupancy/), [SurroundOcc](/posts/papers/surroundocc/), [Occ3D](/posts/papers/occ3d-large-scale-3d-occupancy-prediction-benchmark/), [BEVFormer](/posts/papers/bevformer/), [GaussianWorld](/posts/papers/gaussianworld-gaussian-world-model-for-streaming-3d-occupancy-prediction/)*
