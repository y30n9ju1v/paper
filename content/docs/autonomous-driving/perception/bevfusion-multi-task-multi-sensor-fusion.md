---
title: "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation"
date: 2026-04-17T08:30:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving"]
tags: ["Autonomous Driving", "Sensor Fusion", "BEV", "LiDAR", "3D Object Detection"]
year: 2024
references:
  - "bevformer"
  - "centerpoint-center-based-3d-object-detection-and-tracking"
  - "nuscenes-multimodal-dataset-autonomous-driving"
---

## 💡 한 줄 요약
카메라와 LiDAR 특징을 공유 BEV(Bird's-Eye View) 공간에서 합산(Concat) 및 합성곱(ConvNet)으로 결합하여 기하학적·의미론적 정보를 동시 보존하고, Precomputation과 Interval Reduction으로 Camera-to-BEV 변환 속도를 40배 가속(500ms $\to$ 12ms)하여 nuScenes 3D 탐지(70.2 mAP) 및 맵 분할(62.7 mIoU) SOTA를 달성했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Zhijian Liu, Haotian Tang, Alexander Amini, Xinyu Yang, Huizi Mao, Daniela L. Rus, Song Han (MIT, NVIDIA)
- **발행년도**: 2024 (arXiv 2022, CVPR 2023)
- **주요 기여점**:
  1. **공유 BEV 공간 (Shared BEV Space) 통합 융합**: 포인트 기반 융합(Point-level Fusion) 시 발생하는 카메라 의미론적 밀도 손실을 방지하기 위해 카메라와 LiDAR 특징 모두를 2D BEV 그리드로 이산화하여 채널 결합.
  2. **40배 고속 Camera-to-BEV 변환 (Interval Reduction)**: 카메라 내/외재 파라미터 매핑 테이블 사전 계산(Precomputation)과 커널 기반 구간 합산(Interval Reduction)으로 BEV Pooling 병목을 500ms에서 12ms로 단축.
  3. **완전 합성곱 융합 (Fully-Convolutional Fusion)**: 카메라 깊이 추정 오차로 인한 공간 오정렬(Spatial Misalignment)을 2D ConvNet의 수용 영역(Receptive Field)으로 상쇄 및 수용.
  4. **멀티태스크 멀티센서 아키텍처**: 하나의 융합 BEV 인코더로 3D 객체 탐지(CenterPoint Head)와 BEV 맵 분할(Segmentation Head)을 동시 수행.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **Point-Level Fusion (PointPainting, MVP)**: LiDAR 포인트 3D 좌표 상에 2D 카메라 클래스 점수를 사영(Paint)하는 방식. LiDAR 포인트의 희소성으로 인해 카메라 픽셀의 90% 이상이 버려짐.
2. **Feature-Level Fusion (TransFusion)**: Query 어텐션 기반 융합이나, 카메라와 LiDAR의 뷰 좌표 불일치(View Discrepancy)로 맵 분할 등의 의미론적 태스크 할당이 어려움.
3. **BEVFusion**: 두 모달리티를 모두 표현력이 뛰어난 2D BEV 격자로 가져와 기하(LiDAR)와 의미(Camera)를 완벽 융합.

---

## 📑 목차
- Chapter 1: 공유 BEV 공간 (Shared BEV Space) 아키텍처
- Chapter 2: 고속 Camera-to-BEV 변환 (Precomputation & Interval Reduction)
- Chapter 3: 완전 합성곱 BEV 융합 (Fully-Convolutional Fusion)
- Chapter 4: 멀티태스크 탐지 및 분할 헤드 수식 (Heatmap Focal Loss)
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: 공유 BEV 공간 아키텍처

### 1. 요약
멀티뷰 RGB 이미지는 2D 백본 및 LSS Transform을 통해 카메라 BEV 특징 $F_{\text{BEV}}^{cam} \in \mathbb{R}^{C_{c} \times H \times W}$로, LiDAR 포인트 클라우드는 VoxelNet/PointPillars 백본을 통해 LiDAR BEV 특징 $F_{\text{BEV}}^{lidar} \in \mathbb{R}^{C_{l} \times H \times W}$로 각각 변환 후 결합됩니다.

---

## 🛠️ Chapter 2: 고속 Camera-to-BEV 변환 (Interval Reduction)

### 1. 요약
각 카메라 픽셀 $(u, v)$에 할당된 이산 깊이 분포 $D_i(d, u, v)$와 컨텍스트 특징 $F_i(c, u, v)$의 외적으로 생성된 Frustum 포인트들을 2D BEV 셀 $(x, y)$에 집계할 때, 미리 계산된 Index Table과 GPU 스레드 기반 구간 합산(Interval Reduction) 커널로 지연시간을 12ms로 단축합니다.

### 2. 수식 및 파이썬 코드 설명

$$F_{\text{BEV}}^{cam}(x, y) = \text{Aggregate}\left( \left\{ D_i(d, u, v) \cdot F_i(c, u, v) \ \Big| \ \text{Proj}(u, v, d) \in \text{Cell}(x, y) \right\} \right)$$

```python
import torch

def efficient_camera_to_bev_pooling(
    frustum_feats: torch.Tensor, # (N_points, C) 깊이*컨텍스트 Frustum 특징
    geom_bev_indices: torch.Tensor, # (N_points,) Precomputation으로 구한 BEV 셀 1D 인덱스 (0 ~ H*W-1)
    bev_shape: tuple             # (H, W) BEV 그리드 크기
) -> torch.Tensor:
    """
    Interval Reduction 원리를 모사한 GPU 고속 Scatter Add BEV Pooling
    """
    H, W = bev_shape
    N_points, C = frustum_feats.shape
    
    # 1. 유효한 BEV 인덱스 필터링 (범위 밖 제거)
    valid_mask = (geom_bev_indices >= 0) & (geom_bev_indices < H * W)
    valid_feats = frustum_feats[valid_mask]
    valid_indices = geom_bev_indices[valid_mask]
    
    # 2. CUDA index_add_ / scatter_add 기반 구간 합산 (Interval Reduction)
    bev_flat = torch.zeros((H * W, C), dtype=frustum_feats.dtype, device=frustum_feats.device)
    bev_flat.index_add_(0, valid_indices, valid_feats)
    
    # 3. 2D BEV Feature Map으로 Reshape (C, H, W)
    bev_feat_map = bev_flat.view(H, W, C).permute(2, 0, 1)
    return bev_feat_map

# --- 사용 예시 ---
pts_f = torch.randn(10000, 64)
bev_idx = torch.randint(0, 100*100, (10000,))
print("Camera BEV Feature Map Shape:", efficient_camera_to_bev_pooling(pts_f, bev_idx, (100, 100)).shape)
```

---

## 🛠️ Chapter 3: 완전 합성곱 BEV 융합 (Fully-Convolutional Fusion)

### 1. 요약
카메라 BEV 특징과 LiDAR BEV 특징을 채널 축으로 Concatenate한 후, 2D ResNet 스타일의 합성곱 백본에 입력하여 카메라 깊이 예측 오차로 발생하는 미세 공간 오정렬(Spatial Misalignment)을 융합 연산으로 흡수합니다.

### 2. 수식 및 파이썬 코드 설명

$$F_{\text{BEV}}^{fused} = \text{ConvNet}\Big( \text{Concat}\big( F_{\text{BEV}}^{cam}, \ F_{\text{BEV}}^{lidar} \big) \Big) \in \mathbb{R}^{C_{out} \times H \times W}$$

```python
import torch
import torch.nn as nn

class FullyConvolutionalBEVFusion(nn.Module):
    """
    Camera BEV 및 LiDAR BEV 채널 결합 및 공간 오정렬 보정 ConvNet
    """
    def __init__(self, cam_channels: int = 80, lidar_channels: int = 128, out_channels: int = 256):
        super().__init__()
        in_dim = cam_channels + lidar_channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, cam_bev: torch.Tensor, lidar_bev: torch.Tensor) -> torch.Tensor:
        """
        cam_bev: (B, C_cam, H, W)
        lidar_bev: (B, C_lidar, H, W)
        """
        # 채널 Concatenation
        fused_input = torch.cat([cam_bev, lidar_bev], dim=1)
        # 합성곱 보정 인코딩
        fused_bev = self.fusion_conv(fused_input)
        return fused_bev

# --- 사용 예시 ---
c_b = torch.randn(2, 80, 180, 180)
l_b = torch.randn(2, 128, 180, 180)
fusion_net = FullyConvolutionalBEVFusion()
print("최종 융합 BEV Feature Shape:", fusion_net(c_b, l_b).shape)
```

---

## 🛠️ Chapter 4: CenterPoint 탐지 헤드 및 Focal Loss 수식

### 1. 요약
융합된 BEV 특징 맵 위에서 3D 바운딩 박스 중심점을 탐지하기 위해 CenterPoint 스타일의 Gaussian Focal Loss를 적용합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{L}_{heat} = -\frac{1}{N} \sum_{u, v} \begin{cases} (1 - \hat{Y}_{u,v})^\alpha \log(\hat{Y}_{u,v}) & \text{if } Y_{u,v} = 1 \\ (1 - Y_{u,v})^\beta \hat{Y}_{u,v}^\alpha \log(1 - \hat{Y}_{u,v}) & \text{otherwise} \end{cases}$$

- **$\hat{Y}_{u,v}$**: 예측된 중심점 열지도 확률 ($0 \sim 1$)
- **$Y_{u,v}$**: 가우시안 평활화가 적용된 GT 중심점 열지도 ($0 \sim 1$)
- **$\alpha=2, \beta=4$**: Penalty-reduced Focal Loss 가중치

```python
import torch

def gaussian_focal_loss(pred_heatmap: torch.Tensor, gt_heatmap: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """
    3D CenterPoint 탐지 헤드용 Gaussian Focal Loss
    """
    pos_inds = gt_heatmap.eq(1.0).float()
    neg_inds = gt_heatmap.lt(1.0).float()
    
    pos_loss = torch.log(pred_heatmap + 1e-6) * torch.pow(1.0 - pred_heatmap, alpha) * pos_inds
    neg_loss = torch.log(1.0 - pred_heatmap + 1e-6) * torch.pow(pred_heatmap, alpha) * torch.pow(1.0 - gt_heatmap, beta) * neg_inds
    
    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        return -neg_loss
    else:
        return -(pos_loss + neg_loss) / num_pos

# --- 사용 예시 ---
p_hm = torch.sigmoid(torch.randn(1, 10, 100, 100))
g_hm = torch.zeros(1, 10, 100, 100)
g_hm[0, 2, 50, 50] = 1.0
print("CenterPoint Focal Loss:", gaussian_focal_loss(p_hm, g_hm).item())
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. nuScenes 3D 객체 탐지 리더보드 비교

| 방법 (Method) | 사용 센서 모달리티 | mAP ↑ | NDS ↑ | 지연시간 (Latency) |
|---|---|---|---|---|
| **CenterPoint** | LiDAR Only | 60.3 | 67.3 | 90 ms |
| **PointPainting** | Camera + LiDAR | 65.8 | 69.6 | 185 ms |
| **TransFusion** | Camera + LiDAR | 67.5 | 71.3 | 156 ms |
| **BEVFusion (Ours)** | **Camera + LiDAR** | **70.2** | **72.9** | **119 ms (압도적 SOTA)** |

- **결과**: TransFusion 대비 **mAP +2.7%**, **NDS +1.6%** 성능 향상과 동시에 지연시간을 **119ms로 단축**.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
BEVFusion은 카메라의 풍부한 의미론(Semantic)과 LiDAR의 정확한 3D 기하(Geometry)를 손실 없이 공유 BEV 공간에서 통합하고, 40배 가속된 BEV 풀링을 통해 멀티센서 멀티태스크 융합의 표준을 정립했습니다.

### 2. 한계점 및 아쉬운 점
- Precomputation은 카메라 파라미터가 고정(정적 캘리브레이션)이라는 가정에 의존하므로, 온라인 재캘리브레이션이 필요한 상황에는 그대로 적용하기 어렵다.
- 단순 채널 concatenation과 합성곱 인코더로 오정렬을 보정하는 방식이, 극단적인 깊이 추정 오류나 센서 고장 상황까지 견고하게 대응하는지는 추가 검증이 필요하다.

---

## 참고 자료
- [BEVFusion 공식 GitHub 저장소](https://github.com/mit-han-lab/bevfusion)
- [CVPR 2023 논문 (arXiv:2205.13542)](https://arxiv.org/abs/2205.13542)

*관련 논문: [BEVFormer](/posts/papers/bevformer/), [BEVDepth](/posts/papers/bevdepth/), [PointPillars](/posts/papers/pointpillars-fast-encoders-object-detection-point-clouds/), [CenterPoint](/posts/papers/centerpoint-center-based-3d-object-detection-and-tracking/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
