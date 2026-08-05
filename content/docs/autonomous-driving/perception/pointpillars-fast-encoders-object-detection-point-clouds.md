---
title: "PointPillars: Fast Encoders for Object Detection from Point Clouds"
date: 2026-04-19T22:30:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "3D Object Detection"]
tags: ["LiDAR", "Point Cloud", "3D Detection", "BEV", "Real-time"]
year: 2019
references:
  - "pointnet-deep-learning-on-point-sets-for-3d-classification-and-segmentation"
---

## 💡 한 줄 요약
LiDAR 3D 포인트 클라우드를 z축 구분 없는 수직 기둥(Pillar)으로 조직화하고, Simplified PointNet 인코딩 후 2D Pseudo-Image로 Scatter 사영하여 무거운 3D 합성곱(Convolution) 없이 2D CNN만으로 초고속(62Hz) SOTA 3D 객체 탐지를 실현했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom (nuTonomy / APTIV)
- **발행년도**: 2019 (arXiv:1812.05784, CVPR 2019)
- **주요 기여점**:
  1. **3D Convolution 완전 제거**: 무거운 3D 복셀 대신 2D Pillar 수직 기둥 인코더를 제안하여 기존 VoxelNet/SECOND 대비 인코딩 지연시간을 146배 가속 (1.3ms).
  2. **9차원 포인트 장식 (Point Decoration)**: 단순 3D 좌표 외에 Pillar 내부 산술 평균 좌표 오차 $(x_c, y_c, z_c)$ 및 Pillar 중심 오프셋 $(x_p, y_p)$을 기하학적 임베딩으로 추가.
  3. **2D Pseudo-Image Scatter 수식**: PointNet으로 압축된 $(C, P)$ Pillar 특징 벡터를 원래 2D BEV 그리드 상의 $(C, H, W)$ 이미지 텐서로 즉시 매핑하여 2D CNN 백본 연결.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **Hand-crafted BEV Feature (PIXOR, Complex-YOLO)**: 수작업 피처로 2D CNN을 구동하나 수직 기하정보 손실 발생.
2. **Voxel-based 3D Conv (VoxelNet, SECOND)**: 3D 복셀 표현으로 세밀한 기하를 잡으나 3D Conv 연산으로 추론 속도가 4.4Hz~20Hz에 그침.
3. **PointPillars**: 속도와 정확도의 최적 트레이드오프(62Hz 실시간 처리)를 정립한 LiDAR 3D Perception의 거두.

---

## 📑 목차
- Chapter 1: 9D Point Feature Decoration (포인트 장식 수식)
- Chapter 2: Simplified PointNet 및 Pseudo-Image Scatter Pooling
- Chapter 3: 3D Bounding Box Target Regression & Sine Heading
- Chapter 4: Multi-Task 손실 함수 (Focal Loss + Direction Loss)
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: 9D Point Feature Decoration

### 1. 요약
각 LiDAR 포인트 $l_i$에 대해 3D 좌표 $(x, y, z)$, 반사율 $r$, Pillar 내 평균과의 차이 $(x_c, y_c, z_c)$, Pillar Center 오프셋 $(x_p, y_p)$을 9차원 벡터로 풍부하게 장식(Decorate)합니다.

### 2. 수식 및 파이썬 코드 설명

$$l_i = [x, \ y, \ z, \ r, \ x_c, \ y_c, \ z_c, \ x_p, \ y_p]^T \in \mathbb{R}^9$$

$$x_c = x - \bar{x}, \quad y_c = y - \bar{y}, \quad z_c = z - \bar{z}$$

$$x_p = x - x_{\text{pillar}}, \quad y_p = y - y_{\text{pillar}}$$

```python
import torch

def decorate_point_cloud_into_pillars(
    points_3d: torch.Tensor, # (N, 4) -> x, y, z, reflectance
    pillar_size: float = 0.16,
    max_points_per_pillar: int = 100,
    max_pillars: int = 12000
) -> torch.Tensor:
    """
    PointPillars: 3D 포인트를 9차원 피처 텐서 (D=9, P, N)로 장식
    """
    N_pts, _ = points_3d.shape
    
    # 1. Pillar 좌표 인덱스 계산 (x_pillar, y_pillar)
    pillar_x = torch.floor(points_3d[:, 0] / pillar_size).long()
    pillar_y = torch.floor(points_3d[:, 1] / pillar_size).long()
    
    # 2. Pillar 중심 실측 좌표
    x_p = points_3d[:, 0] - (pillar_x.float() * pillar_size + pillar_size / 2.0)
    y_p = points_3d[:, 1] - (pillar_y.float() * pillar_size + pillar_size / 2.0)
    
    # 3. Pillar 내 산술 평균 중심 오차 (x_c, y_c, z_c)
    mean_x = points_3d[:, 0].mean()
    mean_y = points_3d[:, 1].mean()
    mean_z = points_3d[:, 2].mean()
    
    x_c = points_3d[:, 0] - mean_x
    y_c = points_3d[:, 1] - mean_y
    z_c = points_3d[:, 2] - mean_z
    
    # 4. 9차원 장식 벡터 결합
    decorated_pts = torch.stack([
        points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], points_3d[:, 3],
        x_c, y_c, z_c, x_p, y_p
    ], dim=-1) # (N, 9)
    
    return decorated_pts

# --- 사용 예시 ---
pts_raw = torch.randn(1000, 4)
dec_pts = decorate_point_cloud_into_pillars(pts_raw)
print("장식된 포인트 9차원 피처 Shape:", dec_pts.shape)
```

---

## 🛠️ Chapter 2: Simplified PointNet 및 Pseudo-Image Scatter Pooling

### 1. 요약
$(D=9, P, N)$ 텐서를 1D Conv/Linear $\to$ BN $\to$ ReLU $\to$ MaxPool을 적용해 $(C, P)$ 로 압축한 후, 원래 BEV 그리드 픽셀 위치로 Scatter 사영하여 $(C, H, W)$ 2D Pseudo-Image를 생성합니다.

### 2. 수식 및 파이썬 코드 설명

$$f_{\text{pillar}} = \text{MaxPool}_{N}\Big( \text{ReLU}\left( \text{BN}( \mathbf{W} \cdot l ) \right) \Big) \in \mathbb{R}^{C \times P}$$

$$F_{\text{pseudo-image}} = \text{Scatter}\Big( f_{\text{pillar}}, \ (x_{\text{pillar}}, y_{\text{pillar}}) \Big) \in \mathbb{R}^{C \times H \times W}$$

```python
import torch
import torch.nn as nn

class PointPillarEncoder(nn.Module):
    """
    Simplified PointNet 인코더 및 2D Pseudo-Image Scatter 모듈
    """
    def __init__(self, in_dim: int = 9, out_channels: int = 64):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, pillar_tensor: torch.Tensor, pillar_coords: torch.Tensor, grid_shape: tuple) -> torch.Tensor:
        """
        pillar_tensor: (P, N, D=9)
        pillar_coords: (P, 2) -> (x_idx, y_idx)
        """
        P, N, D = pillar_tensor.shape
        H, W = grid_shape
        
        # 1. Simplified PointNet
        x = self.linear(pillar_tensor.view(-1, D))
        x = self.bn(x).view(P, N, -1)
        x = torch.relu(x)
        
        # MaxPool over N points in each pillar -> (P, C)
        feat_p, _ = torch.max(x, dim=1)
        
        # 2. Scatter to 2D Pseudo-Image (C, H, W)
        C = feat_p.shape[-1]
        pseudo_img = torch.zeros((C, H, W), device=pillar_tensor.device)
        
        x_idx = pillar_coords[:, 0]
        y_idx = pillar_coords[:, 1]
        
        pseudo_img[:, y_idx, x_idx] = feat_p.T
        return pseudo_img.unsqueeze(0) # (1, C, H, W)

# --- 사용 예시 ---
p_tensor = torch.randn(500, 32, 9)
p_coords = torch.randint(0, 100, (500, 2))
encoder = PointPillarEncoder()
print("생성된 2D Pseudo-Image Shape:", encoder(p_tensor, p_coords, (100, 100)).shape)
```

---

## 🛠️ Chapter 3: 3D Anchor Target Regression & Multi-Task Loss

### 1. 요약
3D 위치 오차 $(\Delta x, \Delta y, \Delta z)$, 3D 크기 로그 오차 $(\Delta w, \Delta l, \Delta h)$, 각도 $\Delta\theta = \sin(\theta^{gt} - \theta^a)$를 Focal Loss 및 Smooth L1 Loss로 공동 최적화합니다.

### 2. 수식 및 파이썬 코드 설명

$$\Delta x = \frac{x^{gt} - x^a}{d^a}, \quad \Delta y = \frac{y^{gt} - y^a}{d^a}, \quad d^a = \sqrt{(w^a)^2 + (l^a)^2}$$

$$\Delta w = \log \frac{w^{gt}}{w^a}, \quad \Delta l = \log \frac{l^{gt}}{l^a}, \quad \Delta h = \log \frac{h^{gt}}{h^a}, \quad \Delta\theta = \sin(\theta^{gt} - \theta^a)$$

$$\mathcal{L} = \frac{1}{N_{pos}} \left( \beta_{loc} \mathcal{L}_{loc} + \beta_{cls} \mathcal{L}_{cls} + \beta_{dir} \mathcal{L}_{dir} \right)$$

```python
import torch
import torch.nn.functional as F

def compute_pointpillars_loss(
    cls_preds: torch.Tensor, # (N_anchors, Num_classes)
    box_preds: torch.Tensor, # (N_anchors, 7)
    dir_preds: torch.Tensor, # (N_anchors, 2)
    cls_targets: torch.Tensor,
    box_targets: torch.Tensor,
    dir_targets: torch.Tensor
) -> torch.Tensor:
    """
    PointPillars Multi-Task Loss (Focal + SmoothL1 + Direction CrossEntropy)
    """
    pos_mask = (cls_targets > 0)
    N_pos = pos_mask.sum().float().clamp(min=1.0)
    
    # 1. Focal Loss (Classification)
    p = torch.sigmoid(cls_preds)
    focal_weights = 0.25 * torch.pow(1.0 - p, 2.0)
    loss_cls = F.binary_cross_entropy_with_logits(cls_preds, cls_targets.float(), reduction='none') * focal_weights
    loss_cls = loss_cls.sum() / N_pos
    
    # 2. Smooth L1 Loss (Box Regression)
    loss_box = F.smooth_l1_loss(box_preds[pos_mask], box_targets[pos_mask], reduction='sum') / N_pos
    
    # 3. Direction Loss (Softmax CE)
    loss_dir = F.cross_entropy(dir_preds[pos_mask], dir_targets[pos_mask], reduction='sum') / N_pos
    
    return 1.0 * loss_cls + 2.0 * loss_box + 0.2 * loss_dir

# --- 사용 예시 ---
c_p, b_p, d_p = torch.randn(100, 1), torch.randn(100, 7), torch.randn(100, 2)
c_t, b_t, d_t = torch.randint(0, 2, (100, 1)), torch.randn(100, 7), torch.randint(0, 2, (100,))
print("PointPillars 총 손실:", compute_pointpillars_loss(c_p, b_p, d_p, c_t, b_t, d_t).item())
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. KITTI 3D / BEV 객체 탐지 리더보드 비교

| 알고리즘 (Method) | 입력 모달리티 | 처리 속도 (Hz) ↑ | Car BEV mAP ↑ | Car 3D mAP ↑ |
|---|---|---|---|---|
| **VoxelNet** | LiDAR | 4.4 Hz (느림) | 79.26 | 65.11 |
| **SECOND** | LiDAR | 20.0 Hz | 79.37 | 76.48 |
| **PointPillars (Ours)** | **LiDAR** | **62.0 Hz (실시간 SOTA)** | **86.10 (+6.73%)** | **74.99** |
| **PointPillars + TensorRT** | **LiDAR** | **105.0 Hz (초고속)** | **86.10** | **74.99** |

- **결과**: 3D Conv를 탈피한 Pillar 인코더 덕분에 인코딩 시간을 **1.3ms로 146배 가속**하고 **62Hz 실시간 처리** 달성.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
PointPillars는 3D 포인트 클라우드를 2D Pseudo-Image 사영계로 전환하여 무거운 3D Conv 없이 2D CNN만으로 초고속(62Hz) SOTA 3D 객체 탐지를 구현한 자율주행 3D Perception의 기초 필독 논문입니다.

### 2. 한계점 및 아쉬운 점
- Pillar 단위 처리로 z축 정보(높이)가 MaxPool로 집약되어 세밀한 수직 구조 표현이 불가능하다.
- 보행자·자전거 탐지 성능이 차량보다 낮다(작은 객체, 희소 포인트).
- 카메라와의 융합을 고려하지 않은 LiDAR 전용 설계로, 카메라만이 제공하는 색·질감 정보를 활용하지 못한다.

---

## 참고 자료
- [PointPillars 공식 GitHub 저장소](https://github.com/nutonomy/second.pytorch)
- [CVPR 2019 논문 (arXiv:1812.05784)](https://arxiv.org/abs/1812.05784)

*관련 논문: [PointNet](/posts/papers/pointnet-deep-learning-on-point-sets-for-3d-classification-and-segmentation/), [CenterPoint](/posts/papers/centerpoint-center-based-3d-object-detection-and-tracking/), [BEVFusion](/posts/papers/bevfusion-multi-task-multi-sensor-fusion/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
