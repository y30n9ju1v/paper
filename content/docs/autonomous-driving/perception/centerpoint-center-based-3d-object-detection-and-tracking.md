---
title: "CenterPoint: Center-based 3D Object Detection and Tracking"
date: 2026-04-19T22:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "3D Object Detection"]
tags: ["LiDAR", "3D Detection", "Object Tracking", "Point Cloud", "BEV"]
year: 2021
references:
  - "pointpillars-fast-encoders-object-detection-point-clouds"
---

## 💡 한 줄 요약
LiDAR 포인트 클라우드에서 3D 객체를 바운딩 박스 대신 중심점(Center Point)으로 표현·탐지함으로써 방향 불변성(Rotation Invariance)을 확보하고, Velocity 헤드 예측 기반의 Greedy Closest-Point Matching 알고리즘으로 1ms 속도의 SOTA 3D 추적(Tracking) 성능을 달성했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Tianwei Yin, Xingyi Zhou, Philipp Krähenbühl (University of Texas at Austin)
- **발행년도**: 2021 (arXiv 2020, CVPR 2021, arXiv:2006.11275)
- **주요 기여점**:
  1. **Center-based 3D 객체 표현**: 방향 의존적 Anchor Box 대신 객체의 중심점(Keypoint Peak)을 탐지하는 2D Gaussian Heatmap 구조로 회전된 객체 탐지 정확도 대폭 향상.
  2. **다중 회귀 헤드 (Multi-Task Regression Heads)**: 중심점으로부터 서브 복셀 오프셋, 바닥 높이, 3D 크기 $(\log w, \log l, \log h)$, 회전각 $(\sin\alpha, \cos\alpha)$, 속도 $(v_x, v_y)$를 동시 회귀.
  3. **1ms 초고속 3D 추적 (Greedy Closest-Point Matching)**: 예측된 속도 $(v_x, v_y)$로 이전 프레임 위치를 역산하여 Kalman Filter 없이 1ms 만에 3D Multi-Object Tracking(MOT) 수행.
  4. **Two-Stage Refinement & IoU-guided Confidence**: 3D 박스의 5개 지점(4개 면 중심 + 객체 중심) 피처를 추출하여 3D IoU 기반 Score를 보정.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **Anchor-based 3D Detection (VoxelNet, SECOND, PointPillars)**: BEV 평면상에 축 정렬(Axis-Aligned)된 Anchor Box를 고정 배치하여 탐지.
2. **CenterPoint**: Anchor의 방향 제약을 완전히 철폐하고 2D CenterNet 아이디어를 3D BEV 공간으로 확장하여 표준 3D Detector 디팩토 스탠다드로 자리잡음.

### 기존 Anchor-based 방법의 한계점
- **방향 불연속성 (Rotation Dependency)**: 대각선으로 회전된 차량이나 보행자처럼 가로세로 비율이 극단적인 물체에 Anchor Box가 맞지 않아 탐지율 하락.
- **불필요한 초매개변수 오버헤드**: 회전각마다 Anchor를 중복 배치해야 하므로 연산량 증가 및 IoU 임계값 튜닝 복잡.

---

## 📑 목차
- Chapter 1: 3D Center Heatmap Target 생성 (Gaussian Rendering)
- Chapter 2: CenterPoint 다중 회귀 헤드 수식 $(\sin/\cos, \text{Velocity})$
- Chapter 3: 2단계 3D IoU-guided Score Refinement
- Chapter 4: 1ms Greedy Closest-Point 3D 추적 알고리즘
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: 3D Center Heatmap Target 생성

### 1. 요약
GT 3D 바운딩 박스의 중심 $(x_k, y_k)$를 BEV 표면에 투영하고, 박스 크기 $(w_k, l_k)$에 비례하는 2D 가우시안 반경 $\sigma_k$를 이용해 렌더링된 열지도 Target $Y_{u,v,c}$를 구성합니다.

### 2. 수식 및 파이썬 코드 설명

$$Y_{u,v,c} = \exp\left(-\frac{(u - x_k)^2 + (v - y_k)^2}{2\sigma_k^2}\right)$$

$$\sigma_k = \max\Big( f(w_k, l_k), \ \tau \Big) \quad (\tau = 2)$$

```python
import torch

def render_centerpoint_gaussian_heatmap(
    gt_centers: torch.Tensor, # (N, 2) GT 중심점 좌표 (x, y)
    gt_sizes: torch.Tensor,   # (N, 2) GT 박스 크기 (w, l)
    grid_shape: tuple,        # (H, W) BEV 그리드 해상도
    radius_min: float = 2.0
) -> torch.Tensor:
    """
    CenterPoint 학습을 위한 2D Gaussian Heatmap Target 생성
    """
    H, W = grid_shape
    heatmap = torch.zeros((H, W), dtype=torch.float32)
    
    ys = torch.arange(0, H).float().view(H, 1).repeat(1, W)
    xs = torch.arange(0, W).float().view(1, W).repeat(H, 1)
    
    for i in range(len(gt_centers)):
        cx, cy = gt_centers[i, 0], gt_centers[i, 1]
        w, l = gt_sizes[i, 0], gt_sizes[i, 1]
        
        # 반경 sigma 계산
        sigma = max(float(torch.sqrt(w * l) / 6.0), radius_min)
        
        # Gaussian Kernel 렌더링
        gaussian = torch.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * sigma**2))
        heatmap = torch.maximum(heatmap, gaussian)
        
    return heatmap

# --- 사용 예시 ---
centers = torch.tensor([[50.0, 50.0], [20.0, 80.0]])
sizes = torch.tensor([[4.0, 2.0], [1.5, 0.8]])
hm_gt = render_centerpoint_gaussian_heatmap(centers, sizes, (100, 100))
print("생성된 GT Heatmap Max:", hm_gt.max().item())
```

---

## 🛠️ Chapter 2: CenterPoint 다중 회귀 헤드 수식

### 1. 요약
중심점 Heatmap Peak에서 회전각 $\alpha$의 연속성을 보장하기 위해 $\sin\alpha, \cos\alpha$ 회귀를 사용하며, 프레임 간 이동 속도 $(v_x, v_y)$를 동시 예측합니다.

### 2. 수식 및 파이썬 코드 설명

$$\alpha = \text{atan2}(\sin\alpha, \cos\alpha)$$

$$\hat{\mathbf{v}} = (v_x, v_y) = \frac{\boldsymbol{\mu}_t - \boldsymbol{\mu}_{t-1}}{\Delta t}$$

```python
import torch

def decode_centerpoint_rot_and_velocity(
    sin_cos_pred: torch.Tensor, # (B, 2, H, W) -> sin(alpha), cos(alpha)
    vel_pred: torch.Tensor      # (B, 2, H, W) -> vx, vy
) -> tuple:
    """
    CenterPoint 회귀 헤드로부터 3D Yaw 회전각 alpha 및 속도 (vx, vy) 디코딩
    """
    sin_a = sin_cos_pred[:, 0]
    cos_a = sin_cos_pred[:, 1]
    
    # 1. Yaw angle 복원: alpha = atan2(sin, cos)
    yaw_angle = torch.atan2(sin_a, cos_a) # (B, H, W)
    
    # 2. 속도 벡터
    vx = vel_pred[:, 0]
    vy = vel_pred[:, 1]
    
    return yaw_angle, vx, vy

# --- 사용 예시 ---
sc_pred = torch.tensor([[[[0.7071]], [[0.7071]]]]) # sin(45deg), cos(45deg)
v_pred = torch.tensor([[[[1.2]], [[-0.5]]]])
yaw, vx, vy = decode_centerpoint_rot_and_velocity(sc_pred, v_pred)
print("복원된 Yaw 각도 (deg):", torch.rad2deg(yaw).item())
```

---

## 🛠️ Chapter 3: 2단계 3D IoU-guided Score Refinement

### 1. 요약
1단계 중심점 탐지 점수 $\hat{Y}_t$와 2단계 3D IoU-guided 보정 점수 $\hat{I}_t$의 기하평균을 최종 신뢰도 점수 $\hat{Q}_t$로 산출합니다.

### 2. 수식 및 파이썬 코드 설명

$$I_t = \min\Big(1, \ \max(0, \ 2 \times \text{IoU}_t - 0.5)\Big)$$

$$\hat{Q}_t = \sqrt{\hat{Y}_t \cdot \hat{I}_t}$$

```python
import torch

def compute_iou_guided_confidence_score(
    heatmap_score: torch.Tensor, # (N,) 1단계 Heatmap Peak 점수
    predicted_iou: torch.Tensor  # (N,) 2단계 예측 3D IoU
) -> torch.Tensor:
    """
    Two-Stage CenterPoint의 최종 confidence score Q_hat = sqrt(Y_hat * I_hat)
    """
    # IoU target 정규화 I_t = clamp(2 * IoU - 0.5, 0, 1)
    i_score = torch.clamp(2.0 * predicted_iou - 0.5, min=0.0, max=1.0)
    
    final_score = torch.sqrt(heatmap_score * i_score)
    return final_score

# --- 사용 예시 ---
y_score = torch.tensor([0.9, 0.8])
iou_pred = torch.tensor([0.85, 0.4]) # 0.85 IoU -> I=1.0 / 0.4 IoU -> I=0.3
print("최종 보정 Confidence Score:", compute_iou_guided_confidence_score(y_score, iou_pred))
```

---

## 🛠️ Chapter 4: 1ms Greedy Closest-Point 3D 추적 알고리즘

### 1. 요약
현재 프레임 탐지 객체의 중심 $\boldsymbol{\mu}_t$에서 예측 속도 $\mathbf{v}_t$를 차감하여 이전 위치 $\boldsymbol{\mu}_{t-1}^{pred} = \boldsymbol{\mu}_t - \mathbf{v}_t \Delta t$를 역산하고, 이전 Tracklet과의 L2 거리가 가장 가까운 것끼리 탐욕적(Greedy) 매칭합니다.

```python
import torch

def greedy_closest_point_3d_tracking(
    curr_centers: torch.Tensor,  # (N, 2) 현재 프레임 탐지 중심점
    curr_velocities: torch.Tensor,# (N, 2) 예측 속도 (vx, vy)
    prev_tracklets: torch.Tensor, # (M, 2) 이전 프레임 Tracklet 위치
    dt: float = 0.5,             # 프레임 간 시간 간격 (sec)
    dist_threshold: float = 2.0  # 최대 매칭 거리 임계값 (m)
) -> dict:
    """
    Velocity 기반 1ms Greedy Closest-Point 3D MOT 매칭
    """
    # 1. 현재 중심점을 이전 프레임 위치로 역투영: mu_{t-1} = mu_t - v_t * dt
    pred_prev_centers = curr_centers - curr_velocities * dt # (N, 2)
    
    # 2. L2 거리 행렬 계산 (N, M)
    dist_matrix = torch.cdist(pred_prev_centers, prev_tracklets)
    
    # 3. Greedy Matching
    matched_pairs = []
    for i in range(len(curr_centers)):
        min_dist, min_idx = torch.min(dist_matrix[i], dim=0)
        if min_dist.item() < dist_threshold:
            matched_pairs.append((i, min_idx.item()))
            dist_matrix[:, min_idx] = 1e6 # 중복 매칭 방지
            
    return {"matches": matched_pairs}

# --- 사용 예시 ---
curr_c = torch.tensor([[10.0, 10.0], [30.0, 30.0]])
curr_v = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
prev_t = torch.tensor([[9.0, 10.0], [50.0, 50.0]])
print("1ms 추적 매칭 결과:", greedy_closest_point_3d_tracking(curr_c, curr_v, prev_t, dt=0.5))
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. nuScenes & Waymo 3D 탐지/추적 성능 리더보드

| 벤치마크 (Benchmark) | 모델 (Method) | 탐지 성능 (mAP / NDS) ↑ | 추적 성능 (AMOTA) ↑ | 추적 지연시간 |
|---|---|---|---|---|
| **Waymo (Level 2)** | PointPillars | 55.6 mAP | - | - |
| **Waymo (Level 2)** | **CenterPoint-Voxel** | **72.2 mAP (+16.6%)** | - | - |
| **nuScenes (Test)** | AB3D (Kalman Filter) | - | 15.1 | 73 ms |
| **nuScenes (Test)** | **CenterPoint (Ours)** | **58.0 mAP / 65.5 NDS** | **63.8 (+8.8%)** | **1 ms (73배 가속)** |

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
CenterPoint는 3D 탐지를 2D Gaussian Center Peak 추정 문제로 명쾌하게 전환함으로써 회전 이방성 문제를 해결하고, Velocity 기반 1ms 추적을 완성한 현대 3D Perception의 디팩토 표준 파이프라인입니다.

### 2. 한계점 및 아쉬운 점
- LiDAR 전용이라 카메라 전용 또는 카메라+LiDAR 융합 탐지에는 직접 적용할 수 없다.
- PointPillars 백본 사용 시 보행자처럼 1픽셀 크기의 객체에서는 2단계 정제 효과가 양자화 한계로 제한적이다.
- velocity 예측이 비선형 기동(급회전 등)에서는 부정확할 수 있어, 실제 복잡한 도심 시나리오에서의 추적 신뢰성은 추가 검증이 필요하다.

---

## 참고 자료
- [CenterPoint 공식 GitHub 저장소](https://github.com/tianweiy/CenterPoint)
- [CVPR 2021 논문 (arXiv:2006.11275)](https://arxiv.org/abs/2006.11275)

*관련 논문: [PointNet](/posts/papers/pointnet-deep-learning-on-point-sets-for-3d-classification-and-segmentation/), [PointPillars](/posts/papers/pointpillars-fast-encoders-object-detection-point-clouds/), [BEVFusion](/posts/papers/bevfusion-multi-task-multi-sensor-fusion/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/), [Waymo Open Dataset](/posts/papers/waymo-open-dataset/)*
