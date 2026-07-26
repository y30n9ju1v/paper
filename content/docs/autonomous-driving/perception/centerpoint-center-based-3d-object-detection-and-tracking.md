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
LiDAR 포인트 클라우드에서 3D 객체를 바운딩 박스 대신 중심점(center point)으로 표현·탐지함으로써 방향 불변성을 확보하고, velocity 예측 기반의 greedy closest-point matching으로 Kalman filter 없이도 SOTA 3D 추적 성능을 달성한다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Tianwei Yin, Xingyi Zhou, Philipp Krähenbühl (UT Austin)
- **발행년도**: 2021 (arXiv 2020, CVPR 2021, arXiv:2006.11275)
- **주요 기여점**:
  1. 3D 객체 탐지를 keypoint estimation 문제로 재정의하여, 클래스별 heatmap peak로 객체 중심을 예측하는 Center-based 표현 제안
  2. 중심점 위치에서 sub-voxel offset, height-above-ground, 3D size, rotation, velocity를 회귀하는 통합 헤드 설계
  3. Velocity Head로 예측한 중심 이동량을 이용해 greedy closest-point matching으로 추적을 단순화, Kalman filter(73ms) 대비 73배 빠른 1ms 추적 달성
  4. 예측 박스의 5개 지점(4개 면 중심 + 객체 중심) 피처로 confidence를 재보정하는 Two-Stage Refinement와 IoU-guided confidence score 도입
  5. Waymo·nuScenes 두 벤치마크에서 당시 SOTA 달성, anchor-based 대비 회전된 객체에서 특히 큰 폭의 성능 향상 입증

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: 2D 객체 탐지는 anchor 기반 방법에서 CenterNet 같은 keypoint(중심점) 기반 방법으로 발전했습니다. 3D 탐지 분야에서는 VoxelNet, PointPillars 등이 포인트 클라우드를 voxel/pillar로 인코딩하는 3D 인코더를 발전시켜 왔지만, 탐지 헤드는 여전히 axis-aligned anchor box에 의존하고 있었습니다.
- **기존 한계점**:
  1. 기존 3D 탐지기는 axis-aligned 바운딩 박스(anchor)를 사용하여, 차량이 회전하거나 자전거·보행자처럼 세로로 긴 객체를 탐지할 때 anchor가 맞지 않아 성능이 저하됨
  2. 방향별로 anchor 수가 늘어나고 IoU threshold 튜닝이 필요해 복잡도가 증가하고 false positive가 증가함
  3. 기존 3D 추적은 Kalman filter + Mahalanobis 거리 등 복잡한 연산이 필요함
- **이 논문의 접근 방식**: 객체를 3D 바운딩 박스 대신 중심점(heatmap peak)으로 표현하여 방향에 무관한 탐지를 가능하게 하고, 중심점 velocity 예측으로 추적을 greedy closest-point matching으로 단순화합니다.

## 📑 목차
- Section 1: Introduction
- Section 2: Related Work
- Section 3: Preliminaries (2D CenterNet, 3D Detection 정의)
- Section 4: CenterPoint (Center Heatmap Head / Regression Heads / Velocity Head와 추적 / Two-Stage CenterPoint)
- Section 5: Experiments (Main Results / Ablation Studies)

## 🛠️ Section 3: Preliminaries

**요약**
CenterPoint의 직접적 전신인 2D CenterNet은 이미지 기반 객체 탐지를 키포인트 추정(keypoint estimation) 문제로 재정의합니다. 입력 이미지에서 $K$개 클래스별 heatmap $\hat{Y} \in [0,1]^{w \times h \times K}$을 예측하고, heatmap의 각 local maximum(peak)이 탐지된 객체의 중심이 됩니다. 각 탐지 객체에 대해 크기, 오프셋 등 속성을 중심 위치에서 회귀합니다.

3D Detection 문제는 포인트 클라우드 $\mathcal{P} = \{(x, y, z, r)_i\}$에서 3D 바운딩 박스 집합 $\mathcal{B} = \{b_k\}$를 예측하는 것으로 정의됩니다. 각 박스 $b = (u, v, d, w, l, h, \alpha)$는 BEV 상의 중심 위치 $(u,v)$, 지면으로부터의 높이 $d$, 크기 $(w,l,h)$, yaw 회전각 $\alpha$로 구성됩니다. 현대 3D 탐지기는 VoxelNet 또는 PointPillars 같은 3D 인코더로 포인트 클라우드를 voxel/pillar로 양자화하여 map-view 피처맵 $\mathbf{M} \in \mathbb{R}^{W \times L \times F}$을 생성합니다.

**핵심 개념**
- **2D CenterNet**: 객체를 keypoint(중심점)로 표현하는 2D 탐지기. CenterPoint의 3D 확장의 기반
- **$(u,v,d,w,l,h,\alpha)$**: 3D 바운딩 박스를 표현하는 7개 파라미터
- **VoxelNet / PointPillars**: 포인트 클라우드를 map-view 피처맵으로 변환하는 3D 인코더

## 🛠️ Section 4: CenterPoint

**요약**
전체 파이프라인은 3D Backbone(VoxelNet 또는 PointPillars) → Detection Head(Center Heatmap Head, Sub-voxel Offset Head, Height-above-ground Head, 3D Size Head, Rotation Head, Velocity Head) → (선택적) Two-Stage 정제로 구성됩니다.

**Center Heatmap Head**: 목표는 탐지된 객체의 중심 위치에 heatmap peak를 생성하는 것입니다. 훈련 시, 3D 바운딩 박스 중심점을 BEV에 투영하고 2D Gaussian 커널로 렌더링하여 GT heatmap을 만듭니다. Focal loss로 최적화합니다(heatmap이 매우 희소하므로 배경 억제가 중요).

**Regression Heads**: 각 탐지 객체에 대해 중심점 위치에서 sub-voxel offset($\mathbb{R}^2$), height-above-ground($\mathbb{R}$), 3D size($\mathbb{R}^3$, log-scale), rotation($\mathbb{R}^2$, $\sin/\cos$), velocity($\mathbb{R}^2$)를 회귀합니다. rotation을 $\sin/\cos$로 표현하는 이유는, 각도는 $0°$와 $360°$가 같지만 수치상 멀어서 회귀가 불안정하기 때문입니다. 모든 회귀 출력은 L1 loss로 학습합니다.

**Velocity Head와 추적**: velocity $\mathbf{v} \in \mathbb{R}^2$는 현재 프레임과 직전 프레임 사이의 중심점 이동량입니다. 추적 알고리즘(greedy closest-point matching)은 현재 프레임 탐지 결과에서 velocity로 중심을 이전 프레임으로 역투영하고, 이전 프레임 tracklet과 최근접점 거리로 매칭합니다. 매칭 실패 tracklet은 최대 $T=3$ 프레임 유지 후 삭제되며, 추적 시간은 1ms로 Kalman filter(73ms) 대비 73배 빠릅니다. 객체가 점이면 추적도 점 간 거리 매칭으로 충분하며, 박스 IoU 매칭보다 회전·크기에 무관하게 robust합니다.

**Two-Stage CenterPoint**: 1단계(one-stage)에서 중심점 위치만으로 속성을 추론하면, 센서가 객체 측면만 보는 경우(자율주행에서 흔한 상황) 중심점 피처가 불충분할 수 있습니다. 2단계 보정에서는 예측된 3D 바운딩 박스의 4개 outward-facing 면 중심점 + 객체 중심점(총 5점)에서 backbone map-view 피처맵을 bilinear interpolation으로 추출하고, 이를 concat하여 MLP를 통과시켜 IoU-guided confidence score와 box refinement를 얻습니다.

**핵심 개념**
- **Gaussian 렌더링**: GT 중심점 주변에 박스 크기 비례 Gaussian 분포로 supervision 영역 확장. 작은 객체도 안정적 학습
- **Greedy Closest-Point Matching**: velocity로 역투영한 중심점과 이전 tracklet 중심점 간 최소 거리로 추적. 1ms의 극단적 속도
- **IoU-guided Score**: 2단계 confidence를 박스-GT IoU로 직접 지도. NMS 없이도 품질 반영 가능

**수식 예제**

$$\sigma = \max(f(w \cdot l),\ \tau), \quad \tau = 2$$

**수식 설명**
- **$w, l$**: 객체의 가로·세로 크기 (BEV)
- **$f(\cdot)$**: CornerNet의 반경 함수 — 박스 크기에 비례해 Gaussian을 넓게 설정
- **$\tau = 2$**: 최소 Gaussian 반경 (작은 객체 보호)
- **직관**: 큰 차량일수록 넓은 supervision 영역 → 학습 안정성 향상

**수식 예제**

$$I = \min(1,\ \max(0,\ 2 \times IoU_t - 0.5))$$

$$L_{score} = -I_t \log(\hat{I}_t) - (1-I_t)\log(1-\hat{I}_t)$$

**수식 설명**
- **$IoU_t$**: t번째 제안 박스와 GT 간의 3D IoU
- **$I_t$**: IoU를 $[0,1]$ 범위로 정규화한 confidence target (IoU = 0.5 → $I$ = 0, IoU = 1.0 → $I$ = 1)
- **$\hat{I}_t$**: 예측 confidence
- **직관**: NMS 없이도 confidence score가 박스 품질을 직접 반영하게 학습

**수식 예제**

$$\hat{Q}_t = \sqrt{\hat{Y}_t \cdot \hat{I}_t}$$

**수식 설명**
- 1단계 heatmap 점수 $\hat{Y}_t$와 2단계 IoU 점수 $\hat{I}_t$의 기하평균으로 최종 confidence를 계산

## 📊 주요 실험 및 결과 (Experiments & Results)
- **사용 데이터셋 / 벤치마크**: Waymo Open Dataset(3D Detection), nuScenes(3D Detection, 3D Tracking)

**Waymo Open Dataset — 3D Detection (test set, Level 2)**

| 방법 | Vehicle mAP | Vehicle mAPH | Ped. mAP | Ped. mAPH |
|---|---|---|---|---|
| PointPillars | 55.6 | 55.1 | 45.1 | — |
| PV-RCNN | 65.1 | 64.7 | — | — |
| **CenterPoint-Voxel (ours)** | **72.2** | **71.8** | **72.2** | **66.4** |

**nuScenes — 3D Detection (test set)**

| 방법 | mAP | NDS | PKL |
|---|---|---|---|
| PointPillars | 40.1 | 55.0 | 1.00 |
| CBGS (이전 1위) | 52.8 | 63.3 | 0.77 |
| **CenterPoint (ours)** | **58.0** | **65.5** | **0.69** |

**nuScenes — 3D Tracking (test set)**

| 방법 | AMOTA |
|---|---|
| AB3D | 15.1 |
| Chiu et al. | 55.0 |
| **CenterPoint (ours)** | **63.8** |

**Ablation: Center-based vs Anchor-based (Waymo validation, Level 2 mAPH)**

| Encoder | 방법 | Vehicle | Pedestrian | 평균 |
|---|---|---|---|---|
| VoxelNet | Anchor-based | 66.1 | 54.4 | 60.3 |
| VoxelNet | **Center-based** | **66.5** | **62.7** | **64.6** |
| PointPillars | Anchor-based | 64.1 | 50.8 | 57.5 |
| PointPillars | **Center-based** | **66.5** | **57.4** | **62.0** |

단순히 anchor → center로 표현만 바꿔도 3-4 mAPH 향상됩니다. 회전각별 성능에서는 회전된 객체(30°-45°)에서 center-based가 크게 우세하여 anchor의 방향 의존성 한계를 입증합니다. 2단계 정제는 추가 비용 5-6ms로 1.8 mAPH 향상을 가져옵니다.

## 💡 결론 및 시사점 (Conclusion & Insights)
CenterPoint는 3D 탐지를 keypoint estimation으로 재정의하여 방향 불변성과 단순한 파이프라인을 확보했습니다. anchor → center 전환만으로 3-4 mAPH 향상을 이끌어냈으며, 회전 객체에서 특히 강한 성능을 보였습니다. 추적을 1ms greedy matching으로 단순화하면서도 SOTA 추적 성능을 달성했고, VoxelNet·PointPillars 어느 backbone과도 호환되는 plug-in head로 설계되었습니다.

- BEVFusion의 LiDAR 브랜치 선행 연구로, BEVFusion은 카메라 BEV + LiDAR BEV를 융합할 때 LiDAR BEV 처리에 CenterPoint 구조를 직접 차용합니다.
- E2E 계획 논문들이 탐지 결과를 downstream으로 받을 때 CenterPoint를 perception 기준으로 사용하는 경우가 많습니다.
- LiDAR 기반 3D 탐지의 사실상 표준 baseline으로, 이후 대부분의 비교 논문에서 참조됩니다.
- **한계점 및 아쉬운 점**: LiDAR 전용이라 카메라 전용 또는 카메라+LiDAR 융합 탐지에는 직접 적용할 수 없습니다. PointPillars 백본 사용 시 보행자처럼 1픽셀 크기의 객체에서는 2단계 정제 효과가 양자화 한계로 제한적입니다. 또한 velocity 예측이 비선형 기동(급회전 등)에서는 부정확할 수 있어, 실제 복잡한 도심 시나리오에서의 추적 신뢰성은 추가 검증이 필요합니다.

---

*관련 논문: [PointNet](/posts/papers/pointnet-deep-learning-on-point-sets-for-3d-classification-and-segmentation/), [PointPillars](/posts/papers/pointpillars-fast-encoders-object-detection-point-clouds/), [BEVFusion](/posts/papers/bevfusion-multi-task-multi-sensor-fusion/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/), [Waymo Open Dataset](/posts/papers/waymo-open-dataset/)*
