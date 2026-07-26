---
title: "인지(Perception) 핵심 개념 정리"
date: 2026-07-26T16:10:00+09:00
draft: false
weight: 30
categories: ["Glossary"]
---

이 폴더의 인지(perception) 논문들 — BEV 탐지([Lift-Splat-Shoot](/posts/papers/lift-splat-shoot/), [BEVFormer](/posts/papers/bevformer/), [BEVDepth](/posts/papers/bevdepth/), [BEVFusion](/posts/papers/bevfusion-multi-task-multi-sensor-fusion/), [DETR3D](/posts/papers/detr3d-3d-object-detection-multi-view-images/), [CenterPoint](/posts/papers/centerpoint-center-based-3d-object-detection-and-tracking/), [PointPillars](/posts/papers/pointpillars-fast-encoders-object-detection-point-clouds/)), HD 맵([MapTR](/posts/papers/maptr-structured-modeling-online-vectorized-hd-map-construction/), [VectorMapNet](/posts/papers/vectormapnet-end-to-end-vectorized-hd-map-learning/), [StreamMapNet](/posts/papers/streammapnet-streaming-mapping-network-vectorized-online-hd-map-construction/)), 3D Occupancy([MonoScene](/posts/papers/monoscene-monocular-3d-semantic-scene-completion/), [SurroundOcc](/posts/papers/surroundocc/), [TPVFormer](/posts/papers/tpvformer-tri-perspective-view-3d-semantic-occupancy/), [Occ3D](/posts/papers/occ3d-large-scale-3d-occupancy-prediction-benchmark/), [GaussianWorld](/posts/papers/gaussianworld-gaussian-world-model-for-streaming-3d-occupancy-prediction/)) — 에서 반복적으로 등장하는 핵심 개념을 정리합니다.

## 기초 개념 (사전지식)

논문 본문에서는 당연히 알고 있다고 가정하고 넘어가는 자율주행 인지 배경지식입니다.

- **인지(Perception)**: 자율주행 스택에서 센서 데이터를 입력받아 "주변에 무엇이 어디에 있는가"를 파악하는 단계. 이후 예측(Prediction, 그것들이 어떻게 움직일지)과 계획(Planning, 자차가 어떻게 움직일지) 단계로 이어짐
- **센서 종류**: **카메라**(색상·질감에 강하지만 깊이 추정이 어려움), **LiDAR**(레이저로 거리를 측정해 3D 위치가 정확하지만 데이터가 희소하고 의미 정보가 약함), **레이더(Radar)**(원거리 감지와 도플러 효과 기반 속도 측정에 강함) — 각자 다른 장단점이 있어 서로 보완적으로 융합(sensor fusion)됨
- **Ego(에고) / Ego Frame**: 자율주행 차량 자신을 가리키는 말. Ego frame은 그 차량을 원점으로 삼는 좌표계 — "내 차 기준으로 저 차가 앞 10m, 왼쪽 2m에 있다"는 식의 표현
- **바운딩 박스(Bounding Box)**: 객체를 감싸는 사각형(2D) 또는 직육면체(3D) 상자로, 보통 중심 위치·크기·회전(방향)으로 표현. "이 차가 대략 여기서부터 여기까지 차지하고 있다"는 근사적 표현
- **분류(Classification) vs 회귀(Regression)**: 분류는 "이것이 차인가 보행자인가"처럼 정해진 카테고리 중 하나를 고르는 문제, 회귀는 "이 박스의 중심 좌표는 몇 미터인가"처럼 연속적인 숫자 값을 맞히는 문제. 대부분의 탐지 모델은 두 문제를 동시에 풂
- **NMS (Non-Maximum Suppression, 비최대 억제)**: 같은 객체에 대해 여러 개의 중복된 박스가 예측됐을 때, 가장 확신도 높은 것만 남기고 나머지를 제거하는 후처리 — DETR 계열 모델들이 "NMS가 필요 없다"고 강조하는 것은 이 후처리 없이 곧바로 깔끔한 결과를 낸다는 뜻
- **IoU (Intersection over Union)**: 예측 박스와 정답 박스가 겹치는 면적을 전체 면적으로 나눈 값(0~1). 1에 가까울수록 정확히 겹친다는 뜻이며, 탐지 성능 평가와 정답 매칭(어떤 예측이 어떤 정답과 짝인지)에 널리 쓰임
- **Multi-view / Surround-view (멀티뷰/서라운드뷰)**: 차량에 달린 여러 대의 카메라(보통 6대 안팎)로 동시에 360도를 촬영하는 방식 — 이 여러 이미지를 어떻게 하나의 일관된 3D/BEV 표현으로 합치는지가 이 폴더 논문들의 공통 주제

## BEV(Bird's-Eye-View) 공통 개념

- **BEV (Bird's-Eye-View)**: 차량 위에서 내려다본 2D 평면 표현. 여러 카메라·센서의 정보를 하나의 공간에 통합하고, 객체의 실제 위치·방향을 직접 표현할 수 있어 자율주행 인지의 표준 좌표계로 쓰임
- **Depth Estimation 의존 문제**: 2D 이미지에서 BEV를 만들려면 각 픽셀의 깊이를 추정해야 하는데, 이 추정 오차가 BEV 품질에 그대로 전파되는 근본적인 어려움 (단안 깊이 추정은 ill-posed problem — 동일한 2D 이미지가 여러 3D 장면에서 나올 수 있음)
- **LSS (Lift-Splat-Shoot) 파이프라인**: 각 픽셀마다 깊이에 대한 확률 분포를 예측해(카테고리 깊이 분포) 3D 공간에 배치하고(**Lift**), Pillar 단위로 BEV 격자에 집계한(**Splat**) 뒤, 다운스트림 태스크(탐지·플래닝)를 수행(**Shoot**)하는 3단계 구조 — BEVDepth, BEVFusion 등 이후 다수 방법의 기반
- **Frustum**: 카메라 시야각이 이루는 3D 피라미드 형태의 공간. 픽셀 위치 + 깊이 분포로 3D 공간을 표현하는 단위
- **Pillar Pooling / BEV Pooling**: Frustum 형태로 생성된 3D 점들을 BEV 격자의 각 셀(pillar)에 sum pooling으로 집계하는 연산. Cumulative Sum Trick(정렬+누적합) 등으로 GPU에서 크게 가속됨 (BEVFusion은 기존 대비 40배 가속)
- **Temporal Fusion (시간적 융합)**: 여러 프레임의 BEV 특징을 ego 좌표계로 정렬해 결합하는 방식. 정지 상태의 단일 프레임만으로는 어려운 속도 추정, 가려진 물체 인지에 필수적

## Transformer 기반 BEV/탐지

- **Deformable Attention**: 전체 특징 맵이 아니라 각 쿼리마다 소수의 학습된 key point에만 attention을 수행하는 방식. 표준 cross-attention보다 계산 효율적이며 BEVFormer, MapTR, StreamMapNet 등의 핵심 연산
- **Spatial Cross-Attention (SCA)**: 여러 높이(anchor height)에서 3D 포인트를 샘플링해 2D 이미지에 투영함으로써, 깊이를 직접 추정하지 않고도 멀티카메라 특징을 BEV로 모으는 방식 (BEVFormer)
- **Temporal Self-Attention (TSA)**: 이전 타임스텝의 BEV 특징을 ego-motion으로 보정한 뒤 현재 쿼리와 함께 attention을 수행해 시간 정보를 재귀적으로 융합하는 방식 (BEVFormer)
- **DETR 패러다임**: 학습 가능한 쿼리(query) 집합이 병렬로 모든 객체/맵 요소를 한 번에 예측하는 end-to-end 구조. Bipartite Matching(헝가리안 알고리즘)으로 예측-정답을 1:1 매칭해 NMS 없이 학습
- **top-down 3D→2D 접근**: 깊이를 먼저 추정하는 대신, 3D 공간의 가설(쿼리)을 2D 이미지에 역투영해 검증하는 방식 — pseudo-LiDAR(깊이 추정 후 포인트클라우드 변환)의 오차 누적 문제를 피함 (DETR3D)

## LiDAR / 포인트클라우드 인코더

- **Pillar (기둥)**: 높이가 무한한 voxel. LiDAR 포인트를 수직 기둥 단위로 묶어 PointNet 방식으로 인코딩하면, 이후는 표준 2D CNN만으로 처리할 수 있어 3D convolution을 완전히 제거할 수 있음 (PointPillars)
- **Point-level Fusion vs Shared BEV Space**: 카메라 특징을 LiDAR 포인트에 직접 붙이는 방식은 LiDAR의 희소성 때문에 정보 손실이 크지만, 카메라·LiDAR를 각각 BEV로 변환한 뒤 합치는 방식(Shared BEV)은 두 센서의 기하·의미 정보를 모두 보존 (BEVFusion)
- **CenterNet 기반 탐지**: 객체를 keypoint(중심점)로 표현하는 방식. 박스 회귀 대신 중심점 히트맵을 예측해 NMS 없이도 빠르고 간결한 탐지가 가능 (CenterPoint)
- **Greedy Closest-Point Matching**: 예측 속도로 역투영한 중심점과 이전 프레임 tracklet 중심점 사이의 최소 거리로 객체를 추적하는 매우 가벼운(1ms) 방식 (CenterPoint)

## HD 맵 (벡터화)

- **Vectorized Map (벡터화 맵)**: 픽셀 단위 세그멘테이션이 아니라 점(Point)과 선(Edge)으로 맵 요소를 표현하는 방식. 모션 예측·플래닝에 바로 활용 가능
- **Polyline / Polygon**: 순서 있는 정점들의 집합으로 도로 요소를 표현하는 형식. 차선(polyline)은 양 끝점 어느 쪽에서 시작해도, 횡단보도(polygon)는 어느 정점·방향에서 순회해도 같은 도형이 되는 **순열 모호성(permutation ambiguity)** 문제가 있음
- **Permutation-equivalent Modeling**: 맵 요소를 점 집합 $V$와 그 모든 등가 순열의 집합 $\Gamma$로 표현해, 순열 모호성을 원천적으로 해소하는 방법 — 학습 시 $\Gamma$ 중 GT와 가장 가까운 순열을 정답으로 채택 (MapTR)
- **Hierarchical Bipartite Matching**: 인스턴스 레벨(어떤 예측이 어떤 GT 맵 요소인지) 매칭 후, 포인트 레벨(그 안에서 어떤 순열이 최적인지) 매칭을 순차적으로 수행하는 2단계 매칭
- **Stacking vs Streaming**: 과거 여러 프레임을 한 번에 이어붙여 처리하면(Stacking) 비용이 프레임 수에 비례해 증가하지만, 압축된 메모리 특징만 다음 프레임으로 전달하면(Streaming) 프레임 수와 무관한 상수 비용으로 장기 이력을 통합할 수 있음 (StreamMapNet)
- **IPM (Inverse Perspective Mapping)**: 바닥이 평평하다는 가정 하에, 카메라의 원근 뷰를 기하학적 변환만으로 BEV로 바꾸는 고전적 방법

## 3D Occupancy / Semantic Scene Completion

- **3D Occupancy Prediction**: 3D 공간을 복셀 격자로 나누고, 각 복셀이 점유되어 있는지(occupied/free/unobserved)와 어떤 의미(semantic class)인지를 동시에 예측하는 태스크. 3D 탐지(바운딩 박스)보다 훨씬 세밀하고 임의 형태의 물체(장애물, 잔해 등)도 표현 가능
- **General Objects (GOs)**: 미리 정의된 카테고리 온톨로지에 없는 일반 객체(도로 위 쓰레기통 등). 안전 주행에는 중요하지만 기존 탐지 방법은 놓치기 쉬움
- **FLoSP (Features Line of Sight Projection)**: 카메라 시선(ray) 위에 놓인 모든 3D 복셀이 같은 2D 픽셀에 투영된다는 점을 이용해, 깊이를 모르는 상태에서도 해당 시선의 2D 특징을 모든 깊이의 복셀에 공유시키는 2D-3D 브릿지 (MonoScene)
- **TPV (Tri-Perspective View)**: BEV(위)에 더해 Side, Front까지 3개의 직교 평면으로 3D 공간을 표현. Voxel 대비 한 차원 낮은 복잡도($O(HW{+}DH{+}WD)$ vs $O(HWD)$)로 모든 축의 정보를 보존 (TPVFormer)
- **2D-3D Spatial Attention**: BEV 쿼리 대신 3D 볼륨 쿼리를 직접 정의하고, 각 쿼리의 3D 기준점을 2D 이미지에 투영해 특징을 샘플링하는 방식 (SurroundOcc)
- **Ray Casting + Voxel Densification**: 희소한 LiDAR 포인트를 멀티프레임 누적과 메시 재구성(Poisson Surface Reconstruction 등)으로 밀도 있게 만들고, 카메라/LiDAR 원점에서 광선을 쏘아 각 복셀의 가시성(관측됨/비어있음/미관측)을 판정하는 GT 생성 파이프라인 (Occ3D)
- **World Model 방식의 Occupancy 예측**: 매 프레임을 독립적으로 처리하는 대신, 이전 프레임의 가우시안이 어떻게 이동·변형됐는지를 학습해 현재 상태를 예측하는 스트리밍 방식 — 4D(3D+시간) occupancy forecasting으로 문제를 재정의 (GaussianWorld)

## 평가 지표

- **NDS (nuScenes Detection Score)**: mAP와 5가지 True Positive 지표(mATE 위치, mASE 크기, mAOE 방향, mAVE 속도, mAAE 속성 오차)의 가중합으로 탐지 품질을 종합 평가
- **mIoU (mean Intersection over Union)**: 클래스별 IoU(예측-정답 겹침 비율)의 평균. 세그멘테이션/occupancy 평가의 표준 지표
- **Chamfer Distance / Fréchet Distance**: 두 점 집합·곡선 사이의 유사도 지표. Chamfer는 순서 무관하게 최근접 점끼리 비교하고, Fréchet은 순서(형태·방향)까지 고려해 더 엄격하게 비교
- **AbsRel**: 깊이 추정 오차 지표. $\frac{1}{N}\sum |d-d^*|/d^*$ — 낮을수록 좋음
