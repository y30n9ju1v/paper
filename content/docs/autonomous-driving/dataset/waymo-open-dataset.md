---
title: "Scalability in Perception for Autonomous Driving: Waymo Open Dataset"
date: 2026-04-10T09:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "Benchmark & Dataset"]
tags: ["Autonomous Driving", "Dataset", "LiDAR", "3D Object Detection", "Multi-Object Tracking"]
year: 2020
references: []
---

## 💡 한 줄 요약
기존 데이터셋보다 15배 이상 지리적으로 다양하고, 약 1200만 개의 3D LiDAR 및 카메라 주석을 포함한 대규모 멀티모달 자율주행 데이터셋 Waymo Open Dataset을 공개했다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurélien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo 외 다수 (Waymo LLC, Google LLC)
- **발행년도**: 2020 (arXiv: 1912.04838v7)
- **주요 기여점**:
  1. 1150개 장면(각 20초), LiDAR 5대 + 카메라 5대로 구성된 대규모 멀티모달 데이터셋 공개, 약 1200만 개의 3D/2D 주석 제공
  2. 샌프란시스코·피닉스·마운틴뷰 3개 도시, 76km² 방문 면적으로 기존 최대 데이터셋 대비 15배 이상의 지리적 다양성 확보
  3. 방향각 정확도를 반영하는 새로운 검출 지표 APH(Average Precision Heading) 제안
  4. LiDAR-카메라 간 정밀 동기화 및 롤링 셔터 보정 정보를 제공하여 멀티모달 융합 연구 기반 마련

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: 자율주행 벤치마크는 KITTI(22개 장면, 80K 3D 박스, LiDAR 1대)로 표준화되었으나 규모가 작았고, 이후 nuScenes(1000개 장면, 1.4M 3D 박스, 방문 면적 5km², 지도 정보 포함)와 Argoverse(113개 장면, 993K 3D 박스, 상세 HD 맵 제공하지만 LiDAR 1대)가 규모와 정보를 확장하는 방향으로 발전해왔다. Waymo Open Dataset은 이 흐름에서 규모와 지리적 다양성을 한 단계 더 끌어올린 데이터셋이다.
- **기존 한계점**:
  1. 규모의 제한 — KITTI는 22개 장면, 80K 3D 박스에 불과해 딥러닝 모델의 일반화 검증에 한계가 있었다.
  2. 지리적 다양성 부족 — nuScenes, Argoverse 등은 커버하는 지역과 방문 면적이 제한적이어서 학습 도메인과 실제 운영 환경 사이의 일반화 문제가 존재했다.
  3. 멀티모달 정합성 부족 — 일부 데이터셋은 LiDAR 1대만 사용하거나 지도 정보가 제한적이어서 종합적인 인식 연구에 제약이 있었다.
- **이 논문의 접근 방식**: 여러 고해상도 카메라와 고품질 LiDAR 스캐너로 수집한 데이터로 구성된 새로운 대규모 데이터셋을 구축하여 규모, 지리적 다양성, 주석 품질을 모두 끌어올린다.

## 📑 목차
- Chapter 1: Introduction
- Chapter 2: Related Work
- Chapter 3: Waymo Open Dataset (센서 사양, 좌표계, 그라운드 트루스 레이블, 센서 데이터, 데이터셋 분석)
- Chapter 4: Tasks (객체 검출, 객체 추적)
- Chapter 5: Experiments (베이스라인 3D 검출, 2D 검출, 멀티 객체 추적, 도메인 갭, 데이터셋 크기)

## 🛠️ Chapter 1: Introduction

**요약**

자율주행 기술 발전을 가속하기 위해 Waymo는 가장 크고 다양한 멀티모달 자율주행 데이터셋을 공개한다. 기존 자율주행 데이터셋들은 규모와 지리적 다양성이 제한적이어서, 학습 도메인과 운영 환경 사이의 일반화 문제가 있었다. 이 논문은 여러 고해상도 카메라와 고품질 LiDAR 스캐너로 수집된 데이터로 구성된 새로운 대규모 데이터셋을 소개한다.

**핵심 개념**

- **Waymo Open Dataset**: 1150개 장면(각 20초), 산업용 LiDAR 5대 + 고해상도 카메라 5대로 구성
- **규모**: 약 1200만 개의 3D LiDAR 박스 주석, 약 1200만 개의 카메라 박스 주석, 113k LiDAR 추적 트랙, 250k 카메라 이미지 트랙
- **지리적 다양성**: 샌프란시스코, 피닉스, 마운틴뷰 3개 도시에서 수집, 기존 최대 카메라+LiDAR 데이터셋 대비 15배 더 다양한 지역 커버리지

## 🛠️ Chapter 2: Related Work

**요약**

자율주행 관련 공개 데이터셋들을 비교 분석한다. KITTI(22개 장면), nuScenes(1000개 장면), Argoverse(113개 장면) 등 기존 데이터셋들과 비교하여 Waymo Open Dataset의 차별점을 제시한다. 특히 지도 정보, 3D 주석 수, 방문 면적 등 여러 측면에서 기존 데이터셋을 압도한다.

**핵심 개념**

- **KITTI**: 22개 장면, 80K 3D 박스, LiDAR 1대 — 자율주행 벤치마크의 표준이지만 규모가 작음
- **nuScenes**: 1000개 장면, 1.4M 3D 박스, 방문 면적 5km² — 지도 정보 포함하지만 지리적 다양성 제한
- **Argoverse**: 113개 장면, 993K 3D 박스 — 상세 HD 맵 제공하지만 LiDAR 1대 사용
- **Waymo Open Dataset**: 1150개 장면, 12M 3D 박스, 방문 면적 76km² — 압도적인 규모와 지리적 다양성

## 🛠️ Chapter 3: Waymo Open Dataset

### 3.1 센서 사양 (Sensor Specifications)

**요약**

데이터 수집에는 5대의 고품질 LiDAR와 5대의 고해상도 핀홀 카메라가 사용된다. LiDAR는 전방(TOP), 전면(FRONT), 측면(SIDE_LEFT, SIDE_RIGHT)에 배치되며, 카메라는 전방(F), 전방좌(FL), 전방우(FR), 측면좌(SL), 측면우(SR)에 배치된다.

**핵심 개념**

- **TOP LiDAR**: 수직 FOV [-17.6°, +2.4°], 범위 75m (제한), 초당 2번 회전
- **FRONT/SIDE LiDAR**: 수직 FOV [-90°, 30°], 범위 20m
- **카메라 이미지**: 전방 카메라 1920×1280, 측면 카메라 1920×1040, 수평 FOV ±25.2°
- **롤링 셔터**: 카메라는 롤링 셔터 방식으로 촬영되어 LiDAR와의 동기화를 위한 프로젝션 보정이 필요

### 3.2 좌표계 (Coordinate Systems)

**요약**

데이터셋에서 사용되는 4가지 좌표계를 정의한다. 모든 좌표계는 오른손 법칙을 따르며, 데이터셋 내 임의의 두 프레임 사이의 변환 정보가 포함된다.

**핵심 개념**

- **Global frame**: 동북방향 좌표계, Z축이 중력 반대 방향
- **Vehicle frame**: 차량 중심, X축 전방, Y축 좌측, Z축 상방
- **Sensor frame**: 각 센서별 독립 좌표계, 차량 프레임으로의 변환 행렬(extrinsics) 제공
- **Image frame**: 각 카메라 이미지의 2D 좌표계

**수식 예제 — LiDAR 구형 좌표 변환**

$$\text{range} = \sqrt{x^2 + y^2 + z^2}$$

**수식 설명**
- **range**: LiDAR 센서로부터 포인트까지의 거리 (단위: m)
- **x, y, z**: LiDAR Cartesian 좌표계에서의 3D 위치값

$$\text{azimuth} = \text{atan2}(y, x)$$

**수식 설명**
- **azimuth**: 수평면에서의 각도 (방위각). atan2는 x, y를 이용해 -π ~ π 범위의 각도를 계산하는 함수

$$\text{inclination} = \text{atan2}(z, \sqrt{x^2 + y^2})$$

**수식 설명**
- **inclination**: 수평면으로부터의 수직 각도 (앙각). z값과 수평거리의 비를 이용해 계산

### 3.3 그라운드 트루스 레이블 (Ground Truth Labels)

**요약**

차량, 보행자, 자전거, 표지판에 대한 고품질 수동 주석을 제공한다. LiDAR 레이블은 7-DOF 3D 바운딩 박스(위치, 크기, 방향)로, 카메라 레이블은 4-DOF 2D 바운딩 박스로 구성된다.

**핵심 개념**

- **7-DOF 3D 바운딩 박스**: (cx, cy, cz, l, w, h, θ) — 중심 좌표, 길이/너비/높이, 방향각
- **추적 ID**: 모든 그라운드 트루스 박스에 추적 ID가 부여되어 시간에 따른 동일 객체 매칭 가능
- **난이도 레벨**: LEVEL_1(쉬움), LEVEL_2(누적, LEVEL_1 포함) — KITTI와 유사한 2단계 난이도 시스템
- **레이블 품질**: 전문 주석자가 제작하고 다중 검증 단계를 거쳐 높은 품질 보장

### 3.4 센서 데이터 (Sensor Data)

**요약**

LiDAR 데이터는 레인지 이미지 형식으로 인코딩되어 제공된다. 각 LiDAR 포인트는 range, intensity, elongation 및 vehicle pose를 포함한다. 카메라 데이터는 JPEG 압축 이미지로 제공되며, 롤링 셔터 보정 정보가 포함된다.

**수식 예제 — 카메라-LiDAR 동기화**

$$\text{sync\_accuracy} = \text{camera\_center\_time} - \text{frame\_start\_time} - \text{camera\_center\_offset} / 360° \times 0.1s$$

**수식 설명**
- **camera_center_time**: 이미지 중심 픽셀의 노출 시간
- **frame_start_time**: 해당 데이터 프레임의 시작 시간
- **camera_center_offset**: 각 카메라 센서 프레임의 +x 축 오프셋 (예: FRONT 카메라 90°, FRONT_LEFT 카메라 90°+45°)
- 동기화 오차는 [-6ms, 7ms] 범위, 99.7% 신뢰도

**핵심 개념**

- **레인지 이미지**: LiDAR 포인트를 이미지 형태로 표현, 각 픽셀이 하나의 LiDAR 리턴에 해당
- **Elongation**: 레이저 펄스의 시간 폭 연장 — 먼지, 비, 안개 같은 부유물 분류에 유용
- **Rolling shutter projection**: 롤링 셔터로 촬영된 카메라 이미지에 LiDAR 포인트를 정확히 매핑하는 기법

### 3.5 데이터셋 분석 (Dataset Analysis)

**요약**

데이터셋은 교외/도심 환경, 낮/밤/새벽 등 다양한 시간대에서 수집된 장면으로 구성된다. 지리적 커버리지 지표로 150m 가시거리에서의 희석된 에고 포즈의 합집합 면적을 사용한다.

**핵심 개념**

- **지리적 커버리지**: 피닉스 40km², 샌프란시스코 36km² 커버 — 기존 데이터셋 대비 15.2배 우수
- **시간대 다양성**: 낮(Day), 밤(Night), 새벽(Dawn) 등 다양한 조명 조건 포함
- **훈련/검증/테스트 분할**: 훈련 798개, 검증 202개, 테스트 150개 시퀀스

## 🛠️ Chapter 4: Tasks

### 4.1 객체 검출 (Object Detection)

**요약**

2D 및 3D 객체 검출 태스크를 정의하며, 새로운 평가 지표인 APH(Average Precision Heading)를 도입한다. 기존 AP는 heading 정보를 반영하지 못하는 한계가 있었다.

**수식 예제 — AP 및 APH**

$$\text{AP} = 100 \int_0^1 \max\{p(r') \mid r' \geq r\} \, dr$$

**수식 설명**
- **AP (Average Precision)**: 재현율(r) 전 구간에 걸쳐 최대 정밀도를 적분한 값 (0~100 스케일)
- **p(r)**: P/R 커브 — r은 재현율(recall), p는 정밀도(precision)
- **∫**: 재현율 0부터 1까지 적분 → 면적을 구해 전체 성능을 하나의 숫자로 표현

$$\text{APH} = 100 \int_0^1 \max\{h(r') \mid r' \geq r\} \, dr$$

**수식 설명**
- **APH (Average Precision weighted by Heading)**: heading 정확도로 가중된 AP
- **h(r)**: heading 정확도 가중치가 적용된 정밀도. 가중치는 $\min(\{|\theta - \hat{\theta}|, 2\pi - |\theta - \hat{\theta}|\}) / \pi$
  - **θ**: 예측된 heading 각도
  - **$\hat{\theta}$**: 실제 ground truth heading 각도
  - heading이 완벽히 맞으면 가중치 1, 180° 틀리면 가중치 0

**핵심 개념**

- **IoU 기반 매칭**: 차량/보행자 0.7 IoU, 자전거 0.5 IoU로 매칭 — 예측과 ground truth 박스의 겹침 비율
- **헝가리안 매칭**: 예측-ground truth 쌍 매칭에 헝가리안 알고리즘 사용
- **2D 카메라 검출**: LiDAR 데이터 사용 없이 단일 카메라 이미지만으로 2D 바운딩 박스 예측

### 4.2 객체 추적 (Object Tracking)

**요약**

멀티 객체 추적(MOT)은 장면 내 객체들의 identity, 위치, 속성을 시간에 따라 추적하는 태스크이다. MOTA와 MOTP를 최종 평가 지표로 사용한다.

**수식 예제 — MOTA 및 MOTP**

$$\text{MOTA} = 100 - 100 \frac{\sum_t (m_t + \text{fp}_t + \text{mme}_t)}{\sum_t g_t}$$

**수식 설명**
- **MOTA (Multiple Object Tracking Accuracy)**: 추적 정확도 (높을수록 좋음, 최대 100)
- **$m_t$**: 시간 t에서의 miss (탐지 실패)
- **$\text{fp}_t$**: 시간 t에서의 false positive (오탐지)
- **$\text{mme}_t$**: 시간 t에서의 mismatch (ID 전환 오류)
- **$g_t$**: 시간 t에서의 ground truth 객체 수

$$\text{MOTP} = 100 \frac{\sum_{i,t} d_t^i}{\sum_t c_t}$$

**수식 설명**
- **MOTP (Multiple Object Tracking Precision)**: 매칭된 쌍들의 위치 정밀도
- **$d_t^i$**: 시간 t에서 i번째 매칭 쌍의 거리 (1 - IoU)
- **$c_t$**: 시간 t에서 매칭된 쌍의 수

**핵심 개념**

- **mismatch**: ground truth 객체가 이전과 다른 트랙에 매칭되면 1로 계산 — ID 전환 오류 측정
- **MOTA 선택 기준**: 모든 점수 임계값에서 MOTA를 계산하고 최고값을 최종 지표로 사용

## 📊 주요 실험 및 결과 (Experiments & Results)

- **사용 데이터셋 / 벤치마크**: Waymo Open Dataset (훈련 798, 검증 202, 테스트 150 시퀀스), PointPillars(3D 검출), ResNet-101 기반 Faster R-CNN(2D 검출)
- **주요 성과**:
  - **3D 객체 검출 베이스라인**: PointPillars 재구현, 차량 APH 79.1 / 71.0 (LEVEL_1/2), 보행자 APH 56.1 / 51.1 (LEVEL_1/2). Voxel 크기 차량/보행자 0.33m, 그리드 범위 [-85m, 85m](X,Y), [-3m, 3m](Z)
  - **2D 검출**: ResNet-101 기반 Faster R-CNN, COCO 사전학습 후 파인튜닝
  - **도메인 갭**: 샌프란시스코(SF)로 학습 후 SF 평가 vs. 피닉스+마운틴뷰(SUB) 평가 시 APH 차이 7.6 발생. 보행자는 SUB에 수가 적어 더 큰 도메인 갭 발생 → 반지도학습/비지도 도메인 적응 연구 기회 제공
  - **데이터셋 크기 효과**: 10% → 100% 데이터 사용 시 차량 APH 29.7 → 49.4 (LEVEL_2)로 지속적 향상, 대규모 데이터가 성능에 직접 기여함을 실증

## 💡 결론 및 시사점 (Conclusion & Insights)

Waymo Open Dataset은 기존 자율주행 데이터셋의 한계(규모, 지리적 다양성, 멀티모달 정합성)를 극복한 대규모 공개 데이터셋이다.

- **주요 기여**: 1150개 장면, 20초씩, 10Hz 수집으로 약 1200만 개의 3D/2D 주석을 확보했고, 3개 도시에 걸친 76km² 지리적 커버리지, APH라는 새로운 평가 지표, LiDAR-카메라 간 정밀 동기화 및 롤링 셔터 보정 정보를 제공한다.
- **실무적 시사점**: 도시 간 도메인 갭이 확인되어 반지도학습/비지도 도메인 적응 연구의 필요성이 제기되었고, 데이터 크기와 성능이 비례함이 재확인되었다. LiDAR와 카메라 레이블의 연계로 멀티모달 융합 연구 기반이 마련되었으며, 이후 지도 정보·비레이블 데이터·행동 예측/계획 등으로 확장될 계획을 제시했다.
- **한계점 및 아쉬운 점**:
  - 공개 당시 지도(HD Map) 정보와 행동 예측/계획 태스크가 빠져 있어, nuScenes·Argoverse 대비 초기 활용 범위가 제한적이었음(이후 Waymo Open Motion Dataset 등으로 보완됨)
  - 도메인 갭이 확인되었음에도 해결책은 후속 연구 과제로 남겨두었을 뿐, 논문 자체에서 구체적 완화 기법을 제시하지 않음
  - 3개 도시만 커버하여 "15배 다양성"이라는 주장에도 불구하고 여전히 미국 서부 중심의 환경적 편향이 존재함

---

*관련 논문: [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/), [nuPlan](/posts/papers/nuPlan/), [CenterPoint](/posts/papers/centerpoint-center-based-3d-object-detection-and-tracking/), [PointPillars](/posts/papers/pointpillars-fast-encoders-object-detection-point-clouds/)*
