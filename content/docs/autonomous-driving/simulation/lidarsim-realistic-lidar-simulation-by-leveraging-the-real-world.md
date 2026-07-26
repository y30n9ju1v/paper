---
title: "LiDARsim: Realistic LiDAR Simulation by Leveraging the Real World"
date: 2026-04-24T12:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "Sensor Simulation"]
tags: ["LiDAR", "Simulation", "Synthetic Data", "Point Cloud", "Domain Gap", "Ray Casting"]
year: 2020
references:
  - "waymo-open-dataset"
---

## 💡 한 줄 요약
실제 주행 데이터로 구축한 3D 자산 위에 물리 기반 레이캐스팅과 학습된 raydrop 모델을 결합하여, CARLA 등 아티스트 제작 시뮬레이터보다 sim-to-real 도메인 갭이 훨씬 작은 현실적인 LiDAR 포인트 클라우드를 생성한다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Sivabalan Manivasagam, Shenlong Wang, Kelvin Wong, Wenyuan Zeng, Mikita Sazanovich, Shuhan Tan, Bin Yang, Wei-Chiu Ma, Raquel Urtasun (Uber ATG, University of Toronto, MIT)
- **발행년도**: 2020 (CVPR 2020)
- **주요 기여점**:
  1. 실제 데이터 기반 3D 자산(25,000+ 동적 객체 메시, 대용량 도시 정적 맵)의 자동화된 대량 구축 파이프라인
  2. 물리 레이캐스팅 + ML 기반 raydrop을 결합한 하이브리드 접근으로 레이캐스팅 단독 대비 현실성 향상
  3. 추가 학습 없이 실제 데이터 학습 모델을 LiDARsim에 직접 적용 가능할 만큼 낮은 도메인 갭 실증
  4. 실제로 수집 불가능한 롱테일·안전 임계 시나리오의 폐루프(closed-loop) 평가 가능성 검증

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: 자율주행 검증을 위한 시뮬레이션은 CARLA, AirSim 같은 게임 엔진 기반 가상 환경에서 출발했다. 이들은 아티스트가 제작한 3D 에셋과 단순화된 물리 모델을 사용해 빠르게 다양한 장면을 만들 수 있었지만, 실제 센서 데이터와의 통계적 분포 차이(도메인 갭)가 크다는 문제가 있었다. LiDARsim은 이 갭을 줄이기 위해 "가상 에셋 제작" 대신 "실제 주행 데이터로부터 자산을 재구성"하는 방향으로 접근을 전환한다.
- **기존 한계점**:
  1. 가상 환경의 도메인 갭 — CARLA, AirSim 같은 기존 시뮬레이터는 실제 LiDAR 데이터와 통계적 분포가 크게 달라, 시뮬레이터에서 학습한 모델이 실제 환경에서 성능이 급락함
  2. 안전 시나리오 수집의 어려움 — 재귀율이 낮은 롱테일 이벤트(도로 위 동물, 갑자기 끼어드는 차량 등)는 실제 주행으로 충분한 데이터를 모으는 것이 현실적으로 불가능하고 위험함
  3. 레이캐스팅만으로는 부족한 현실성 — 순수 물리 기반 레이캐스팅은 실제 LiDAR보다 약 10% 많은 포인트를 생성하고, 재질 반사율·입사각·대기 투과율 등 복잡한 물리 현상을 모두 모델링하지 못함
- **이 논문의 접근 방식**: 실제 주행 데이터로부터 대용량 3D 정적 맵과 동적 객체 메시 카탈로그를 구축하고, 레이캐스팅으로 초기 렌더링 후 신경망으로 "raydrop"을 학습해 현실적 포인트 클라우드를 생성한다.

## 📑 목차
- Section 1: Introduction
- Section 2: Related Work
- Section 3: Reconstructing the World for Simulation (자산 구축)
- Section 4: Realistic Simulation for Self-driving (센서 시뮬레이션)
- Section 5: Experimental Evaluation
- Section 6: Conclusion

## 🛠️ Section 1: Introduction

**요약**

자율주행차(SDV)의 안전을 보장하려면 SDV가 한 번도 보지 못한 객체를 탐지하고, 안전 임계 시나리오에서 올바르게 반응하는지 검증해야 합니다. 실제 도로 주행으로 이를 모두 테스트하는 것은 비용·위험·데이터 부족 문제로 현실적으로 불가능합니다.

저자들은 "실제 데이터를 활용하면 아티스트 제작 가상 세계보다 더 현실적인 시뮬레이션이 가능하다"는 전제 하에 LiDARsim을 제안합니다. LiDARsim은 두 단계로 구성됩니다:

1. **자산 생성(Assets Creation)**: 실제 주행 데이터에서 3D 정적 맵과 동적 객체 메시 카탈로그를 구축
2. **센서 시뮬레이션(Sensor Simulation)**: 물리 레이캐스팅 + ML 기반 raydrop 학습으로 현실적 LiDAR 포인트 클라우드 생성

**핵심 개념**

- **Sim-to-Real 도메인 갭**: 시뮬레이터에서 학습한 모델이 실제 환경에 적용될 때 발생하는 성능 저하. 가상 환경의 기하·광학 특성이 현실과 다르기 때문에 발생.
- **롱테일 이벤트(Long-tail Events)**: 희귀하지만 안전에 치명적인 시나리오. 실제 주행 데이터만으로는 충분한 샘플 확보가 불가능.
- **클로즈드루프 평가(Closed-loop Evaluation)**: 에이전트의 행동이 환경 상태를 변경하고 그 변경된 환경에서 다시 감지·행동하는 반응형 평가. 오픈루프와 달리 계획 오류가 누적됨.

## 🛠️ Section 3: Reconstructing the World for Simulation

**요약**

LiDARsim의 첫 단계는 시뮬레이션에 사용할 고품질 3D 자산을 실제 데이터에서 구축하는 것입니다. 정적 환경(맵)과 동적 객체(차량 등) 두 가지를 별도로 생성합니다.

### 3.1 정적 맵 구축 (3D Mapping for Simulation)

동일 구역을 여러 번 주행하며 수집한 LiDAR 스캔을 정합하여 고밀도 3D 맵을 구축합니다.

**프로세스**:
1. **다중 주행 데이터 수집**: 같은 장소를 평균 3회 이상 주행, 계절·날씨 변화 포함
2. **Graph-SLAM으로 정합**: 휠 오도메트리·IMU·LiDAR·GPS를 융합하여 센티미터 수준 정밀도 달성
3. **동적 객체 제거**: LiDAR 시맨틱 세그멘테이션으로 차량·보행자·자전거 자동 제거
4. **Surfel 메시 변환**: 복셀 다운샘플링(4×4×4 cm³) → 주성분 분석으로 법선 추정 → 디스크 서펠 생성
5. **메타데이터 기록**: 각 서펠에 반사 강도(intensity)·거리·입사각 저장 → 이후 현실적 렌더링에 활용

**핵심 개념**

- **Surfel(Surface Element)**: 점이 아닌 디스크 형태의 렌더링 프리미티브. 중심 위치와 법선 방향을 가지며, 폐색 처리와 충돌 검사에 효율적.
- **Graph-SLAM**: 로봇의 이동 궤적을 그래프로 표현하고 루프 클로징으로 누적 오차를 최소화하는 위치 추정 기법.

### 3.2 동적 객체 구축 (3D Reconstruction of Objects for Simulation)

차량 같은 동적 객체는 바운딩 박스 어노테이션과 차량의 좌우 대칭성을 활용하여 완전한 3D 메시를 생성합니다.

**프로세스**:
1. **바운딩 박스 내 포인트 추출**: 25초 스니펫에서 객체 상대 좌표로 변환
2. **대칭 완성(Symmetry Completion)**: 차량 진행 방향 축으로 포인트 미러링 후 원본과 합산 → 가려진 면 복원
3. **Color-ICP 정렬**: 기록된 강도를 컬러로 활용한 반복 최근점 정합으로 형상 정밀도 향상
4. **Surfel-disk 메시화**: 최종 완전한 3D 객체 메시 생성
5. **25,000개 이상** 동적 객체 카탈로그 구축

**핵심 개념**

- **Color-ICP(Iterative Closest Point)**: 포인트 클라우드 정합 알고리즘. 색상(여기서는 LiDAR 반사 강도) 정보를 추가 제약으로 사용해 정합 정확도 향상.
- **CAD 모델 대비 장점**: CAD 모델은 표준화된 형상만 표현하지만, 실제 데이터 기반 메시는 열린 트렁크·지붕 위 자전거·특수 차량 등 다양한 실제 변형을 자동으로 포착.

## 🛠️ Section 4: Realistic Simulation for Self-driving

**요약**

구축된 3D 자산(정적 맵 + 동적 객체)을 합성하여 가상 장면을 구성한 뒤, 두 단계로 현실적 LiDAR 포인트 클라우드를 생성합니다: (1) 물리 기반 레이캐스팅, (2) ML 기반 raydrop 보정.

### 4.1 물리 기반 시뮬레이션 (Physics-based Simulation)

Velodyne HDL-64E LiDAR를 모델링합니다. 64개 에미터-수신기 쌍이 360도 회전하며 약 110,000개 리턴을 생성합니다.

**레이캐스팅 프로세스**:
- 센서의 6-DOF 자세와 속도를 입력으로 받아, LiDAR 스윕 동안의 롤링 셔터 효과를 시뮬레이션
- 자차(ego-car)의 이동에 의한 모션 왜곡 보정
- LiDAR 스윕 중 다른 차량의 움직임을 36개 등간격 자세 업데이트로 시뮬레이션
- Intel Embree 레이캐스팅 엔진(Möller-Trumbore 알고리즘)으로 모든 서펠에 대한 레이-삼각형 충돌 검사

**수식 예제**

$$\mathbf{c} = \mathbf{c}_0 + (t_1 - t_0)\mathbf{v}_0, \quad \mathbf{n} = \mathbf{R}_0[\cos\theta\cos\phi,\ \cos\theta\sin\phi,\ \sin\theta]^T$$

**수식 설명**

이 수식은 LiDAR 센서가 발사하는 개별 레이(ray)의 출발 위치와 방향을 정의합니다:
- **$\mathbf{c}$**: 레이의 출발점(소스 위치). 스윕 중 센서가 이동하므로 시간에 따라 달라짐.
- **$\mathbf{c}_0$**: 스윕 시작 시점의 센서 3D 위치
- **$(t_1 - t_0)$**: 스윕 시작($t_0$)과 해당 레이 발사 시각($t_1$) 사이의 시간 차
- **$\mathbf{v}_0$**: 스윕 시작 시점의 센서 속도 벡터
- **$\mathbf{n}$**: 레이의 방향 벡터 (단위 벡터)
- **$\mathbf{R}_0$**: 스윕 시작 시점의 센서 자세를 나타내는 3D 회전 행렬
- **$\theta$**: 레이의 수직(고도) 각도
- **$\phi$**: 레이의 수평(방위) 각도
- **직관**: 자동차가 이동하면서 LiDAR를 쏘므로, 각 레이는 살짝 다른 위치에서 발사됨. 이 수식이 그 위치 오프셋을 정확히 계산해 롤링 셔터 효과를 재현함.

### 4.2 Raydrop 학습 (Learning to Simulate Raydrop)

**문제**: 실제 LiDAR는 레이캐스팅보다 약 10% 적은 포인트를 반환합니다. 레이가 물체에 닿아도 신호가 수신기에 돌아오지 않는 현상을 **raydrop**이라 합니다.

**원인**: 재질 반사율, 입사각, 거리, 빔 발산각, 대기 투과율 등 복잡한 물리 현상이 복합적으로 작용. 아티스트 제작 시뮬레이터에서는 이런 요인들이 불가용.

**해결책**: LiDAR 포인트 클라우드를 64×2048 2D 극좌표 이미지 그리드로 변환한 뒤, U-Net으로 각 픽셀(= 레이)이 리턴될 확률을 예측.

**입력 피처 (관측 가능한 물리 요인들)**:
- **실수 채널**: 거리(range), 기록된 강도(original intensity), 서펠 입사각(incidence angle)
- **정수 채널**: 레이저 ID, 시맨틱 클래스(도로/차량/배경)
- **이진 채널**: 초기 레이캐스팅 점유 마스크

**학습**:
- 크로스 엔트로피 손실로 binary classification 학습 (리턴/미리턴)
- 추론 시 확률 맵에서 샘플링 → 결정론적 thresholding 대신 샘플링을 사용해 실제 LiDAR의 확률적 특성 재현
- 맵 구축 데이터의 6% 스니펫 사용, Adam 옵티마이저(lr=1e-4)

**핵심 개념**

- **Raydrop**: LiDAR 레이가 물체에 닿았음에도 수신기가 신호를 감지하지 못해 포인트가 생성되지 않는 현상. 약한 반사 신호가 임계값 이하로 떨어질 때 발생.
- **극좌표 이미지 그리드(Polar Image Grid)**: LiDAR 스캔을 64(수직 채널) × 2048(수평 해상도)의 2D 이미지로 표현. 각 픽셀이 하나의 레이에 대응.
- **U-Net**: 인코더-디코더 구조의 CNN으로, 공간적 맥락을 보존하며 픽셀 단위 예측에 강점.

## 🛠️ Section 5: Experimental Evaluation

**요약**

LiDARsim을 4가지 관점에서 검증합니다: (1) CARLA 대비 높은 재현도, (2) 실제 데이터 대비 성능, (3) 합성 데이터를 통한 성능 향상, (4) 안전 시나리오 평가.

**핵심 개념**

- **데이터셋**: 자체 도시 데이터셋(5,500개 25초 스니펫, 140만 LiDAR 스윕, 북미 여러 도시, 사계절 포함 — 맵 구축 ~87%, 다운스트림 인식 학습/검증/테스트 ~7%/~1%/~5%), KITTI(공개 벤치마크로 타 시뮬레이터와 비교 시 사용)

## 📊 주요 실험 및 결과 (Experiments & Results)
- **사용 데이터셋 / 벤치마크**: 자체 대규모 도시 주행 데이터셋(5,500개 25초 스니펫, 140만 LiDAR 스윕), KITTI, SemanticKITTI

- **CARLA 대비 우월한 재현도**

| 학습 데이터 | 차량 세그멘테이션 mIOU | 차량 탐지 mAP (IoU 0.5) |
|------------|----------------------|------------------------|
| CARLA-Default | 0.36 | 20.0 |
| CARLA-Modified | - | 57.4 |
| **LiDARsim (Ours)** | **0.79** | **84.6** |
| SemanticKITTI Oracle | 0.81 | 88.1 |

LiDARsim은 실제 데이터(Oracle)에 근접하며 CARLA를 크게 상회.

- **Raydrop 효과 분석**

| Raydrop 방법 | 차량 탐지 mAP (≥1pt, IoU 0.7) |
|-------------|-------------------------------|
| No raydrop | 69.2 |
| Random raydrop (10%) | 69.4 |
| **ML raydrop (Ours)** | **71.6** |
| GT raydrop (Oracle) | 72.3 |
| Real data | 75.2 |

ML raydrop이 무작위 드롭이나 드롭 없음 대비 ~2% AP 향상, Oracle에 근접.

- **실제 데이터 기반 객체 vs CAD 모델**

| 객체 소스 | 차량 탐지 mAP |
|----------|--------------|
| CAD 모델 | 65.9 |
| **실제 데이터 객체 (Ours)** | **71.6** |
| Real data | 75.2 |

실제 데이터 기반 객체가 CAD 대비 ~6% AP 향상.

- **합성 데이터를 활용한 데이터 증강**

| 학습 셋 | 차량 세그 mIOU |
|---------|--------------|
| Real 10k | 94.6 |
| Sim 100k + Real 10k | 95.8 |
| Real 100k | 96.1 |
| Sim 100k + Real 100k | **96.3** |

실제 100k에 합성 100k를 추가하면 추가 성능 향상(96.1→96.3).

- **안전 시나리오 및 롱테일 테스팅**: Rare Object Testing에서 CAD 모델 동물·구조물을 배치해 미지 객체 탐지 능력 평가(OSIS 알고리즘, Unknown UQ: 실제 54.9 vs LiDARsim 66.0). Safety-critical Testing에서 "버스 뒤에서 갑자기 끼어드는 차량" 시나리오를 110개 지역·교통 설정에서 생성, NMP(Neural Motion Planner)가 90% 성공률 달성. Real2Sim 평가에서 Detection Agreement($\kappa_{det}$) = 86.5% (IoU 0.7) 달성.

- **주요 성과**: CARLA 대비 차량 탐지 mAP 84.6 vs 20.0으로 압도적 향상, 실제 데이터(Oracle) 대비 성능 격차를 크게 줄임. ML raydrop과 실제 데이터 기반 동적 객체가 각각 현실성 향상에 핵심적으로 기여.

## 💡 결론 및 시사점 (Conclusion & Insights)
LiDARsim은 실제 주행 데이터를 활용해 현존 LiDAR 시뮬레이터 중 가장 낮은 sim-to-real 도메인 갭을 달성한다. LiDAR 합성 데이터의 핵심은 기하 정확성(레이캐스팅)뿐 아니라 포인트 밀도 분포의 현실성(raydrop)에 있다는 점, 실제 데이터 기반 동적 객체 카탈로그가 CAD 모델 대비 훨씬 높은 다양성과 현실성을 제공한다는 점, 합성 데이터가 실제 데이터가 충분할 때도 추가적 성능 향상(데이터 증강 효과)에 기여한다는 점이 실무적으로 중요한 시사점이다. 시뮬레이터의 유효성을 Detection Agreement($\kappa_{det}$) 같은 새로운 지표로 정량화할 수 있다는 방법론적 기여도 눈여겨볼 만하다.

- 이 논문은 이후 카메라와 LiDAR를 동시에 NeRF 기반으로 시뮬레이션하는 방향(예: UniSim, CVPR 2023)으로 이어지는 직접적 선행 연구로 평가된다.
- **한계점 및 아쉬운 점**:
  1. 현재는 강체(rigid) 차량 위주로 설계되어 보행자·자전거 등 변형 가능 객체로의 확장이 필요하다.
  2. 날씨·조명 조건 변화에 대한 시뮬레이션을 지원하지 않아, 조건부 생성 모델과의 통합이 후속 과제로 남는다.
  3. Raydrop 모델이 학습 데이터 분포(특정 도시·계절)에 의존적이라, 완전히 새로운 센서나 환경으로의 일반화는 검증되지 않았다.

---

*관련 논문: [PointNet](/posts/papers/pointnet-deep-learning-on-point-sets-for-3d-classification-and-segmentation/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
