---
title: "Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting"
date: 2026-04-19T07:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "Autonomous Driving", "Novel View Synthesis", "Dynamic Scene", "LiDAR", "Real-Time Rendering"]
year: 2024
references:
  - "3d-gaussian-splatting"
  - "nuscenes-multimodal-dataset-autonomous-driving"
---

## 💡 한 줄 요약
Street Gaussians는 정적 배경과 동적 차량을 각각 별도의 명시적 포인트 클라우드(3D Gaussian)로 표현하고 차량 외관을 4D 구형 조화 함수로 인코딩하여, 30분 내 훈련과 135 FPS 실시간 렌더링으로 Waymo 기준 PSNR 34.61을 달성한다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, Sida Peng (Zhejiang University, Li Auto)
- **발행년도**: 2024 (arXiv:2401.01339v3)
- **주요 기여점**:
  1. 정적 배경과 동적 차량을 분리된 포인트 클라우드(Background Model / Object Model)로 표현하는 명시적 Street Gaussians 프레임워크 제안
  2. 차량의 시간에 따른 외관 변화를 저장 효율적으로 표현하는 4D 구형 조화 함수(4D Spherical Harmonics, 푸리에 계수 기반) 도입
  3. 노이즈가 있는 트래커 포즈를 렌더링 손실로 직접 보정하는 Tracking Pose Optimization으로 GT 포즈보다도 높은 품질 달성
  4. 30분 내 훈련, 135 FPS 실시간 렌더링, 명시적 표현 기반의 씬 편집(회전·이동·교체) 지원

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: 정적 장면 NeRF(MLP + 체적 렌더링) → Block-NeRF, GridNeRF 등 대규모 도시 확장 → NSG, MARS, EmerNeRF 등 추적된 차량 포즈를 활용한 동적 장면 NeRF 확장 → 3D Gaussian Splatting(3D-GS)의 명시적·고속 렌더링 등장이라는 흐름 속에서, Street Gaussians는 3D-GS를 동적 도심 장면으로 확장하는 방향을 취한다.
- **기존 한계점**:
  1. EmerNeRF, MARS 등 NeRF 기반 방법들은 훈련에 2.5시간 이상 소요되며 렌더링도 0.2~0.7 FPS에 불과해 느림
  2. 기존 방법들은 구형 조화 함수를 단일 timestep 단위로 별도 할당하거나, 글로벌 외관 변화를 제대로 포착하지 못함
  3. NeRF 기반 암시적 표현은 객체별 분리 및 편집이 어려워 장면 편집이 사실상 불가능함
- **이 논문의 접근 방식**: 명시적 포인트 클라우드 기반 3D Gaussian을 활용해 빠른 훈련과 렌더링을 달성하고, 4D 구형 조화 함수로 동적 외관을 저장 효율적으로 표현한다. 명시적 표현 덕분에 차량 회전·이동·교체 등 씬 편집도 자연스럽게 지원한다.

## 📑 목차
- Chapter 1: Introduction
- Chapter 2: Related Work
- Chapter 3: Method (배경/객체 모델, 렌더링, 훈련)
- Chapter 4: Implementation Details
- Chapter 5: Experiments
- Chapter 6: Conclusion

## 🛠️ Chapter 1: Introduction

**요약**

자율주행 시뮬레이션은 실제 도심 주행 환경을 정확하게 모델링해야 합니다. 최근 NeRF 기반 방법들(EmerNeRF, MARS, NSG 등)이 추적된 차량 포즈를 활용해 동적 도심 씬을 재구성하는 데 성공했지만, 훈련과 렌더링 속도가 크게 느리다는 한계가 있습니다.

이 논문은 **Street Gaussians**를 제안합니다. 핵심 아이디어는 다음과 같습니다:

- 동적 도심 씬을 **포인트 클라우드** 집합으로 표현 (명시적 표현)
- 각 포인트에 3D Gaussian을 부착해 정적 배경 또는 동적 차량을 모델링
- 차량의 시간 변화 외관은 **4D 구형 조화 함수(4D SH)**로 표현
- 30분 내 훈련 완료, 1066×1600 해상도에서 135 FPS 실시간 렌더링

**핵심 개념**

- **명시적 표현(Explicit Representation)**: 3D 공간을 점, 면 등 직접 정의할 수 있는 구조로 표현. NeRF의 암시적(implicit) MLP 표현과 달리 구조를 직접 편집 가능
- **3D Gaussian Splatting (3D-GS)**: 수백만 개의 3D Gaussian 타원체로 씬을 표현하고, 2D 이미지 평면에 투영(splatting)하여 빠르게 렌더링
- **구형 조화 함수(Spherical Harmonics, SH)**: 빛의 방향에 따라 달라지는 색상을 표현하는 수학적 함수 집합. 카메라 시점이 바뀌어도 자연스러운 하이라이트·반사 표현 가능

## 🛠️ Chapter 2: Related Work

**요약**

**정적 장면 모델링**: NeRF(MLP + 체적 렌더링)는 단일 정적 장면을 고품질로 재구성하지만 훈련·렌더링이 느림. Block-NeRF, GridNeRF 등이 대규모 도시 씬으로 확장을 시도. 3D Gaussian Splatting(3D-GS)은 명시적 표현으로 빠른 렌더링을 달성하지만 정적 장면만 지원.

**동적 장면 모델링**: NSG, MARS, EmerNeRF 등이 움직이는 차량을 포함한 동적 씬을 재구성. 이들은 추적된 차량 포즈를 활용해 관측 공간과 캐노니컬 공간을 매핑하지만, 여전히 훈련·렌더링 속도가 느림.

**자율주행 시뮬레이션**: CARLA, AirSim 같은 게임 엔진 기반 시뮬레이터는 현실감이 부족하고, LiDAR 집계 기반 방법들은 고해상도 이미지 처리에 한계가 있음.

**핵심 개념**

- **캐노니컬 공간(Canonical Space)**: 시간이나 포즈에 무관한 객체의 표준 좌표계. 여기서 모델링된 객체를 추적 포즈로 변환해 씬에 배치
- **NSG (Neural Scene Graph)**: 씬을 그래프로 표현하고 각 노드에 NeRF 네트워크를 할당하는 구조
- **EmerNeRF**: 정적·동적 필드를 분리 학습하는 NeRF (이 논문과 직접 비교됨)

## 🛠️ Chapter 3: Method

### 3.1 Street Gaussians

**요약**

Street Gaussians는 도심 씬을 **두 종류의 포인트 클라우드**로 분리하여 표현합니다:

1. **배경 모델(Background Model)**: 정적 배경을 월드 좌표계의 포인트 클라우드로 표현
2. **객체 모델(Object Model)**: 각 움직이는 차량을 별도의 포인트 클라우드로 표현

#### 배경 모델

배경의 각 포인트에는 다음이 할당됩니다:
- 위치 벡터 $\mu_b \in \mathbb{R}^3$와 공분산 행렬 $\Sigma_b$ (Gaussian 형태 결정)
- 불투명도 $\alpha_b$
- 구형 조화 계수 $\mathbf{z}_b$ (시점 의존적 색상)
- 시맨틱 로짓 $\boldsymbol{\beta}_b \in \mathbb{R}^M$ (M개 클래스의 의미 정보)

공분산 행렬은 스케일 행렬 $\mathbf{S}_b$와 회전 행렬 $\mathbf{R}_b$로 분해하여 최적화 중 유효성을 유지합니다:

**수식 예제**

$$\boldsymbol{\Sigma}_b = \mathbf{R}_b \mathbf{S}_b \mathbf{S}_b^T \mathbf{R}_b^T \tag{1}$$

**수식 설명** — 3D Gaussian의 공분산 행렬 분해:
- **$\boldsymbol{\Sigma}_b$**: 3D Gaussian의 모양과 방향을 결정하는 $3\times3$ 공분산 행렬
- **$\mathbf{S}_b$**: 스케일 행렬 (대각 원소가 x, y, z 방향 크기)
- **$\mathbf{R}_b$**: 회전 행렬 (단위 사원수로 표현, 최적화 중 유효 범위 보장)
- **직관**: 타원체의 크기($S$)와 방향($R$)을 분리하여 그래디언트 기반 최적화가 안정적으로 동작하도록 함

#### 객체 모델

씬에 $N$개의 움직이는 차량이 있을 때, 각 차량은 별도의 포인트 클라우드로 모델링됩니다. 차량의 Gaussian은 **객체 로컬 좌표계**에서 정의되며, 추적된 포즈(Tracked Pose)로 월드 좌표계로 변환됩니다:

$$\boldsymbol{\mu}_w = \mathbf{R}_t \boldsymbol{\mu}_o + \mathbf{T}_t$$
$$\mathbf{R}_w = \mathbf{R}_t \mathbf{R}_o \tag{2}$$

**수식 설명** — 객체 로컬 좌표 → 월드 좌표 변환:
- **$\boldsymbol{\mu}_w, \mathbf{R}_w$**: 월드 좌표계에서의 Gaussian 위치와 회전
- **$\boldsymbol{\mu}_o, \mathbf{R}_o$**: 객체 로컬 좌표계에서의 Gaussian 위치와 회전
- **$\mathbf{R}_t, \mathbf{T}_t$**: 시각 $t$에서 off-the-shelf 트래커가 제공하는 차량의 회전·이동 행렬
- **직관**: 차량을 자체 좌표계에서 독립적으로 모델링한 후, 트래커가 알려주는 위치·방향으로 씬에 배치

#### 4D 구형 조화 함수(4D Spherical Harmonics)

차량의 외관은 시간과 함께 변합니다(조명 변화, 그림자 등). 단순히 각 timestep마다 별도 SH를 쓰면 저장 비용이 폭발적으로 증가합니다. 대신, 각 SH 계수 $z_{m,l}$을 **푸리에 변환 계수**의 집합으로 대체합니다:

$$z_{m,l} = \sum_{i=0}^{k-1} f_i \cos\left(\frac{i\pi}{N_t} t\right) \tag{3}$$

**수식 설명** — 시간에 따라 변하는 SH 계수를 푸리에 기저로 표현:
- **$z_{m,l}$**: 시각 $t$에서의 SH 계수 (차수 $l$, 차원 $m$)
- **$f_i \in \mathbb{R}^k$**: 학습되는 푸리에 변환 계수 ($k$개, 이 논문에서는 $k=5$)
- **$N_t$**: 전체 프레임 수
- **$\cos(\cdot)$**: 코사인 기저 함수 (실수값 역 이산 푸리에 변환)
- **직관**: "차량 색상이 시간에 따라 부드럽게 변한다"는 사실을 $k$개의 푸리에 계수로 압축 표현. 시간 정보를 추가 저장 없이 외관에 인코딩

### 3.2 Rendering of Street Gaussians

**요약**

렌더링은 다음 순서로 수행됩니다:

1. 식 (3)으로 각 객체 Gaussian의 현재 SH 계수 계산
2. 식 (2)로 객체 포인트 클라우드를 월드 좌표계로 변환
3. 배경 포인트 클라우드와 합산하여 통합 포인트 클라우드 형성
4. 카메라 외부 파라미터 $\mathbf{W}$와 내부 파라미터 $\mathbf{K}$로 2D 이미지 공간에 투영

**수식 예제**

$$\boldsymbol{\mu}' = \mathbf{K}\mathbf{W}\boldsymbol{\mu}$$
$$\boldsymbol{\Sigma}' = \mathbf{J}\mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^T\mathbf{J}^T \tag{4}$$

**수식 설명** — 3D Gaussian을 2D 이미지 공간으로 투영:
- **$\boldsymbol{\mu}'$**: 2D 이미지 평면에서의 Gaussian 중심
- **$\boldsymbol{\Sigma}'$**: 2D 이미지 평면에서의 Gaussian 공분산 (타원 형태)
- **$\mathbf{J}$**: 원근 투영의 야코비안 행렬 (3D→2D 비선형 변환의 국소 선형 근사)
- **$\mathbf{W}$**: 월드→카메라 좌표계 변환 행렬
- **직관**: 3D 타원체를 카메라 방향으로 "납작하게" 눌러서 2D 타원으로 만드는 과정

5. 깊이 순서로 정렬 후 알파 블렌딩(alpha blending)으로 최종 픽셀 색상 계산:

$$\mathbf{C} = \sum_{i \in N} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j) \tag{5}$$

**수식 설명** — 여러 Gaussian의 색상을 투명도에 따라 합성:
- **$\mathbf{C}$**: 최종 픽셀 색상
- **$\mathbf{c}_i$**: $i$번째 Gaussian의 색상 (SH로 계산)
- **$\alpha_i$**: $i$번째 Gaussian의 불투명도 (2D Gaussian 확률 × 학습된 $\alpha$)
- **$\prod_{j=1}^{i-1}(1 - \alpha_j)$**: 앞쪽 Gaussian들을 통과한 빛의 투과율
  - 예: 앞의 Gaussian이 불투명도 0.5라면, 뒤의 Gaussian은 50%만 보임
- **직관**: 카메라 광선이 여러 반투명 Gaussian을 순서대로 통과하며 색상을 쌓는 물리적 과정

**하늘 처리**: 3D Gaussian은 유클리드 공간에서 정의되므로 하늘과 같은 먼 영역 표현이 어렵습니다. 별도의 고해상도 큐브맵(cubemap)으로 하늘 색상 $\mathbf{C}_{sky}$를 모델링하고 최종 색상과 블렌딩합니다:

$$\mathbf{C} = \mathbf{C}_g + (1 - \mathbf{O}_g) \cdot \mathbf{C}_{sky} \tag{8}$$

**수식 설명**:
- **$\mathbf{C}_g$**: Gaussian들로 렌더링된 색상
- **$\mathbf{O}_g$**: Gaussian 전체 불투명도 (하늘이 얼마나 가려졌는지)
- **$\mathbf{C}_{sky}$**: 큐브맵에서 조회한 하늘 색상
- **직관**: Gaussian이 덮지 못한 픽셀(투명한 부분)에 하늘 색상을 채워 넣음

### 3.3 Training

#### Tracking Pose Optimization

off-the-shelf 트래커가 제공하는 바운딩 박스 포즈는 노이즈가 있습니다. 이를 보정하기 위해, 회전 행렬 $\mathbf{R}_t$와 이동 행렬 $\mathbf{T}_t$에 학습 가능한 변환 $\Delta\mathbf{R}_t, \Delta\mathbf{T}_t$를 추가합니다:

$$\mathbf{R}'_t = \mathbf{R}_t \Delta\mathbf{R}_t$$
$$\mathbf{T}'_t = \mathbf{T}_t + \Delta\mathbf{T}_t \tag{6}$$

**수식 설명**:
- **$\Delta\mathbf{R}_t$**: yaw 오프셋 각도 $\Delta\theta_t$로부터 변환된 회전 행렬 (1개 스칼라 학습)
- **$\Delta\mathbf{T}_t$**: 이동 보정값 (3D 벡터, 학습)
- **직관**: 트래커의 노이즈를 렌더링 품질 손실로 직접 역전파하여 보정. 실험에서 GT 포즈를 쓴 것보다도 더 좋은 결과가 나옴 (GT 어노테이션도 노이즈가 있기 때문)

#### Loss Function

$$\mathcal{L} = \mathcal{L}_{color} + \lambda_1 \mathcal{L}_{depth} + \lambda_2 \mathcal{L}_{sky} + \lambda_3 \mathcal{L}_{sem} + \lambda_4 \mathcal{L}_{reg} \tag{7}$$

**수식 설명** — 5가지 손실 항의 구성:
- **$\mathcal{L}_{color}$**: 렌더링 이미지와 실제 이미지의 $\mathcal{L}_1$ + D-SSIM 손실 (주 훈련 신호)
- **$\mathcal{L}_{depth}$**: 렌더링 깊이와 LiDAR 깊이의 $\mathcal{L}_1$ 손실 ($\lambda_1 = 0.01$)
- **$\mathcal{L}_{sky}$**: 렌더링 불투명도와 예측된 하늘 마스크의 이진 교차 엔트로피 ($\lambda_2 = 0.05$)
- **$\mathcal{L}_{sem}$**: 렌더링 시맨틱 로짓과 2D 세그멘테이션의 소프트맥스 교차 엔트로피 ($\lambda_3 = 0.1$, 선택적)
- **$\mathcal{L}_{reg}$**: 전경 객체 분리를 개선하는 정규화 항 ($\lambda_4 = 0.1$)

**정규화 손실** — 전경 객체의 알파 누적값 $\mathbf{O}_{obj}$에 대한 엔트로피 손실:

$$\mathcal{L}_{reg} = -\sum(\mathbf{O}_{obj}\log\mathbf{O}_{obj} + (1 - \mathbf{O}_{obj})\log(1 - \mathbf{O}_{obj})) \tag{13}$$

**수식 설명**:
- 엔트로피를 최소화하면 $\mathbf{O}_{obj}$가 0 또는 1에 가까워짐
- 즉, 각 Gaussian이 배경이 아니면 완전히 전경 객체에 속하도록 유도
- **직관**: 전경 객체 주변의 '유령 Gaussian(floaters)'을 제거하여 깨끗한 분리 달성

## 🛠️ Chapter 4: Implementation Details

**요약**

- **훈련**: RTX 4090 GPU 1개, Adam optimizer, 30,000 iterations
- **학습률**: $\Delta\mathbf{T}_t$는 $5\times10^{-3}$에서 $5\times10^{-5}$로 감소, $\Delta\mathbf{R}_t$는 $1\times10^{-3}$에서 $1\times10^{-5}$로 감소
- **하늘 큐브맵**: 해상도 1024, 학습률 $1\times10^{-2}$에서 $1\times10^{-4}$로 감소
- **초기화**: 배경은 LiDAR + SfM 포인트 클라우드, 객체는 바운딩 박스 내부 LiDAR 포인트(부족 시 8K 랜덤 샘플링)
- **적응 제어**: 배경 모델 스케일 고정(20m), 객체 모델 스케일은 바운딩 박스 크기로 결정
- **푸리에 계수**: $k=5$ (성능과 저장 비용의 균형)
- **SH 차수**: 1차 (도심 씬의 뷰 의존성이 상대적으로 약하여 오버피팅 방지)

## 📊 주요 실험 및 결과 (Experiments & Results)
- **사용 데이터셋 / 벤치마크**: Waymo Open Dataset(8개 시퀀스, 10Hz, 약 100프레임), KITTI/VKITTI2(MARS 설정을 따라 75%/50%/25% 훈련 비율로 평가). 비교 방법: NSG, MARS, 3D GS, EmerNeRF. 평가 지표: PSNR, SSIM, LPIPS, PSNR*(이동 객체 영역만), FPS

- **Waymo 벤치마크 (1066×1600 해상도)**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | PSNR\*↑ | FPS↑ |
|--------|-------|-------|--------|---------|------|
| 3D GS | 29.64 | 0.918 | 0.117 | 21.25 | **205** |
| NSG | 28.31 | 0.862 | 0.346 | 24.32 | 0.47 |
| MARS | 29.75 | 0.886 | 0.264 | 26.54 | 0.68 |
| EmerNeRF | 30.87 | 0.905 | 0.133 | 21.67 | 0.21 |
| **Ours** | **34.61** | **0.938** | **0.079** | **30.23** | 135 |

- 기존 최고 대비 PSNR +3.74dB, PSNR\* +3.69dB 향상. 렌더링 속도는 NeRF 기반 대비 100배 이상 빠름 (EmerNeRF 대비 642배)

- **KITTI / VKITTI2 벤치마크 (375×1242 해상도)**

| Method | KITTI-75% PSNR↑ | VKITTI2-75% PSNR↑ |
|--------|----------------|-------------------|
| 3D GS | 19.19 | 21.12 |
| NSG | 21.53 | 23.41 |
| MARS | 24.23 | 29.63 |
| **Ours** | **25.79** | **30.10** |

모든 데이터셋, 모든 분할 비율에서 일관된 SOTA 달성.

- **Ablation Study (Waymo)**

| 설정 | PSNR↑ | PSNR\*↑ | SSIM↑ | LPIPS↓ |
|-----|-------|---------|-------|--------|
| w/o LiDAR | 34.02 | 29.53 | 0.934 | 0.087 |
| w/o 4DSH | 34.36 | 29.27 | 0.937 | 0.081 |
| w/o pose opt. | 34.18 | 28.24 | 0.935 | 0.081 |
| w/ GT pose | 34.61 | 29.84 | 0.937 | 0.080 |
| **Complete** | **34.61** | **30.23** | **0.938** | **0.079** |

- Tracking Pose Optimization은 GT 포즈보다도 PSNR\* +0.39dB 높음 (GT 어노테이션에도 노이즈가 있기 때문). 4D SH는 이동 객체의 PSNR\* 향상에 가장 큰 기여. LiDAR는 배경과 이동 객체 모두에서 성능 향상에 기여.
- **응용(Applications)**: 명시적 표현 덕분에 차량 회전·이동·교체 등 씬 편집 가능. NSG는 전경 분리가 불가하고 MARS는 플로터가 발생하는 반면 Street Gaussians는 깨끗한 전경 차량 분리 달성. KITTI에서 시맨틱 세그멘테이션 mIoU 58.81 달성(Video K-Net GT 57.94, rendered 53.81 대비 최고).

- **주요 성과**: Waymo에서 PSNR 34.61, PSNR* 30.23으로 SOTA 대비 대폭 향상. 30분 내 훈련, 135 FPS 실시간 렌더링으로 NeRF 기반 대비 최대 642배 빠른 속도 달성.

## 💡 결론 및 시사점 (Conclusion & Insights)
Street Gaussians는 자율주행 합성 데이터 생성 파이프라인에서 NeRF 기반 방법을 대체하는 유력한 후보로, 실시간 렌더링(135 FPS)은 회귀 테스트 및 대규모 시나리오 생성에 직접 활용 가능하다. 트래커 노이즈를 학습 가능한 파라미터로 흡수해 GT 포즈보다도 좋은 결과를 얻는다는 발견은, 어노테이션 품질에 대한 통념(GT가 항상 최선)을 재고하게 만드는 실용적 인사이트다. 명시적 표현 기반의 씬 편집 능력은 코너 케이스 생성 등 시뮬레이션 활용도를 크게 넓힌다.

- **한계점 및 아쉬운 점**:
  1. 보행자와 같은 비강체(non-rigid) 동적 객체는 처리 불가하며 차량만 지원한다.
  2. off-the-shelf 트래커에 의존하므로 트래커가 차량을 놓치면 품질이 저하된다.
  3. per-scene 최적화 방식이라 일반화(generalizable) 3D Gaussian 방향으로의 확장이 필요하며, 새로운 장면마다 재학습이 필요하다는 점은 실제 대규모 배포에서 비용 부담이 될 수 있다.

---

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [4D Gaussian Splatting](/posts/papers/4d-gaussian-splatting/), [DrivingGaussian](/posts/papers/driving-gaussian-composite-gaussian-splatting/), [OmniRe](/posts/papers/omnire-omni-urban-scene-reconstruction/), [HUGSIM](/posts/papers/hugsim-real-time-photorealistic-closed-loop-simulator/)*
