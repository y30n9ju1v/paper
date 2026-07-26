---
title: "DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes"
date: 2026-04-10T10:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "Autonomous Driving", "Novel View Synthesis", "LiDAR"]
year: 2024
references:
  - "3d-gaussian-splatting"
  - "street-gaussians-modeling-dynamic-urban-scenes"
  - "nuscenes-multimodal-dataset-autonomous-driving"
---

## 💡 한 줄 요약
DrivingGaussian은 정적 배경을 점진적으로 쌓아가는 Incremental Static 3D Gaussians와 동적 객체를 그래프로 관리하는 Composite Dynamic Gaussian Graph를 결합해, 자율주행 서라운드 뷰(멀티카메라) 동적 장면을 nuScenes 기준 PSNR 28.74로 기존 최고 대비 2dB 이상 향상시키며 재구성한다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, Ming-Hsuan Yang (Peking University, Google Research, UC Merced)
- **발행년도**: 2024 (arXiv:2312.07920v3)
- **주요 기여점**:
  1. 정적 배경을 LiDAR 깊이 범위 기준 N개 bin으로 나눠 점진적으로 재구성하는 Incremental Static 3D Gaussians 제안
  2. 여러 동적 객체를 그래프 구조로 개별 모델링 후 통합하는 Composite Dynamic Gaussian Graph 제안
  3. LiDAR 포인트를 SfM 대신 Gaussian 초기화 프라이어로 활용하고 멀티카메라 Bundle Adjustment로 정밀도 향상
  4. 재구성된 Gaussian 필드에 임의 객체를 삽입하는 코너 케이스 시뮬레이션 지원으로 안전 검증까지 확장

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: 제한된 공간의 NeRF(MipNeRF, Point-NeRF) → 대규모 무한 공간을 다루는 Unbounded NeRF(Urban-NeRF, EmerNeRF) → 빠른 명시적 렌더링을 위한 3D Gaussian Splatting(3D-GS) → 단일 객체 동적 확장(HexPlane, D-NeRF) 순으로 발전해 왔으나, 멀티카메라 서라운드 뷰의 자율주행 동적 장면 전용 확장은 부재했다.
- **기존 한계점**:
  1. 멀티카메라 환경에서 다양한 조명 변화 및 뷰 차이에 취약 (NeRF의 ray sampling 의존성)
  2. LiDAR를 보조 깊이 감독으로만 활용하고 기하학적 프라이어로는 미활용
  3. 정적 장면 가정으로 빠르게 움직이는 동적 객체 표현 불가
- **이 논문의 접근 방식**: 정적 배경은 Incremental Static 3D Gaussians로 점진적·순차적으로, 동적 객체는 Composite Dynamic Gaussian Graph로 개별 재구성한 뒤 결합하며, LiDAR를 Gaussian 초기화의 기하학적 프라이어로 적극 활용한다.

## 📑 목차
- Chapter 1: Introduction
- Chapter 2: Related Work
- Chapter 3: Method (Composite Gaussian Splatting, LiDAR Prior, Global Rendering)
- Chapter 4: Experiments
- Chapter 5: Conclusion

## 🛠️ Chapter 1: Introduction

**요약**

자율주행 시스템은 대규모 동적 3D 장면을 정확히 모델링해야 합니다. 기존 NeRF 기반 방법들은 다음의 한계가 있었습니다:
- 멀티카메라 환경에서 다양한 조명 변화 및 뷰 차이에 취약
- LiDAR를 보조 깊이 감독으로만 활용하고 기하학적 프라이어로는 미활용
- 정적 장면 가정으로 빠르게 움직이는 동적 객체 표현 불가

DrivingGaussian은 이 문제를 해결하기 위해 Composite Gaussian Splatting을 도입합니다:
1. **Incremental Static 3D Gaussians**: 정적 배경을 순차적·점진적으로 재구성
2. **Composite Dynamic Gaussian Graph**: 여러 동적 객체를 개별적으로 재구성한 후 씬에 통합
3. **LiDAR Prior**: 보다 정확한 기하 구조와 멀티카메라 일관성 유지

**핵심 개념**

- **서라운드 뷰 합성(Surrounding View Synthesis)**: 차량 주변 6방향 카메라 이미지를 동시에 고품질로 생성하는 기술
- **Composite Gaussian Splatting**: 정적 배경과 동적 객체를 각각 독립적으로 Gaussian으로 모델링한 후 합성하는 방식
- **NeRF 기반 방법의 한계**: Ray sampling에 의존해 멀티카메라 환경에서 품질 저하 발생

## 🛠️ Chapter 2: Related Work

**요약**

기존 방법들을 크게 세 범주로 정리합니다.

**NeRF for Bounded Scenes**: MipNeRF, Point-NeRF 등은 제한된 공간에서 좋은 성능을 보이지만 자율주행의 대규모 무한 공간에는 적용 어려움.

**NeRF for Unbounded Scenes**: Urban-NeRF, EmerNeRF 등이 대규모 도시 장면 모델링을 시도하지만, 동적 객체를 충분히 처리하지 못하거나 LiDAR 활용이 제한적.

**3D Gaussian Splatting (3D-GS)**: 명시적 표현으로 빠른 렌더링과 미분가능한 최적화가 가능하지만, 원본 3D-GS는 정적 장면용으로 설계됨.

**Dynamic 3D Gaussian Splatting**: HexPlane, D-nerf 등이 동적 단일 객체 씬으로 확장했으나, 멀티카메라 자율주행 씬에는 적합하지 않음.

**핵심 개념**

- **3D Gaussian Splatting (3D-GS)**: 3D 공간을 수백만 개의 Gaussian 타원체로 표현하고, 이를 2D 이미지 평면에 투영(splatting)해 빠른 렌더링을 가능하게 하는 방법
- **미분가능 렌더링(Differentiable Rendering)**: 렌더링 과정 자체를 미분 가능하게 만들어 역전파로 Gaussian 파라미터를 최적화
- **SfM (Structure-from-Motion)**: 여러 이미지에서 카메라 포즈와 희소 3D 포인트 클라우드를 동시에 추정하는 기법

## 🛠️ Chapter 3: Method

### 3.1 Composite Gaussian Splatting

**요약**

DrivingGaussian의 핵심 구조는 두 컴포넌트로 구성됩니다.

#### Incremental Static 3D Gaussians

자율주행 데이터는 차량이 이동하면서 넓은 범위를 촬영하므로, 전체 장면을 한 번에 모델링하면 먼 과거 프레임과 현재 프레임의 혼동이 발생합니다. 이를 해결하기 위해:

- LiDAR 깊이 범위를 기준으로 전체 씬을 **N개의 bin**으로 균등 분할
- 첫 번째 bin은 LiDAR prior로 Gaussian 모델 초기화 (식 1)
- 이후 bin들은 이전 bin의 Gaussian을 포지션 프라이어로 활용 (식 2)
- 멀티카메라 이미지의 겹치는 영역을 공동 정렬에 활용

**수식 예제**

$$p_{b_0}(l|\mu, \Sigma) = e^{-\frac{1}{2}(l-\mu)^\top \Sigma^{-1}(l-\mu)}$$

**수식 설명** — LiDAR 포인트 클라우드로 초기 Gaussian을 생성하는 확률 밀도 함수:
- **$l \in \mathbb{R}^3$**: LiDAR prior의 위치 (3D 공간의 x, y, z 좌표)
- **$\mu$**: LiDAR 포인트들의 평균 위치
- **$\Sigma \in \mathbb{R}^{3\times3}$**: 이방성 공분산 행렬 (Gaussian의 형태와 방향을 결정)
- **직관**: 이 수식은 LiDAR 포인트 주변에 Gaussian 구름을 뿌려 초기 3D 구조를 만드는 것

$$\tilde{P}_{b+1}(G_s) = P_b(G_s) \bigcup (x_{b+1}, y_{b+1}, z_{b+1})$$

**수식 설명** — 이전 bin의 Gaussian을 다음 bin의 프라이어로 활용:
- **$\tilde{P}_{b+1}(G_s)$**: b+1번째 bin을 위한 초기 Gaussian 위치 집합
- **$P_b(G_s)$**: b번째 bin에서 학습된 Gaussian 위치
- **$(x_{b+1}, y_{b+1}, z_{b+1})$**: b+1 구역 내의 새로운 LiDAR 좌표
- **직관**: 이미 알고 있는 장면(이전 bin)을 기반으로 새로운 부분을 점진적으로 확장

Incremental Static Gaussian의 렌더링 색상:

$$\tilde{C}(G_s) = \sum_{b=1}^{N} \Gamma_b \, \alpha_b \, C_b, \quad \Gamma_b = \prod_{i=1}^{b-1}(1 - \alpha_i)$$

**수식 설명** — 알파 합성(alpha compositing)으로 여러 bin의 Gaussian을 합쳐 최종 색상 계산:
- **$\tilde{C}(G_s)$**: 해당 카메라 시점의 최종 렌더링 색상
- **$\alpha_b$**: b번째 bin Gaussian의 불투명도 (0=완전 투명, 1=완전 불투명)
- **$C_b$**: b번째 bin의 색상
- **$\Gamma_b$**: b번째 bin에 도달하는 빛의 투과율 (앞쪽 bin들이 얼마나 막는지)
  - 예: 앞 bin이 불투명도 0.3이면 뒤 bin은 70%만 보임
- **직관**: 빛이 여러 레이어를 통과할 때 각 레이어에서 흡수·반사되는 물리 현상 모방

멀티카메라 정렬을 위한 최적 색상:

$$\hat{C} = \varsigma(G_s) \sum \omega(\tilde{C}(G_s) | R, T)$$

**수식 설명**:
- **$\hat{C}$**: 최적화된 픽셀 색상
- **$\varsigma$**: 미분가능 splatting 함수
- **$\omega$**: 서로 다른 카메라 뷰에 대한 가중치
- **$[R, T]$**: 뷰 정렬을 위한 회전·이동 행렬 (카메라 외부 파라미터)

#### Composite Dynamic Gaussian Graph

동적 객체 처리를 위해 그래프 구조를 도입:

$$H = \langle O, G_d, M, P, A, T \rangle$$

**수식 설명** — 동적 Gaussian 그래프의 구성 요소:
- **$O$**: 인스턴스 객체 집합 (각 차량, 보행자 등)
- **$G_d$**: 각 객체에 대응하는 동적 Gaussian
- **$M$**: 각 객체의 변환 행렬 (위치, 방향)
- **$P$**: 바운딩 박스 중심 좌표
- **$A$**: 바운딩 박스 방향
- **$T$**: 각 객체가 등장하는 시간 스텝 집합

각 동적 객체의 좌표계 변환:

$$m_o^{-1} = R_o^{-1} S_o^{-1}$$

**수식 설명**:
- **$m_o^{-1}$**: 객체의 월드 좌표계 → 객체 좌표계 역변환
- **$R_o^{-1}$**: 회전 역행렬
- **$S_o^{-1}$**: 스케일 역행렬
- **직관**: 각 차량을 자신만의 로컬 좌표계에서 독립적으로 모델링하여, 차량이 이동해도 일관된 형태 유지

가려짐(occlusion) 처리를 위한 동적 객체 불투명도:

$$\alpha_{o,t} = \sum \frac{(p_t - b_o)^2 \cdot \cot \alpha_o}{[\rho(b_o R_{o,t} S_{o,t}) - \rho]^2} \pi_{p_0}$$

**수식 설명** — 카메라에서 가까울수록 더 불투명하게 처리:
- **$\alpha_{o,t}$**: 시각 t에서 객체 o의 조정된 불투명도
- **$p_t$**: 시각 t에서의 카메라 중심
- **$b_o$**: 객체 바운딩 박스 중심
- **$\rho$**: 카메라 중심까지의 거리
- **직관**: 빛의 전파 원리에 따라 가까운 객체가 뒤의 객체를 가리는 현상을 구현

최종 복합 Gaussian 필드:

$$G_{comp} = \sum H \langle O, G_d, M, P, A, T \rangle + G_s$$

**수식 설명**:
- **$G_{comp}$**: 정적 배경과 모든 동적 객체를 합친 최종 Gaussian 필드
- **$G_s$**: Incremental Static 3D Gaussians (정적 배경)
- **$\sum H$**: 그래프의 모든 동적 객체 Gaussian

### 3.2 LiDAR Prior with surrounding views

**요약**

원래 3D-GS는 SfM으로 Gaussian을 초기화하지만, 자율주행의 대규모 무한 배경에서는 SfM 포인트가 너무 희소합니다. DrivingGaussian은 LiDAR 포인트 클라우드를 프라이어로 활용합니다:

1. 여러 LiDAR 스윕을 합쳐 완전한 포인트 클라우드 $L$ 생성
2. 각 LiDAR 포인트를 카메라 이미지로 투영하여 색상 할당

**수식 예제**

$$x_p^q = K[R_t^q \cdot l_s + T_t^q]$$

**수식 설명**:
- **$x_p^q$**: 이미지 q에서의 2D 픽셀 좌표
- **$K \in \mathbb{R}^{3\times3}$**: 카메라 내부 파라미터 행렬 (초점거리, 주점 등)
- **$R_t^q, T_t^q$**: 시각 t에서 카메라 q의 회전·이동 행렬 (외부 파라미터)
- **$l_s$**: LiDAR 포인트의 3D 위치
- **직관**: 3D 공간의 LiDAR 포인트를 2D 카메라 이미지에 "찍어서" 색상을 얻는 과정

추가로 **Multi-camera Bundle Adjustment (DBA)**를 적용하여 LiDAR 포인트의 정확도를 높이고 Gaussian 기하 구조를 개선합니다.

**핵심 개념**

- **LiDAR Prior**: LiDAR 센서의 정확한 깊이 정보를 Gaussian 초기화에 활용하여 기하 구조의 정확도 향상
- **Bundle Adjustment**: 카메라 포즈와 3D 포인트를 동시에 최적화하는 기법. 멀티카메라로 확장하면 모든 카메라의 일관성 보장 가능
- **멀티카메라 일관성**: 여러 카메라가 겹치는 영역에서 동일한 3D 구조가 일관되게 나타나야 함

### 3.3 Global Rendering via Gaussian Splatting

**요약**

복합 Gaussian 필드 $G_{comp}$를 2D 이미지로 렌더링합니다.

**수식 예제**

$$\tilde{\Sigma} = JE\Sigma E^\top J^\top$$

**수식 설명**:
- **$\tilde{\Sigma}$**: 2D 이미지 평면에서의 Gaussian 공분산 (타원 모양 결정)
- **$\Sigma$**: 3D 공간에서의 Gaussian 공분산
- **$J$**: 원근 투영의 야코비안 행렬 (3D→2D 변환의 국소 선형 근사)
- **$E$**: 월드→카메라 좌표계 변환 행렬
- **직관**: 3D 타원체를 카메라로 바라볼 때 2D 이미지에서 어떤 타원으로 보이는지 계산

**손실 함수**는 세 가지로 구성:

1. **Tile Structural Similarity Loss** (TSSIM):

$$L_{TSSIM}(\delta) = 1 - \frac{1}{Z} \sum SSIM(\Psi(\hat{C}), \Psi(C))$$

**수식 설명**:
- **$\hat{C}$**: 렌더링된 타일
- **$C$**: 실제 이미지 타일 (ground truth)
- **$\Psi$**: 화면을 $M$개 타일로 분할하는 함수
- **$SSIM$**: 구조적 유사도 지수 (밝기, 대비, 구조를 동시에 비교, 1이 완벽한 일치)
- **직관**: 픽셀 단위 차이뿐 아니라 이미지의 구조적 패턴도 보존되도록 학습

2. **Robust Loss** (이상치 제거):

$$L_{Robust}(\delta) = \kappa(\|I - \hat{I}\|_2)$$

**수식 설명**:
- **$\kappa \in [0,1]$**: 이상치 내성을 조절하는 shape 파라미터
- **$I$**: 실제 이미지, **$\hat{I}$**: 합성 이미지
- **직관**: 일반 L2 손실 대신 Barron robust loss를 사용해 Gaussian 이상치에 덜 민감하게 학습

3. **LiDAR Loss** (기하 감독):

$$L_{LiDAR}(\delta) = \frac{1}{S} \sum \|P(G_{comp}) - L_s\|^2$$

**수식 설명**:
- **$P(G_{comp})$**: 복합 Gaussian의 3D 위치
- **$L_s$**: LiDAR 포인트 위치
- **직관**: Gaussian 위치가 실제 LiDAR 측정값과 가까워지도록 강제하여 정확한 기하 구조 유지

## 📊 주요 실험 및 결과 (Experiments & Results)
- **사용 데이터셋 / 벤치마크**: nuScenes(1000개 드라이빙 씬, 6카메라 + 1 LiDAR, 23개 객체 클래스, 6개 도전적 씬의 키프레임 320K+ 이미지·포인트 클라우드 사용), KITTI-360(멀티센서 데이터셋, 단안 카메라 씬 검증)
- **주요 성과**:
  - nuScenes: EmerNeRF(PSNR 26.75), 3D-GS(26.08) 대비 Ours-L이 **PSNR 28.74, SSIM 0.865, LPIPS 0.237**로 최고 성능 (기존 최고 대비 PSNR 약 2dB 이상 향상)
  - KITTI-360: DNMP(PSNR 23.41) 대비 Ours-L **PSNR 25.62**
  - Ablation: LiDAR-2M 초기화(PSNR 28.78)가 랜덤 초기화(22.18), SfM 초기화(28.51) 대비 최고. Composite Dynamic Gaussian Graph(CDGG) 제거 시 PSNR 26.97로 급락해 동적 객체 처리 모듈의 기여도가 큼을 확인
  - 코너 케이스 시뮬레이션: 재구성된 Gaussian 필드에 임의 객체를 삽입해 보행자 낙상, 차량 근접 등 시간적·센서 간 일관성을 유지한 안전 검증 시나리오 생성 가능

## 💡 결론 및 시사점 (Conclusion & Insights)
DrivingGaussian은 NeRF 기반 방법 대신 Gaussian Splatting 기반 복합 표현을 채택하면 자율주행 서라운드 뷰 합성에서 렌더링 속도와 품질을 모두 향상시킬 수 있음을 보여준다. 특히 LiDAR를 단순 깊이 감독이 아닌 Gaussian 초기화 프라이어로 활용하는 방식이 멀티카메라 일관성 문제 해결에 효과적이다. 정적/동적 분리 모델링 구조는 이후 Street Gaussians, OmniRe 등 후속 연구의 공통 설계 패턴으로 자리잡았다.

- **한계점 및 아쉬운 점**:
  1. 동적 객체 모델링이 주로 차량 등 강체(rigid) 위주로 설명되며, 보행자 같은 비강체 객체에 대한 처리는 상대적으로 深이 얕다.
  2. Incremental Static Gaussian의 bin 개수 $N$ 선택 기준이나 민감도에 대한 분석이 부족해 실제 적용 시 하이퍼파라미터 튜닝 부담이 있을 수 있다.
  3. 실시간 렌더링 속도(FPS)에 대한 정량적 수치가 본문에 제시되지 않아, Street Gaussians·HUGSIM 등과의 직접적인 속도 비교가 어렵다.

---

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [4D Gaussian Splatting](/posts/papers/4d-gaussian-splatting/), [Street Gaussians](/posts/papers/street-gaussians-modeling-dynamic-urban-scenes/), [OmniRe](/posts/papers/omnire-omni-urban-scene-reconstruction/), [HUGSIM](/posts/papers/hugsim-real-time-photorealistic-closed-loop-simulator/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
