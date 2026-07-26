---
title: "MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction"
date: 2026-04-20T12:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "HD Map"]
tags: ["HD Map", "Autonomous Driving", "Transformer", "BEV", "Vectorized Map"]
year: 2022
references:
  - "detr-end-to-end-object-detection-with-transformers"
  - "vectormapnet-end-to-end-vectorized-hd-map-learning"
  - "nuscenes-multimodal-dataset-autonomous-driving"
---

## 💡 한 줄 요약
맵 요소를 등가 순열 집합(permutation-equivalent point set)으로 모델링하고 DETR 스타일 병렬 Transformer로 예측함으로써, 카메라 이미지만으로 실시간(25 FPS)에 기존 멀티모달 방법보다 정확한 벡터화 HD 맵을 온라인으로 생성한다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Bencheng Liao, Shaoyu Chen, Xinggang Wang, Tianheng Cheng, Qian Zhang, Wenyu Liu, Chang Huang (Huazhong University of Science & Technology, Horizon Robotics)
- **발행년도**: 2022 (ICLR 2023)
- **주요 기여점**:
  1. 맵 요소를 점 집합 $V$와 등가 순열 그룹 $\Gamma$로 표현하는 permutation-equivalent modeling을 제안하여 순열 모호성(permutation ambiguity) 문제를 원천 해소
  2. 인스턴스 레벨 → 포인트 레벨 순서로 진행되는 2단계 계층적 이분 매칭(hierarchical bipartite matching)으로 예측-GT 매칭 안정화
  3. 인스턴스 쿼리와 포인트 쿼리를 합산하는 계층적 쿼리(hierarchical query) 임베딩과, DETR 스타일 병렬 디코더로 실시간 추론 달성
  4. Point2Point Loss와 Edge Direction Loss를 결합하여 점 위치뿐 아니라 점 간 연결 방향까지 감독
  5. nuScenes에서 카메라 기반 SOTA 대비 5.0 mAP 높고 8배 빠른 실시간 추론(25.1 FPS)을 달성하며, LiDAR 포함 멀티모달 방법보다도 높은 성능 확보

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: HD 맵 구축은 SLAM 기반 오프라인 방식에서, 차량 탑재 센서로 실시간 구축하는 온라인 방식으로 발전해왔습니다. 온라인 방식 중에서도 HDMapNet 같은 BEV 세그멘테이션(래스터 맵) 기반 방법에서, VectorMapNet 같은 end-to-end 벡터화 방법으로 이어졌으나 자기회귀 디코더의 속도 한계가 있었습니다.
- **기존 한계점**:
  1. 기존 HD 맵은 SLAM 기반 오프라인 방식으로 구축되어 복잡한 파이프라인과 높은 유지비용이 필요함
  2. BEV 세그멘테이션 기반 방식(BEVFormer 등)은 래스터 맵을 생성하지만, 차선 구조 같은 인스턴스 레벨 벡터 정보가 없어 모션 예측·플래닝에 한계가 있음
  3. 최초 end-to-end 벡터화 방식인 VectorMapNet은 포인트를 순차적으로 예측하는 자기회귀 디코더를 써서 추론이 느리고 순열 모호성 문제가 있음
- **이 논문의 접근 방식**: 맵 요소를 "등가 순열 집합(permutation-equivalent point set)"으로 모델링하는 통합 표현을 제안하고, DETR 스타일 병렬 Transformer로 실시간 추론을 달성합니다.

## 📑 목차
- Section 1: Introduction
- Section 2: Related Work
- Section 3.1: Permutation-equivalent Modeling
- Section 3.2: Hierarchical Matching
- Section 3.3: Training Loss
- Section 3.4: Architecture
- Section 4: Experiments

## 🛠️ Section 1: Introduction

**요약**
HD(High-Definition) 맵은 자율주행 플래닝의 핵심 인프라입니다. 기존에는 오프라인으로 구축했지만, 차량 탑재 센서로 실시간 구축하는 온라인 방식이 주목받고 있습니다. 기존 온라인 방법들은 래스터 맵(픽셀 단위 세그멘테이션)이나 느린 자기회귀 벡터화에 머물러 실시간 적용이 어렵습니다.

MapTR는 DETR 패러다임을 HD 맵 구축에 도입한 최초의 실시간 SOTA 방법으로, 카메라 입력만으로 기존 멀티모달 방법을 능가합니다.

**핵심 개념**
- **HD Map (고정밀 지도)**: 차선 경계, 보행자 횡단보도, 도로 경계 등을 인스턴스 레벨 벡터로 표현한 고정밀 지도
- **Vectorized Map**: 픽셀이 아닌 점(Point)과 선(Edge)으로 구성된 맵 — 다운스트림 모션 예측·플래닝에 바로 활용 가능
- **Online Construction**: 사전 제작 없이 주행 중 실시간으로 맵을 생성
- **DETR 패러다임**: Detection TRansformer — 학습 가능한 쿼리(query)가 병렬로 모든 객체를 한 번에 예측하는 end-to-end 구조

## 🛠️ Section 2: Related Work

**요약**
관련 연구는 크게 HD 맵 구축, 차선 감지, 컨투어 기반 인스턴스 세그멘테이션 세 분야로 나뉩니다. BEV 세그멘테이션 방법들은 래스터 맵만 생성하고, VectorMapNet은 첫 end-to-end 벡터화 방법이지만 자기회귀 디코더로 인해 속도가 느립니다. MapTR은 이 한계를 DETR 스타일 병렬 디코더와 통합 모델링으로 극복합니다.

**핵심 개념**
- **HDMapNet**: 픽셀 단위 세그멘테이션 후 후처리로 인스턴스를 생성 — 속도 느리고 복잡
- **VectorMapNet**: 첫 end-to-end 벡터화 HD 맵 방법, 2단계 coarse-to-fine + 자기회귀 디코더
- **BEVFormer**: 시공간 Transformer로 BEV 특징 생성 — MapTR의 BEV 인코더에도 영향

## 🛠️ Section 3.1: Permutation-equivalent Modeling (순열-등가 모델링)

**요약**
맵 요소(차선, 보행자 횡단보도 등)를 점 집합으로 표현할 때 핵심 문제는 순열 모호성(permutation ambiguity)입니다. 예를 들어, 차선(polyline)은 양 끝점 중 어느 것을 시작점으로 해도 같은 형태이고, 보행자 횡단보도(polygon)는 임의의 점에서 시계/반시계 방향으로 순회해도 같은 도형입니다.

기존 방법은 임의로 하나의 순열을 고정("vanilla" 방식)하여 모호성을 무시했고, 이는 학습 불안정을 야기했습니다. MapTR은 맵 요소를 $\mathcal{V} = (V, \Gamma)$로 표현합니다.

**핵심 개념**
- **$V = \{v_j\}_{j=0}^{N_v - 1}$**: 맵 요소의 점 집합 ($N_v$개의 점)
- **$\Gamma = \{\gamma^k\}$**: 점 집합 $V$의 등가 순열 그룹 — 기하학적으로 동일한 모든 순서 배치를 포함
- **Polyline (개방형)**: 방향이 2가지 (정방향, 역방향) → $|\Gamma| = 2$
- **Polygon (폐쇄형)**: 임의의 시작점 × 2방향 → $|\Gamma| = 2 \times N_v$

**수식 예제**

$$\Gamma_{\text{polyline}} = \{\gamma^0, \gamma^1\} \quad \begin{cases} \gamma^0(j) = j \mod N_v \\ \gamma^1(j) = (N_v - 1 - j) \mod N_v \end{cases}$$

**수식 설명**
이 수식은 차선(polyline)의 두 가지 등가 표현을 정의합니다:
- **$\gamma^0$**: 원래 순서 그대로 (점 0 → 1 → 2 → ...)
- **$\gamma^1$**: 역방향 순서 (점 마지막 → ... → 1 → 0)
- 두 표현 모두 기하학적으로 같은 차선이므로, 둘 다 정답으로 허용

**수식 예제**

$$\Gamma_{\text{polygon}} = \{\gamma^0, \ldots, \gamma^{2 \times N_v - 1}\} \quad \begin{cases} \gamma^0(j) = j \mod N_v \\ \gamma^1(j) = (N_v - 1 - j) \mod N_v \\ \gamma^2(j) = (j + 1) \mod N_v \\ \gamma^3(j) = (N_v - 1 - (j+1)) \mod N_v \\ \vdots \end{cases}$$

**수식 설명**
보행자 횡단보도(polygon)는 훨씬 더 많은 등가 표현이 있습니다:
- **$\gamma^0$**: 원래 순서
- **$\gamma^1$**: 역방향 (반시계 ↔ 시계 방향 전환)
- **$\gamma^2$**: 시작점을 한 칸 이동한 순서 (0 → 1 → 2 → ... 대신 1 → 2 → 0 → ...)
- 총 $2 \times N_v$가지 모두 같은 도형 — 모두 정답으로 허용

## 🛠️ Section 3.2: Hierarchical Matching (계층적 매칭)

**요약**
MapTR은 DETR처럼 N개의 맵 요소를 병렬로 한 번에 예측합니다. 학습 시 예측과 GT를 올바르게 매칭하기 위해 2단계 계층적 이분 매칭을 수행합니다.

1. **인스턴스 레벨 매칭**: 예측된 N개 요소와 GT 요소를 헝가리안 알고리즘으로 1:1 매칭
2. **포인트 레벨 매칭**: 매칭된 각 인스턴스 쌍에서 등가 순열 그룹 $\Gamma$ 중 최적 순열 $\hat{\gamma}$를 선택

**핵심 개념**
- **Instance-level Matching**: 클래스 레이블 + 포인트 위치를 함께 고려한 비용 함수로 최적 인스턴스 매칭
- **Point-level Matching**: 허용된 모든 순열 중 Manhattan 거리 합이 가장 작은 순열을 정답 순열로 선택

**수식 예제**

$$\mathcal{L}_{\text{ins\_match}}(\hat{y}_{\pi(i)}, y_i) = \mathcal{L}_{\text{Focal}}(\hat{p}_{\pi(i)}, c_i) + \mathcal{L}_{\text{position}}(\hat{V}_{\pi(i)}, V_i)$$

**수식 설명**
- **$\mathcal{L}_{\text{Focal}}$**: 분류 비용 — 예측 클래스 확률과 GT 클래스 레이블의 Focal Loss
- **$\mathcal{L}_{\text{position}}$**: 위치 비용 — 예측 점 집합과 GT 점 집합의 위치 상관 비용
- **$\hat{p}_{\pi(i)}$**: $\pi(i)$번째 예측 인스턴스의 분류 점수
- **$c_i$**: i번째 GT의 클래스 레이블

**수식 예제**

$$\hat{\gamma} = \arg\min_{\gamma \in \Gamma} \sum_{j=0}^{N_v - 1} D_{\text{Manhattan}}(\hat{v}_j, v_{\gamma(j)})$$

**수식 설명**
- **$\hat{v}_j$**: 예측된 j번째 점
- **$v_{\gamma(j)}$**: GT의 $\gamma(j)$번째 점 (순열 $\gamma$에 따라 재배열된 GT 점)
- **$D_{\text{Manhattan}}$**: 맨해튼 거리 (|x1-x2| + |y1-y2|) — L1 거리로 이상치에 강건
- 모든 허용 순열 중 거리 합이 최소인 것을 선택 → 학습 시 가장 적합한 GT 배열 사용

## 🛠️ Section 3.3: Training Loss (학습 손실 함수)

**요약**
MapTR의 전체 손실 함수는 3가지 항으로 구성됩니다: 분류 손실, point2point 손실, 에지 방향 손실.

**핵심 개념**
- **$\mathcal{L}_{\text{cls}}$**: 분류 손실 (Focal Loss) — 어떤 카테고리인지 맞추는 손실
- **$\mathcal{L}_{\text{p2p}}$**: Point-to-point 손실 — 각 점의 위치를 정확히 맞추는 손실
- **$\mathcal{L}_{\text{dir}}$**: 에지 방향 손실 — 점 간 연결선(에지)의 방향을 맞추는 손실

**수식 예제**

$$\mathcal{L} = \lambda \mathcal{L}_{\text{cls}} + \alpha \mathcal{L}_{\text{p2p}} + \beta \mathcal{L}_{\text{dir}}$$

**수식 설명**
- **$\lambda, \alpha, \beta$**: 각 손실 항의 가중치 (균형 조절)

**수식 예제**

$$\mathcal{L}_{\text{p2p}} = \sum_{i=0}^{N-1} \mathbb{1}_{\{c_i \neq \varnothing\}} \sum_{j=0}^{N_v - 1} D_{\text{Manhattan}}(\hat{v}_{\hat{\pi}(i), j},\ v_{i, \hat{\gamma}_i(j)})$$

**수식 설명**
- **$\mathbb{1}_{\{c_i \neq \varnothing\}}$**: GT가 "빈 객체"가 아닐 때만 계산 (빈 패딩 제외)
- **$\hat{v}_{\hat{\pi}(i), j}$**: 매칭된 예측 인스턴스의 j번째 예측 점
- **$v_{i, \hat{\gamma}_i(j)}$**: GT의 최적 순열로 재배열된 j번째 GT 점
- 각 점 쌍의 맨해튼 거리 합산 → 점 위치 정확도 향상

**수식 예제**

$$\mathcal{L}_{\text{dir}} = -\sum_{i=0}^{N-1} \mathbb{1}_{\{c_i \neq \varnothing\}} \sum_{j=0}^{N_v - 1} \text{cosine\_similarity}(\hat{e}_{\hat{\pi}(i), j},\ e_{i, \hat{\gamma}_i(j)})$$

**수식 설명**
에지(점과 다음 점을 잇는 선분)의 방향 벡터를 코사인 유사도로 비교합니다:
- **$\hat{e}_{\hat{\pi}(i), j} = \hat{v}_{\hat{\pi}(i), j} - \hat{v}_{\hat{\pi}(i), (j+1) \mod N_v}$**: 예측 에지 벡터 (j번 점 → 다음 점)
- **$e_{i, \hat{\gamma}_i(j)}$**: GT 에지 벡터
- **cosine_similarity**: 두 벡터의 방향이 같을수록 1에 가까워짐 (음수를 붙여 최소화 목표)
- Point2point는 점의 위치만 보지만, Edge direction은 점 사이 연결 방향까지 감독 → 더 정확한 형상 재현

## 🛠️ Section 3.4: Architecture (아키텍처)

**요약**
MapTR은 Map Encoder + Map Decoder의 인코더-디코더 구조를 사용합니다.

**Map Encoder**: 다중 카메라 이미지 → BEV 특징 맵
- 백본(ResNet18/50)으로 멀티뷰 이미지 특징 추출
- GKT(Geometry-guided Kernel Transformer)로 2D → BEV 변환
- 출력: $\mathcal{B} \in \mathbb{R}^{H \times W \times C}$의 BEV 특징

**Map Decoder**: 계층적 쿼리 임베딩으로 맵 요소 예측
- **인스턴스 쿼리** $q_i^{\text{ins}}$: 각 맵 요소 전체를 표현
- **포인트 쿼리** $q_j^{\text{pt}}$: 각 점의 위치를 표현 (모든 인스턴스가 공유)
- **계층적 쿼리**: $q_{ij}^{\text{hie}} = q_i^{\text{ins}} + q_j^{\text{pt}}$ — 인스턴스와 포인트 정보를 동시에 인코딩

**핵심 개념**
- **MHSA (Multi-Head Self-Attention)**: 쿼리들끼리 서로 정보 교환 (인스턴스 간, 점 간 모두)
- **Deformable Attention**: 각 쿼리의 레퍼런스 포인트 주변 BEV 특징만 선택적으로 참조 — 계산 효율적
- **Prediction Head**: 분류 브랜치(클래스 점수) + 점 회귀 브랜치($2N_v$차원 BEV 좌표 벡터)

**수식 예제**

$$q_{ij}^{\text{hie}} = q_i^{\text{ins}} + q_j^{\text{pt}}$$

**수식 설명**
- **$q_i^{\text{ins}}$**: i번째 맵 요소 전체의 맥락 (어떤 맵 요소인지)
- **$q_j^{\text{pt}}$**: j번째 점의 위치 정보 (요소 내 몇 번째 점인지)
- 두 쿼리를 더하면 "i번째 요소의 j번째 점"에 대한 계층적 표현이 됨
- 이 쿼리들이 MHSA(Multi-Head Self-Attention)와 Deformable Attention으로 BEV 특징과 상호작용

## 📊 주요 실험 및 결과 (Experiments & Results)
- **사용 데이터셋 / 벤치마크**: nuScenes 데이터셋(1000개 장면, 6카메라, 360° FOV)에서 보행자 횡단보도, 차선 구분선, 도로 경계 3가지 맵 요소를 평가. 평가지표는 Chamfer 거리 기반 Average Precision(AP)

**주요 성능 비교**

| 방법 | 입력 | Backbone | mAP | FPS |
|------|------|----------|-----|-----|
| VectorMapNet-C | Camera | R50 | 40.9 | 2.9 |
| **MapTR-nano** | **Camera** | **R18** | **45.9** | **25.1** |
| **MapTR-tiny** | **Camera** | **R50** | **50.3** | **11.2** |
| **MapTR-tiny (110 epoch)** | **Camera** | **R50** | **58.7** | **11.2** |
| VectorMapNet C&L | Camera+LiDAR | - | 45.2 | - |

**주요 성과**:
- MapTR-nano: 카메라 기반 SOTA보다 5.0 mAP 높고 8배 빠른 실시간 추론(25.1 FPS)
- MapTR-tiny(110 epoch): LiDAR 포함 멀티모달 방법보다도 13.5 mAP 높음
- Permutation-equivalent 모델링: vanilla 대비 +5.9 mAP (보행자 횡단보도 +11.9 mAP)
- 에지 방향 손실 효과($\beta = 5 \times 10^{-3}$): 미사용 시 mAP 48.2 → 적용 시 50.3

## 💡 결론 및 시사점 (Conclusion & Insights)
MapTR은 HD 맵 구축에서 두 가지 핵심 혁신을 달성했습니다: 맵 요소를 등가 순열 집합으로 모델링하여 형상 모호성을 원천적으로 해소함으로써 학습 안정성과 정확도를 동시에 향상시켰고, DETR 스타일 병렬 Transformer + 계층적 쿼리로 실시간 추론 가능한 최초 SOTA 벡터화 맵 구축 시스템을 구현했습니다.

- 합성 데이터 생성 시 벡터화 HD 맵을 자동으로 얻을 수 있어 annotation 비용이 절감됩니다.
- 플래닝·모션 예측에 바로 사용 가능한 벡터 표현으로, 다운스트림 태스크와 직접 연동됩니다.
- 카메라만으로 LiDAR 수준 맵 품질을 달성하여 저비용 센서 구성에서도 활용 가능합니다.
- MapTR 이후 MapTRv2 등으로 발전하며 온라인 맵 구축의 표준 프레임워크로 자리잡았습니다.
- **한계점 및 아쉬운 점**: permutation-equivalent 모델링은 맵 요소의 점 개수 $N_v$가 고정되어야 하는 제약이 있어, 매우 복잡하거나 불규칙한 형상은 표현력이 제한될 수 있습니다. 또한 실험이 nuScenes 도심 환경에 집중되어 있어, 고속도로나 복잡한 교차로 등 다양한 도로 토폴로지에서의 일반화 성능은 추가 검증이 필요합니다.

---

*관련 논문: [VectorMapNet](/posts/papers/vectormapnet-end-to-end-vectorized-hd-map-learning/), [DETR](/posts/papers/detr-end-to-end-object-detection-with-transformers/), [BEVFormer](/posts/papers/BEVFormer/), [StreamMapNet](/posts/papers/streammapnet-streaming-mapping-network-vectorized-online-hd-map-construction/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
