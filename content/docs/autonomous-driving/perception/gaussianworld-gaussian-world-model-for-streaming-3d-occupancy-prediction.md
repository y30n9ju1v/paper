---
title: "GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction"
date: 2026-04-24T13:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "3D Occupancy"]
tags: ["3D Gaussian Splatting", "Occupancy Prediction", "World Model", "Temporal Modeling", "Streaming", "nuScenes"]
year: 2024
references:
  - "3d-gaussian-splatting"
  - "occ3d-large-scale-3d-occupancy-prediction-benchmark"
  - "BEVFormer"
---

## 💡 한 줄 요약
3D occupancy 예측을 이전 프레임의 3D Gaussian 표현과 현재 센서 입력에 조건화된 4D occupancy 예측 문제로 재정의하여, 추가 연산 비용 없이 단일 프레임 대비 mIoU 2% 이상을 향상시키는 World Model 기반 스트리밍 프레임워크를 제안한다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Sicheng Zuo, Wenzhao Zheng, Yuanhui Huang, Jie Zhou, Jiwen Lu (Tsinghua University)
- **발행년도**: 2024 (arXiv 2412.10373)
- **주요 기여점**:
  1. 3D occupancy 예측을 이전 Gaussian 상태 $\mathbf{z}^{T-1}$과 현재 관측 $\mathbf{x}^T$에 조건화된 World Model $\mathbf{z}^T = \mathbf{w}(\mathbf{z}^{T-1}, \mathbf{x}^T)$로 재정의
  2. 드라이빙 씬의 진화를 자차 이동 정합, 동적 객체 이동, 신규 영역 완성 세 요소로 분해하여 각각을 명시적·물리적으로 타당하게 모델링
  3. Motion 모드와 Perception 모드를 공유 아키텍처로 통합한 Evolution Layer 및 Unified Refinement Block 설계
  4. 시퀀스 길이를 점진적으로 늘리고 확률적으로 이전 프레임을 폐기하는 스트리밍 학습 전략으로 긴 시퀀스 적응력 확보
  5. nuScenes에서 단일 프레임 GaussianFormer-B 대비 mIoU +2.4%, 기존 시간 융합 방법 대비 낮은 레이턴시·메모리로 우월한 성능 달성

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: 3D semantic occupancy prediction은 MonoScene, BEVFormer, TPVFormer, SurroundOcc 등 단일 프레임 기반 방법에서, 시간 정보를 활용하는 시간적 퍼셉션 방법(BEVFormer의 시공간 어텐션, StreamPETR의 객체 쿼리 전파 등)으로 발전해왔고, GaussianFormer 같은 3D Gaussian 기반 occupancy 표현과 GAIA-1 같은 World Model 계보가 이 논문에서 교차합니다.
- **기존 한계점**:
  1. BEVFormer 등 기존 시간적 퍼셉션 방법들은 각 프레임의 BEV/볼륨 피처를 독립적으로 추출한 뒤 정합·융합하는 방식을 사용하여, 인접 프레임의 장면 표현이 자차 이동과 동적 객체 움직임으로부터 직접 진화한다는 자연스러운 연속성(continuity)과 단순성(simplicity)을 무시함
  2. 시간적 융합을 위해 추가 인코딩·정합·융합 모듈이 필요하여 단일 프레임 대비 레이턴시·메모리가 크게 증가함
  3. StreamPETR 같은 객체 쿼리 기반 방법은 동적 객체의 움직임을 암묵적으로만 표현하여 밀집 occupancy 예측에 부적합함
- **이 논문의 접근 방식**: 3D Gaussian으로 장면을 명시적으로 표현하고, 장면 진화를 세 요소(자차 정합·동적 객체 이동·신규 영역 완성)로 분해하여 World Model이 이를 학습합니다. 단일 프레임 입력만으로 스트리밍 방식의 4D occupancy 예측을 수행합니다.

## 📑 목차
- Section 1 & 2: Introduction & Related Work
- Section 3.1: World Models for Perception
- Section 3.2: Explicit Scene Evolution Modeling
- Section 3.3: 3D Gaussian World Model
- Section 4: Experiments

## 🛠️ Section 1 & 2: Introduction & Related Work

**요약**
3D semantic occupancy prediction은 장면 내 모든 복셀의 점유 여부와 시맨틱 레이블을 예측하는 과제로, 자율주행의 안전한 경로 계획에 필수적입니다. 기존의 시간적 퍼셉션 방법들은 과거 프레임의 표현을 현재로 정합·융합하는 방식을 쓰지만, 이는 드라이빙 씬이 연속적으로 진화한다는 강한 사전 지식(prior)을 활용하지 못합니다.

저자들은 World Model 기반 패러다임을 도입하여 장면 표현이 어떻게 진화하는지를 명시적으로 학습합니다. 3D Gaussian을 장면 표현으로 채택함으로써, BEV/Voxel 같은 암묵적 표현과 달리 객체의 위치·움직임을 연속적이고 명시적으로 모델링할 수 있습니다.

**핵심 개념**
- **World Model**: 미래 상태를 예측하는 생성 모델. 자율주행에서는 "세상이 어떻게 작동하는지"를 학습하여 센서 입력 없이 다음 상태를 예측하는 데 활용.
- **Streaming Prediction**: 매 프레임마다 이전 프레임의 예측 결과를 조건으로 현재 상태를 갱신하는 온라인 방식. 고정된 시간 윈도우를 재처리하지 않아 효율적.
- **4D Occupancy Forecasting**: 3D 공간 + 시간 축을 포함한 점유 예측. 현재 프레임의 센서 입력에 조건화되어 시공간적 장면을 예측하는 문제로 재정의.

## 🛠️ Section 3.1: World Models for Perception

**요약**
기존 시간적 퍼셉션 파이프라인은 각 프레임을 독립적으로 인코딩한 뒤 ego 궤적으로 정합하고 여러 프레임을 융합하는 3단계 구조입니다. GaussianWorld는 이를 단일 recurrence 식으로 대체합니다.

**핵심 개념**
- **기존 파이프라인의 한계**: 각 프레임을 따로 처리한 뒤 뭉치는 방식이라 인접 프레임 간 연속성 사전 지식을 무시함
- **World Model 방식**: 이전 Gaussian이 어떻게 이동·변형되는지를 학습하여 현재 장면을 예측. 추가 인코딩 없이 단 하나의 과거 프레임만 사용

**수식 예제**

$$\mathbf{z}^n = P_{er}(\mathbf{x}^n),\quad \mathbf{a}^n = T_{trans}(\mathbf{z}^n, \mathbf{p}^n),\quad \mathbf{y}^T = F_{use}(\mathbf{a}^T, \ldots, \mathbf{a}^{T-t})$$

**수식 설명**
- **$\mathbf{z}^n$**: n번째 프레임의 장면 표현 (BEV 피처 또는 볼륨 피처)
- **$P_{er}$**: 각 프레임을 독립적으로 인코딩하는 퍼셉션 모듈
- **$\mathbf{a}^n$**: ego 궤적 $\mathbf{p}^n$을 이용해 현재 프레임 좌표계로 정합된 표현
- **$T_{trans}$**: ego 자세 기반 좌표 변환 모듈
- **$F_{use}$**: 과거 $t$개 프레임의 정합된 표현을 융합하는 모듈
- **직관**: 각 프레임을 따로 처리한 뒤 뭉치는 방식 → 인접 프레임 간 연속성 사전 지식을 무시

**수식 예제**

$$\mathbf{z}^T = \mathbf{w}(\mathbf{z}^{T-1}, \mathbf{x}^T)$$

$$\mathbf{y}^T = \mathbf{h}(\mathbf{w}(\mathbf{z}^{T-1}, \mathbf{x}^T))$$

**수식 설명**
- **$\mathbf{z}^{T-1}$**: 이전 프레임에서 예측된 3D Gaussian 집합
- **$\mathbf{x}^T$**: 현재 프레임의 RGB 카메라 입력
- **$\mathbf{w}$**: World Model — 이전 장면 표현과 현재 관측을 받아 현재 장면 표현을 예측
- **$\mathbf{h}$**: 퍼셉션 헤드 — 3D Gaussian에서 occupancy로 변환
- **직관**: 이전 Gaussian이 어떻게 이동·변형되는지를 학습하여 현재 장면을 예측. 추가 인코딩 없이 단 하나의 과거 프레임만 사용

## 🛠️ Section 3.2: Explicit Scene Evolution Modeling

**요약**
드라이빙 씬의 진화는 대부분 단순하고 연속적입니다. GaussianWorld는 이를 세 가지 독립적 요소로 분해하여 각각을 명시적으로 모델링합니다.

1. **자차 이동 정합 (Ego Motion Alignment)**: 이전 프레임의 3D Gaussian 전체에 ego 궤적 기반 전역 아핀 변환을 적용. 자동차가 앞으로 이동하면 정적 건물들이 뒤로 밀리는 것처럼, 모든 Gaussian을 새 자차 좌표계로 평행이동·회전
2. **동적 객체 이동 (Local Movements of Dynamic Objects)**: 정합된 Gaussian을 동적($\{g_D\}$)과 정적($\{g_S\}$) 집합으로 분리하고, 동적 Gaussian의 시맨틱 확률을 소프트 가중치로 사용하여 위치만 업데이트. 주변 차량·보행자의 실제 이동을 RGB 관측으로부터 학습하여 반영
3. **신규 영역 완성 (Completion of Newly-Observed Areas)**: ego가 전진하면 일부 Gaussian은 인식 범위 밖으로 나가고 새 영역이 보이게 되는데, 새 영역에 랜덤 초기화 Gaussian을 배치하고 퍼셉션 레이어로 모든 속성 예측. "처음 보는 교차로"를 현재 카메라 이미지로부터 채워 넣는 과정

**핵심 개념**
- **3D Gaussian 속성**: 위치, 크기, 회전, 시맨틱 확률, 시간적 피처의 5가지 속성으로 구성
- **$I(\cdot)$**: 동적 Gaussian 여부를 나타내는 지시 함수

**수식 예제**

$$\mathbf{g} = \{\mathbf{p}, \mathbf{s}, \mathbf{r}, \mathbf{c}, \mathbf{f}\}$$

**수식 설명**
- **$\mathbf{p}$**: 3D 위치 (position)
- **$\mathbf{s}$**: 크기 (scale)
- **$\mathbf{r}$**: 회전 (rotation)
- **$\mathbf{c}$**: 시맨틱 확률 (semantic probability) — 각 클래스에 속할 확률
- **$\mathbf{f}$**: 시간적 피처 (temporal feature) — Gaussian의 역사적 정보를 담는 추가 속성

**수식 예제**

$$\mathbf{g}_A^T = A_{lign}(\mathbf{g}^{T-1}, \mathbf{M}_{ego}) = R_{ef}(\mathbf{g}^{T-1}; \mathbf{M}_{ego} \cdot A_{ttr}(\mathbf{g}^{T-1}; \mathbf{p}); \mathbf{p})$$

**수식 설명**
- **$\mathbf{M}_{ego}$**: 이전 프레임에서 현재 프레임으로의 ego 이동 변환 행렬
- **직관**: 자동차가 앞으로 이동하면 정적 건물들이 뒤로 밀리는 것처럼, 모든 Gaussian을 새 자차 좌표계로 평행이동·회전

**수식 예제**

$$\mathbf{g}_M^T = M_{ove}(\mathbf{g}_A^T, \mathbf{x}_T) = R_{ef}(\mathbf{g}_A^T; E_{nc}(\mathbf{g}_A^T, \mathbf{x}_T) \cdot I(\mathbf{g}_A^T \in \{\mathbf{g}_D\}); \mathbf{p})$$

**수식 설명**
- 정합된 Gaussian을 동적($\{g_D\}$)과 정적($\{g_S\}$) 집합으로 분리
- 동적 Gaussian의 시맨틱 확률을 소프트 가중치로 사용하여 위치만 업데이트
- **직관**: 주변 차량·보행자의 실제 이동을 RGB 관측으로부터 학습하여 반영

**수식 예제**

$$\mathbf{g}_C^T = P_{er}(\mathbf{g}_I^T, \mathbf{x}_T) = R_{ef}(\mathbf{g}_I^T; E_{nc}(\mathbf{g}_I^T, \mathbf{x}_T); \{\mathbf{p}, \mathbf{s}, \mathbf{r}, \mathbf{c}, \mathbf{f}\})$$

**수식 설명**
- ego가 전진하면 일부 Gaussian은 인식 범위 밖으로 나가고, 새 영역이 보이게 됨
- 새 영역에 랜덤 초기화 Gaussian $\mathbf{g}_I^T$를 배치하고 퍼셉션 레이어로 모든 속성 예측
- **직관**: "처음 보는 교차로"를 현재 카메라 이미지로부터 채워 넣는 과정

## 🛠️ Section 3.3: 3D Gaussian World Model

**요약**
세 요소를 모델링하는 통합 프레임워크입니다. Motion Layer와 Perception Layer가 같은 아키텍처를 공유하여 계산 효율을 높입니다.

새 Gaussian이면 Perception 모드(모든 속성 예측), 역사 Gaussian이면 Motion 모드(위치만 업데이트)로 동작하는 통합 Evolution Layer를 $n_e$개 스택하여 반복 정제하고, 이후 $n_r$개의 Unified Refinement Block으로 3D Gaussian 표현과 실제 세계 간 정렬 오차를 보정합니다.

**핵심 모듈 구성**

| 모듈 | 역할 |
|------|------|
| Self-Encoding | 3D Gaussian을 복셀화 후 3D sparse convolution으로 Gaussian 간 상호작용 학습 |
| Cross-Attention | Deformable Attention으로 3D Gaussian과 멀티스케일 이미지 피처 간 상호작용 |
| Unified Refinement Block | Motion / Perception 모드를 통합한 Gaussian 속성 예측 블록 |
| GS-to-Occ | 정제된 3D Gaussian에서 occupancy 복셀 그리드로 변환 |

**스트리밍 학습 전략**: 초기에 짧은 시퀀스로 시작하여 점진적으로 길이를 늘리고(시퀀스 길이 [5, 10, 20, 30, 38] 단계적 증가), 일정 확률 $p$로 이전 프레임의 3D Gaussian 표현을 랜덤 폐기하여 긴 시퀀스에 적응합니다. 처음에는 "짧은 기억"으로 학습하다가 점점 "긴 기억"으로 확장하는 확률적 모델링으로 훈련 안정성을 향상시킵니다.

**핵심 개념**
- **Evolution Layer**: $\mathbf{g}_{l+1}^T = E_{vol}(\mathbf{g}_l^T, \mathbf{x}_T)$ — 새 Gaussian이면 Perception 모드, 역사 Gaussian이면 Motion 모드로 분기 처리
- **Unified Refinement Block**: $\mathbf{g}_{n+1}^T = R_{efine}(\mathbf{g}_n^T, \mathbf{x}_T)$ — Evolution Layer와 달리 모든 Gaussian의 모든 속성을 업데이트

## 📊 주요 실험 및 결과 (Experiments & Results)
- **사용 데이터셋 / 벤치마크**: nuScenes validation

**주요 결과 (nuScenes validation)**

| Method | mIoU | IoU |
|--------|------|-----|
| MonoScene | 7.31 | 23.96 |
| BEVFormer | 26.88 | 30.50 |
| TPVFormer | 11.66 | 11.51 |
| SurroundOcc | 20.30 | 31.49 |
| OccFormer | 19.03 | 31.39 |
| GaussianFormer-B (단일 프레임) | 19.10 | 29.83 |
| GaussianFormer-T (시간 융합) | 20.42 | 31.34 |
| **GaussianWorld (ours)** | **22.13** | **33.40** |

GaussianWorld는 단일 프레임 GaussianFormer-B 대비 mIoU +2.4%, IoU +2.7% 향상, 시간 융합 GaussianFormer-T 대비 mIoU +1.7%, IoU +2.0% 향상.

**시간적 모델링 방법 비교 (효율성)**

| 방법 | 역사 프레임 수 | 레이턴시 | 메모리 | mIoU |
|------|-------------|---------|-------|------|
| Single-Frame | 0 | 225 ms | 6958 M | 19.73 |
| 3D Gaussian Fusion | 3 | 379 ms | 9993 M | 20.24 |
| Perspective View Fusion | 3 | 382 ms | 10019 M | 20.42 |
| **GaussianWorld** | **1** | **228 ms** | **7030 M** | **21.87** |

GaussianWorld는 역사 프레임 1개만 사용하면서 단일 프레임과 거의 같은 레이턴시·메모리로 시간 융합 방법들을 모두 상회.

**Ablation: 장면 진화 세 요소의 기여**

| Ego 정합 | 동적 이동 | 신규 완성 | mIoU | IoU |
|---------|---------|---------|------|-----|
| ✗ | ✓ | ✓ | 18.47 | 28.88 |
| ✓ | ✗ | ✓ | 21.17 | 32.49 |
| ✓ | ✓ | ✗ | 학습 붕괴 | 학습 붕괴 |
| ✓ | ✓ | ✓ | **21.50** | **32.72** |

신규 영역 완성이 없으면 학습 자체가 붕괴합니다(ego가 계속 전진하면 결국 모든 Gaussian이 인식 범위 밖으로 나가 장면 표현이 소실됨). ego 정합이 가장 큰 기여(3.0% mIoU)를 하고, 동적 이동 모델링도 유의미한 향상을 가져옵니다.

스트리밍 프레임 수가 늘어날수록 성능이 향상되나, 약 20 프레임 이후 소폭 하락합니다. 이는 기존 3D occupancy GT가 멀티 프레임 LiDAR 누적으로 생성되어 가장자리가 희소하므로 긴 시퀀스에서 주석 품질 한계에 봉착하기 때문입니다.

## 💡 결론 및 시사점 (Conclusion & Insights)
GaussianWorld는 3D Gaussian의 명시적·연속적 장면 표현 특성을 활용하여 "장면이 어떻게 진화하는가"를 World Model로 학습합니다. 3D occupancy를 이전 Gaussian + 현재 관측에 조건화된 예측 문제로 재구성하여 시간적 연속성을 자연스럽게 활용하고, 단일 과거 프레임만 사용하면서도 기존 시간 융합 방법 대비 낮은 레이턴시·메모리로 우월한 성능을 냅니다.

- 이 논문은 GaussianFormer(3DGS 기반 occupancy 표현)의 직접적 확장이며, 3DGS → occupancy prediction 계보와 World Model 계보가 만나는 교차점입니다.
- 자율주행 합성 데이터 생성 관점에서는, 3D Gaussian으로 장면을 명시적으로 진화시키는 이 패러다임이 향후 클로즈드루프 센서 시뮬레이터의 장면 상태 관리에 직접 응용될 가능성이 있습니다.
- 추가 연산 없이 시간 정보를 활용할 수 있다는 점은 온보드 실시간 시스템에 실질적인 이점을 제공합니다.
- **한계점 및 아쉬운 점**: 동적 요소와 정적 요소의 분리가 완전하지 않아 정적 장면의 크로스 프레임 일관성을 완벽히 보장하지 못합니다. 또한 신규 영역 완성 요소가 없으면 학습이 붕괴할 정도로 세 요소 간 의존성이 강해, 아키텍처 설계의 견고성 확보를 위해 세밀한 균형 조정이 필요합니다. GT 자체가 멀티 프레임 LiDAR 누적으로 생성되어 희소한 한계로 인해, 긴 시퀀스에서는 성능이 오히려 하락하는 문제도 완전히 해결되지 않았습니다.

---

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [MonoScene](/posts/papers/monoscene-monocular-3d-semantic-scene-completion/), [TPVFormer](/posts/papers/tpvformer-tri-perspective-view-3d-semantic-occupancy/), [SurroundOcc](/posts/papers/SurroundOcc/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
