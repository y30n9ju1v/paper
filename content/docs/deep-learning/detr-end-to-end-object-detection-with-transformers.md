---
title: "DETR: End-to-End Object Detection with Transformers"
date: 2026-04-20T14:00:00+09:00
draft: false
categories: ["Papers", "Computer Vision", "Deep Learning"]
tags: ["Object Detection", "Transformer", "Bipartite Matching", "DETR", "Facebook AI"]
year: 2020
references:
  - "resnet-deep-residual-learning-for-image-recognition"
  - "attention-is-all-you-need"
---

## 💡 한 줄 요약
객체 탐지를 직접 집합 예측 문제로 정의하고 이분 매칭(bipartite matching) 손실과 Transformer 인코더-디코더를 결합하여, NMS·anchor 같은 수작업 컴포넌트 없이 end-to-end로 객체를 탐지하는 DETR을 제안했다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko (Facebook AI)
- **발행년도**: 2020 (ECCV 2020)
- **주요 기여점**:
  1. 헝가리안 알고리즘 기반 이분 매칭으로 예측-GT를 1:1 매칭하는 집합 손실을 정의해 NMS·anchor를 완전히 제거
  2. CNN 백본 + Transformer 인코더-디코더 + N개의 학습 가능한 Object Query로 구성된 단순한 end-to-end 아키텍처 제안
  3. COCO에서 Faster R-CNN과 대등하거나 더 나은 AP를 달성하면서(특히 대형 객체에서 +7.8 AP), 마스크 헤드만 추가하면 Panoptic Segmentation까지 확장 가능함을 실증
  4. PyTorch 50줄 이내로 구현 가능할 만큼 구조가 단순함을 보여 이후 쿼리 기반 탐지 연구의 원형을 제시

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: 관련 연구는 세 갈래로 발전해왔다: (1) 집합 예측을 위한 이분 매칭 손실 — 헝가리안 알고리즘 기반 매칭 손실을 쓴 초기 탐지기들이 있었지만 CNN 기반이라 객체 간 관계 모델링이 약하고 NMS가 필요했다, (2) Transformer와 병렬 디코딩 — RNN 기반 자기회귀 방식에서 벗어나 병렬 생성이 가능한 구조로 발전, (3) 현대 객체 탐지 방법론 — Faster R-CNN 등 anchor·proposal 기반 2-stage/1-stage 탐지기가 주류였다.
- **기존 한계점**:
  1. 수작업 설계 컴포넌트 의존 — Faster R-CNN 등 기존 탐지기는 anchor 생성, NMS(Non-Maximum Suppression), proposal 매칭 휴리스틱 등 도메인 지식을 하드코딩한 컴포넌트에 의존한다.
  2. 중복 예측 문제 — 고정 anchor나 proposal 기반 방법은 같은 객체에 여러 box를 예측하고, NMS로 후처리해야 한다.
  3. 자기회귀 디코더의 느린 추론 — 이전 end-to-end 시도(RNN 기반)는 순차적으로 박스를 예측해 추론이 느리고 병렬화가 어렵다.
- **이 논문의 접근 방식**: 이분 매칭(bipartite matching)으로 예측-GT를 1:1 매칭하는 집합 손실과 Transformer 인코더-디코더로 모든 객체를 병렬로 예측하여 NMS·anchor를 완전히 제거한다.

## 📑 목차
- Chapter 1: Introduction
- Chapter 2: Related Work (집합 예측, Transformer, 객체 탐지)
- Chapter 3: DETR 모델 (집합 예측 손실 + 아키텍처)
- Chapter 4: Experiments (COCO 비교, Ablation, Panoptic Segmentation)
- Chapter 5: Conclusion

## 🛠️ Chapter 1: Introduction

**요약**

현대 객체 탐지기들은 실제로 원하는 것("이미지에서 객체 위치와 클래스를 예측")을 대신하는 surrogate task(anchor 분류, proposal 회귀 등)를 풀도록 설계되어 있습니다. 이 과정에서 NMS, anchor 설계 같은 수작업 컴포넌트가 필수가 됩니다.

DETR은 이를 근본적으로 다르게 접근합니다. 객체 탐지를 **직접 집합 예측(direct set prediction)** 문제로 정의하고, 이분 매칭으로 중복 없는 1:1 예측을 보장하며, Transformer의 self-attention으로 전체 이미지 맥락에서 객체 간 관계를 모델링합니다.

**핵심 개념**

- **Direct Set Prediction**: 고정 크기 N개의 예측을 한 번에 출력 — NMS 없이 중복 제거
- **Bipartite Matching**: 헝가리안 알고리즘으로 예측 집합과 GT 집합을 최적 1:1 매칭 → 순열 불변 손실
- **Object Query**: 디코더에 입력되는 N개의 학습 가능한 위치 임베딩 — 각 쿼리가 이미지의 서로 다른 영역/크기를 담당

## 🛠️ Chapter 2: Related Work

**요약**

관련 연구는 세 갈래입니다: (1) 집합 예측을 위한 이분 매칭 손실, (2) Transformer와 병렬 디코딩, (3) 현대 객체 탐지 방법론.

핵심 선행 연구로 헝가리안 알고리즘 기반 매칭 손실을 쓴 초기 탐지기들이 있지만, CNN 기반이라 객체 간 관계 모델링이 약하고 NMS가 필요했습니다. DETR은 Transformer의 전역 self-attention으로 이 한계를 극복합니다.

**핵심 개념**

- **Hungarian Algorithm (헝가리안 알고리즘)**: 비용 행렬에서 최소 비용 1:1 매칭을 찾는 고전 알고리즘 — DETR 매칭의 핵심
- **Permutation-invariant Loss**: 어떤 순서로 예측해도 같은 손실값 → 병렬 예측 가능
- **Non-autoregressive Decoding**: RNN처럼 순차적으로 출력하지 않고, 모든 출력을 동시에(병렬로) 생성

## 🛠️ Chapter 3: DETR 모델

### 3.1 Object Detection Set Prediction Loss (집합 예측 손실)

**요약**

DETR은 고정 크기 N개의 예측을 출력합니다(N은 이미지 내 객체 수보다 충분히 크게 설정). 학습 시 예측 집합과 GT 집합을 헝가리안 알고리즘으로 최적 매칭한 뒤, 매칭된 쌍에 대해서만 손실을 계산합니다.

**수식 예제 — 최적 매칭 탐색**

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i}^{N} \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})$$

**수식 설명**

- **$\mathfrak{S}_N$**: N개 원소의 모든 순열 집합
- **$y_i$**: i번째 GT 객체 (클래스 레이블 $c_i$와 박스 $b_i$로 구성)
- **$\hat{y}_{\sigma(i)}$**: 순열 $\sigma$에서 i번째 GT에 매칭된 예측
- **$\mathcal{L}_{\text{match}}$**: 매칭 비용 — 클래스 확률 + 박스 위치 유사도
- 이 식의 핵심: N개 예측을 N개 GT에 **1:1로** 할당하는 최적 순열 $\hat{\sigma}$를 찾는다. 나머지 슬롯은 "no object(∅)"에 할당됨

**수식 예제 — 헝가리안 손실 (Hungarian Loss)**

$$\mathcal{L}_{\text{Hungarian}}(y, \hat{y}) = \sum_{i=1}^{N} \left[ -\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)}) \right]$$

**수식 설명**

매칭이 완료된 후 실제 학습에 사용되는 손실입니다:
- **$-\log \hat{p}_{\hat{\sigma}(i)}(c_i)$**: 분류 손실 — 예측 클래스 확률의 음의 로그 우도 (cross-entropy)
- **$\mathbb{1}_{\{c_i \neq \varnothing\}}$**: GT가 실제 객체일 때만 박스 손실 계산 (빈 슬롯 제외)
- **$\mathcal{L}_{\text{box}}$**: 박스 손실 — $\ell_1$ 손실 + GIoU 손실의 선형 결합
- **왜 GIoU도 쓰나?**: $\ell_1$ 손실은 크고 작은 박스에서 스케일이 달라 불균형 발생 → GIoU(스케일 불변)를 함께 사용

**수식 예제 — 박스 손실**

$$\mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)}) = \lambda_{\text{iou}} \mathcal{L}_{\text{iou}}(b_i, \hat{b}_{\hat{\sigma}(i)}) + \lambda_{\text{L1}} \| b_i - \hat{b}_{\hat{\sigma}(i)} \|_1$$

**수식 설명**

- **$\lambda_{\text{iou}}, \lambda_{\text{L1}}$**: GIoU 손실과 L1 손실의 가중치 하이퍼파라미터
- **$\mathcal{L}_{\text{iou}}$**: GIoU(Generalized IoU) — 박스가 겹치지 않아도 기울기 제공, 스케일 불변
- **$\| b_i - \hat{b} \|_1$**: 박스 좌표의 L1 거리 (중심점 x, y, 너비, 높이 — 이미지 크기로 정규화된 값)

### 3.2 DETR Architecture (아키텍처)

**요약**

DETR은 세 가지 주요 컴포넌트로 구성됩니다: CNN 백본, Transformer 인코더-디코더, FFN 예측 헤드.

```
이미지 → CNN Backbone → feature map (C×H×W)
       → 1×1 conv → (d×H×W) → flatten → (HW×d) sequence
       → Positional Encoding 추가
       → Transformer Encoder (전역 self-attention)
       → Transformer Decoder (N개 object query)
       → FFN Head → N개 (클래스, 박스) 예측
```

**핵심 개념**

- **CNN Backbone**: ResNet으로 이미지에서 피처 맵 추출. 기본값 $C=2048$, $H=W=H_0/32$
- **1×1 Convolution**: 채널 차원을 $C$에서 $d$로 축소 (보통 $d=256$)
- **Positional Encoding (위치 인코딩)**: Transformer는 순서를 모르므로, 각 피처 위치에 고정 사인파 인코딩을 추가 — 인코더의 모든 attention 레이어에 반복 추가됨
- **Object Query**: 디코더에 입력되는 N개의 학습 가능한 임베딩 (=출력 위치 인코딩) — 각 쿼리가 서로 다른 객체를 탐지하도록 학습됨
- **FFN (Feed-Forward Network)**: 3층 MLP + ReLU → 박스 4좌표 예측 (정규화된 중심점 x, y, 너비, 높이)
- **Auxiliary Decoding Loss**: 디코더의 각 레이어마다 FFN 헤드와 헝가리안 손실을 추가 → 학습 안정성 향상, +8.2 AP

**Transformer Encoder의 역할**

인코더의 self-attention은 이미지 전체에서 전역 관계를 학습합니다. 실험에서 인코더가 개별 인스턴스를 이미 분리하는 것이 관찰됐습니다 (Figure 3): 특정 점에 대한 self-attention이 같은 객체 내부는 높게, 다른 객체는 낮게 활성화됩니다.

**Transformer Decoder의 역할**

디코더는 N개의 object query를 인코더 출력과 cross-attention으로 상호작용시킵니다. 디코더의 attention은 객체 경계(머리, 다리 등 extremities)에 집중하는 경향이 있습니다 — 인코더가 이미 전역 분리를 수행했으므로 디코더는 세부 위치에 집중.

## 📊 주요 실험 및 결과 (Experiments & Results)

- **사용 데이터셋 / 벤치마크**: COCO 2017 (118k 학습, 5k 검증). 평가 지표는 박스 AP(다양한 IoU 임계값의 평균).
- **주요 성과**: Faster R-CNN과의 비교 (ResNet-50 기준)

| 모델 | AP | AP$_{50}$ | AP$_S$ | AP$_M$ | AP$_L$ | FPS |
|------|-----|-----------|--------|--------|--------|-----|
| Faster RCNN-FPN+ | 42.0 | 62.1 | 26.6 | 45.4 | 53.4 | 26 |
| **DETR** | 42.0 | 62.4 | 20.5 | 45.8 | **61.1** | 28 |
| **DETR-DC5** | 43.3 | 63.1 | 22.5 | 47.3 | **61.1** | 12 |

  핵심 관찰:
  - 대형 객체(AP$_L$): DETR이 Faster R-CNN보다 **+7.8 AP** — Transformer의 전역 reasoning 덕분
  - 소형 객체(AP$_S$): DETR이 -5.5 AP 낮음 — 소형 객체는 지역 특징이 중요한데, 전역 attention이 불리
  - 추론 속도: 비슷한 FPS (28 vs 26 FPS, ResNet-50 기준)
  - 구현 단순성: PyTorch로 50줄 이내 구현 가능

  Ablation 주요 결과:

| 컴포넌트 제거 | AP 변화 |
|-------------|---------|
| Encoder 없음 (0층) | -3.9 AP (특히 대형 객체 -6.0) |
| Decoder 1층만 | -8.2 AP (NMS 추가하면 회복) |
| FFN 제거 | -2.3 AP |
| Positional Encoding 없음 | -7.8 AP |
| GIoU Loss 없음 | AP$_M$, AP$_L$ 하락 |

  인코더의 중요성: 인코더 레이어 수를 늘릴수록 AP가 단조 증가 (0층: 36.7 → 12층: 41.6). 전역 self-attention이 객체 분리에 핵심.

  Panoptic Segmentation으로 확장: DETR 디코더 출력에 마스크 헤드(Multi-head attention → FPN → pixel-wise argmax)를 추가하면 별도의 stuff/thing 분기 없이 통합 panoptic segmentation이 가능합니다. COCO val에서 DETR-R101이 PQ 45.1로 UPSNet(43.0)과 PanopticFPN(44.1)을 능가합니다.

## 💡 결론 및 시사점 (Conclusion & Insights)

DETR은 객체 탐지 패러다임에서 두 가지 핵심 기여를 했습니다:

1. **패러다임 전환**: anchor·NMS·proposal 없는 순수 end-to-end 탐지기 — 도메인 지식 없이 표준 CNN + Transformer만으로 구현 가능
2. **쿼리 기반 탐지의 원형**: Object query 개념이 이후 DETR3D, BEVFormer, MapTR 등 자율주행 인식 연구 전반에 확산

- **MapTR와의 연결**: MapTR의 계층적 쿼리(인스턴스 쿼리 + 포인트 쿼리)는 DETR의 object query를 HD 맵 요소에 맞게 확장한 것이며, MapTR의 계층적 이분 매칭(인스턴스 레벨 → 포인트 레벨)은 DETR의 헝가리안 매칭을 2단계로 발전시킨 것이다. DETR이 없었다면 MapTR, DETR3D, UniAD의 query-based 설계 모두 나오기 어려웠을 것이다.
- **한계점 및 아쉬운 점**:
  - 소형 객체 성능이 Faster R-CNN 대비 열세 — Deformable DETR(2021)이 deformable attention으로 해결
  - 매우 긴 학습 스케줄 필요 (500 epoch) — Conditional DETR, DAB-DETR 등으로 수렴 속도 개선
  - 인코더의 높은 계산 비용 (HW 크기 self-attention) — Deformable DETR, Sparse DETR로 효율화
  - 헝가리안 매칭이 학습 초반 불안정할 수 있다는 점은 논문에서 충분히 다루지 않은 아쉬운 부분

---

*관련 논문: [Attention Is All You Need](/posts/papers/attention-is-all-you-need/), [ResNet](/posts/papers/resnet-deep-residual-learning-for-image-recognition/), [DETR3D](/posts/papers/detr3d-3d-object-detection-multi-view-images/), [VectorMapNet](/posts/papers/vectormapnet-end-to-end-vectorized-hd-map-learning/), [MapTR](/posts/papers/maptr-structured-modeling-online-vectorized-hd-map-construction/)*
