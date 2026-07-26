---
title: "MonoScene: Monocular 3D Semantic Scene Completion"
date: 2026-04-19T21:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "3D Scene Understanding"]
tags: ["Occupancy Prediction", "Semantic Scene Completion", "NeRF", "BEV", "3D Understanding"]
year: 2022
references:
  - "resnet-deep-residual-learning-for-image-recognition"
---

## 💡 한 줄 요약
단일 RGB 이미지만으로 3D 공간의 기하와 의미를 동시에 추론하는 최초의 실내·실외 통합 3D Semantic Scene Completion(SSC) 프레임워크로, FLoSP(2D-3D 피처 투영)와 3D CRP(맥락 관계 사전), 새로운 손실 함수를 통해 깊이 센서 없이도 밀집한 3D 시맨틱 장면 완성을 달성한다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Anh-Quan Cao, Raoul de Charette (Inria)
- **발행년도**: 2022 (arXiv 2021, CVPR 2022, arXiv:2112.00726)
- **주요 기여점**:
  1. FLoSP(Features Line of Sight Projection)로 카메라 시선을 따라 멀티스케일 2D 피처를 3D 복셀로 투영하는 2D-3D 브릿지 설계
  2. 3D CRP(3D Context Relation Prior)로 복셀 간 free/occupied × 의미 유사/다름의 4가지 공간-의미 관계를 3D UNet 병목에서 학습
  3. 클래스별 Precision/Recall/Specificity를 직접 최적화하는 Scene-Class Affinity Loss로 cross-entropy의 맥락 무시 문제 보완
  4. 이미지 패치 단위 frustum의 클래스 분포 KL 발산을 최소화하는 Frustum Proportion Loss로 occlusion으로 인한 편향 예측 억제
  5. NYUv2(실내), SemanticKITTI(실외) 양쪽에서 단일 RGB만으로 depth/point cloud 기반 일부 방법보다 높은 mIoU 달성

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: 단일 객체의 3D 재구성에 집중하던 초기 딥러닝 연구는 점차 전체 장면의 holistic 이해로 발전했습니다. 3D Semantic Scene Completion(SSC)은 SSCNet이 처음 정의했으며, 이후 depth, TSDF, point cloud, occupancy grid 등 다양한 입력 기반 방법(LMSCNet, AICNet, 3DSketch 등)이 실내(NYUv2) 또는 실외(SemanticKITTI) 전용으로 분리 설계되어 발전했습니다. 한편 2D 시맨틱 분할에서는 ASPP, self-attention 같은 맥락 학습 기법이 효과적으로 사용되어 왔습니다.
- **기존 한계점**:
  1. 기존 SSC 방법은 모두 depth map, point cloud, occupancy grid 등 2.5D/3D 입력을 필요로 하여, 비싼 센서이거나 별도의 깊이 추정 단계가 필요했음
  2. 실내용(NYUv2) 또는 실외용(SemanticKITTI) 전용 모델로 설계되어 범용 적용이 불가능했음
  3. 기존 SSC 학습은 cross-entropy loss로 각 복셀을 독립적으로 최적화하여 그룹 수준의 맥락 정보를 활용하지 못했음
- **이 논문의 접근 방식**: 단일 RGB 이미지 → FLoSP(2D-3D 피처 투영) + 3D CRP(맥락 관계 사전) + 새로운 SSC 손실 함수로 2D 정보만으로 밀집한 3D 시맨틱 장면을 완성합니다.

## 📑 목차
- Section 1: Introduction
- Section 2: Related Works
- Section 3.1: Features Line of Sight Projection (FLoSP)
- Section 3.2: 3D Context Relation Prior (3D CRP)
- Section 3.3: Losses (Scene-Class Affinity Loss + Frustum Proportion Loss)
- Section 4: Experiments

## 🛠️ Section 1: Introduction

**요약**
인간은 단일 시점의 이미지에서도 3D 장면의 구조와 의미를 자연스럽게 이해합니다. 이를 컴퓨터가 수행하는 것이 3D Semantic Scene Completion(SSC) — 이미지로부터 전체 3D 복셀 공간의 점유 여부와 의미 클래스를 동시에 예측하는 태스크입니다.

기존 연구들은 모두 LiDAR, depth camera, 점령 격자(occupancy grid) 같은 3D 입력에 의존했습니다. MonoScene은 이 제약을 없애고 단일 RGB 이미지만으로 같은 태스크를 수행하는 첫 번째 방법입니다.

핵심 도전 과제는 2D → 3D 공간 복원의 어려움입니다. 카메라가 찍은 2D 이미지에서 깊이 정보가 소실되기 때문에 3D 복원이 근본적으로 ill-posed 문제입니다. MonoScene은 이를 광학(optics)에서 영감을 받은 시선(line of sight) 투영으로 해결합니다.

## 🛠️ Section 2: Related Works

**요약**
3D from a single image 분야에서는 초기 딥러닝 연구들이 단일 객체 재구성에 집중했으며, 점차 전체 장면의 holistic 이해로 발전했습니다. 3D Semantic Scene Completion(SSC)은 SSCNet이 처음 태스크를 정의했으며, 이후 depth, TSDF, point cloud, occupancy grid 입력 기반 방법들이 발전했지만 기존 방법들은 실내 또는 실외 전용으로 분리 설계되었습니다. Contextual awareness 측면에서는 2D 시맨틱 분할의 맥락 학습(ASPP, self-attention 등)을 MonoScene이 3D SSC에 새롭게 도입합니다.

## 🛠️ Section 3.1: Features Line of Sight Projection (FLoSP)

**요약**
카메라에서 3D 복셀 중심점으로 향하는 시선(ray) 위의 2D 피처들이 해당 복셀 이해에 관련이 있다는 것이 핵심 직관입니다. 광학의 원리를 활용하여 2D-3D 정보 브릿지를 구성합니다.

동작 방식은 멀티스케일 2D 디코더 피처맵 $F_{2D}^{1:s}$ (스케일 $s \in \{1, 2, 4, 8\}$)에서, 각 3D 복셀 중심점 $x^c$를 카메라 투영 $\rho(\cdot)$으로 2D에 매핑한 뒤, 해당 위치의 피처를 샘플링하여 합산합니다. 이미지 FOV 밖의 복셀은 피처 벡터를 0으로 설정하며, 이를 통해 FOV 외부 장면도 "암묵적 추론"이 가능해집니다.

**핵심 개념**
- **FLoSP**: 카메라 시선을 따라 2D 멀티스케일 피처를 3D 복셀로 투영하는 모듈. 2D-3D 정보 브릿지 역할
- **전체 파이프라인**: RGB 이미지 → 2D UNet(EfficientNetB7 백본) → FLoSP → 3D UNet → 3D CRP → ASPP+Softmax → 3D 시맨틱 복셀 예측

**수식 예제**

$$\mathbf{F}_{3D} = \sum_{s \in S} \Phi_{\rho(x^c)}\!\left(F_{2D}^{1:s}\right)$$

**수식 설명**
- **$\mathbf{F}_{3D}$**: 최종 3D 피처맵 (3D UNet의 입력)
- **$s$**: 2D 피처맵의 스케일 (1, 2, 4, 8배 다운샘플)
- **$\rho(x^c)$**: 3D 복셀 중심점 $x^c$를 카메라 내부 파라미터로 2D에 투영한 픽셀 좌표
- **$\Phi_a(b)$**: 피처맵 $b$에서 좌표 $a$의 피처를 bilinear 샘플링
- **핵심 아이디어**: 카메라 시선 위에 놓인 모든 3D 복셀이 같은 픽셀에 투영됨 → 깊이를 모르더라도 해당 시선의 2D 피처를 모든 깊이의 복셀에 공유

## 🛠️ Section 3.2: 3D Context Relation Prior (3D CRP)

**요약**
SSC는 맥락에 매우 의존적입니다. 예를 들어 "도로 옆은 인도일 가능성이 높다", "차 옆에는 사람이 있을 수 있다"는 공간 의미 관계(spatio-semantic relation)를 학습합니다. 3D UNet 병목(bottleneck)에 삽입되어, ASPP 컨볼루션으로 큰 수용야(receptive field)를 확보한 뒤, 복셀 간 n-way 관계 행렬 $\hat{A}^m$을 예측합니다.

4가지 관계 유형은 둘 다 free(빈 공간)이고 의미 유사($\mathbf{f_s}$), 둘 다 free이고 의미 다름($\mathbf{f_d}$), 둘 다 occupied(점유)이고 의미 유사($\mathbf{o_s}$), 둘 다 occupied이고 의미 다름($\mathbf{o_d}$)입니다. 메모리 효율을 위해 $s^3$ 이웃 복셀을 하나의 supervoxel로 묶어 관계 행렬 크기를 $N^2$에서 $\frac{N^2}{s^3}$으로 줄입니다.

**핵심 개념**
- **3D CRP**: 복셀 간 free/occupied × 의미유사/다름 4가지 관계를 학습하는 3D UNet 병목 레이어. 공간 맥락 인식
- **Supervoxel**: $s^3$개 인접 복셀 그룹. 관계 행렬 메모리 효율화를 위해 복셀 대신 supervoxel 단위로 관계 예측

**수식 예제**

$$\mathcal{L}_{rel} = -\sum_{m \in \mathcal{M}, i} \left[(1 - A_i^m)\log(1 - \hat{A}_i^m) + w_m A_i^m \log \hat{A}_i^m\right]$$

**수식 설명**
- **$A_i^m$**: i번째 쌍에 대한 관계 $m$의 GT (0 또는 1)
- **$\hat{A}_i^m$**: 예측된 관계 확률
- **$w_m = \frac{\sum_i(1 - A_i^m)}{\sum_i A_i^m}$**: free/occupied 불균형(약 9:1)을 보정하는 클래스 가중치
- **효과**: 관계 행렬이 supervision 없이도 self-attention처럼 작동 가능 (w/o $\mathcal{L}_{rel}$ 시 암묵적 어텐션으로 동작)

## 🛠️ Section 3.3: 손실 함수 (Losses)

**요약**
기존 cross-entropy는 각 복셀을 독립적으로 최적화하여 전역 장면 수준 성능을 직접 최적화하지 못합니다. MonoScene은 클래스별 Precision, Recall, Specificity를 직접 최적화하는 Scene-Class Affinity Loss($\mathcal{L}_{scal}$)를 사용합니다. 시맨틱 GT로 계산하는 $\mathcal{L}_{scal}^{sem}$과 기하 GT로 계산하는 $\mathcal{L}_{scal}^{geo}$ 두 가지 버전이 있습니다.

단일 시점에서 가려진(occluded) 복셀은 같은 시선(frustum) 위의 앞 객체에 묻혀 예측되는 경향이 있습니다. 이를 frustum 내 클래스 분포 정렬로 해결하는 것이 Frustum Proportion Loss($\mathcal{L}_{fp}$)입니다. 이미지를 $\ell \times \ell$ 패치로 분할하면 각 패치는 3D 장면의 하나의 frustum(절두체)에 대응되며, 해당 frustum의 예측 클래스 분포와 GT 분포 사이의 KL 발산을 최소화합니다.

최종 손실은 $\mathcal{L}_{total} = \mathcal{L}_{ce} + \mathcal{L}_{rel} + \mathcal{L}_{scal}^{sem} + \mathcal{L}_{scal}^{geo} + \mathcal{L}_{fp}$ 로 구성됩니다.

**핵심 개념**
- **Scene-Class Affinity Loss**: 클래스별 Precision/Recall/Specificity를 직접 최적화하는 전역 손실. cross-entropy의 맥락 무시 문제 보완
- **Frustum Proportion Loss**: 이미지 패치-frustum 단위 클래스 분포 KL 발산 최소화. occlusion으로 인한 편향 예측 억제
- **Hallucination**: 카메라 FOV 밖의 장면도 통계적으로 그럴듯하게 예측하는 능력. FLoSP가 시선 방향에 따라 학습된 결과

**수식 예제**

$$P_c(\hat{p}, p) = \log \frac{\sum_i \hat{p}_{i,c}}{\sum_i \hat{p}_{i,c}}, \quad R_c(\hat{p}, p) = \log \frac{\sum_i \hat{p}_{i,c}\llbracket p_i = c \rrbracket}{\sum_i \llbracket p_i = c \rrbracket}$$

$$\mathcal{L}_{scal}(\hat{p}, p) = -\frac{1}{C} \sum_{c=1}^{C}(P_c + R_c + S_c)$$

**수식 설명**
- **$\hat{p}_{i,c}$**: 복셀 $i$가 클래스 $c$일 예측 확률
- **$p_i$**: 복셀 $i$의 GT 클래스
- **$P_c$**: 클래스 $c$ 예측 정밀도 (precision) — false positive 억제
- **$R_c$**: 클래스 $c$ 재현율 (recall) — false negative 억제
- **$S_c$**: 특이도 (specificity) — 다른 클래스 복셀을 잘못 예측하지 않도록

**수식 예제**

$$\mathcal{L}_{fp} = \sum_{k=1}^{\ell^2} D_{KL}(P_k \| \hat{P}_k) = \sum_{k=1}^{\ell^2} \sum_{c \in C_k} P_k(c) \log \frac{P_k(c)}{\hat{P}_k(c)}$$

**수식 설명**
- **$k$**: $\ell \times \ell$ 분할된 이미지 패치 인덱스 (기본 $\ell=8$)
- **$P_k(c)$**: frustum $k$에서 클래스 $c$의 GT 비율
- **$\hat{P}_k(c)$**: frustum $k$에서 클래스 $c$의 예측 비율
- **$C_k$**: frustum $k$에 GT가 존재하는 클래스만 (KL이 undefined인 경우 제외)
- **효과**: occlusion으로 가려진 복셀도 해당 frustum의 전체 클래스 분포로 간접 지도

## 📊 주요 실험 및 결과 (Experiments & Results)
- **사용 데이터셋 / 벤치마크**: NYUv2(실내, Kinect, 240×144×240 복셀, 11+1+1 클래스, 640×480 입력), SemanticKITTI(실외, LiDAR scan, 256×256×32 복셀, 19+1+1 클래스, 1226×370 crop 입력)

**NYUv2 (test set)** — mIoU 기준

| 방법 | 입력 | IoU | mIoU |
|---|---|---|---|
| LMSCNet (RGB-inferred) | occ | 33.93 | 15.88 |
| 3DSketch (RGB-inferred) | RGB+TSDF | 38.64 | 22.91 |
| **MonoScene (ours)** | **RGB only** | **42.51** | **26.94** |

**SemanticKITTI (hidden test set)** — mIoU 기준

| 방법 | 입력 | IoU | mIoU |
|---|---|---|---|
| LMSCNet (RGB-inferred) | occ | 31.38 | 7.07 |
| AICNet (RGB-inferred) | RGB+depth | 23.93 | 7.09 |
| **MonoScene (ours)** | **RGB only** | **34.16** | **11.08** |

**Ablation 결과 (NYUv2 test / SemanticKITTI val)**

| 구성 | NYUv2 IoU | NYUv2 mIoU | SemKITTI IoU | SemKITTI mIoU |
|---|---|---|---|---|
| Full Model | 42.51 | 26.94 | 37.12 | 11.50 |
| w/o FLoSP | 28.39 | 14.11 | 27.55 | 4.78 |
| w/o 3D CRP | 41.39 | 26.27 | 36.20 | 10.96 |
| w/o $\mathcal{L}_{scal}^{sem}$ | 42.82 | 25.33 | 36.78 | 9.89 |
| w/o $\mathcal{L}_{fp}$ | 41.90 | 26.37 | 36.74 | 11.11 |

FLoSP 제거 시 IoU가 14포인트 급락하여, 2D-3D 연결이 MonoScene의 핵심 요소임을 확인했습니다.

**FOV 내/외 성능 (SemanticKITTI validation)**

| 영역 | IoU | mIoU |
|---|---|---|
| In-FOV | 39.13 | 12.78 |
| Out-FOV | 31.60 | 7.45 |
| Whole Scene | 37.12 | 11.50 |

카메라 FOV 밖의 장면도 일정 수준 예측 가능함을 확인했습니다(hallucination 능력).

## 💡 결론 및 시사점 (Conclusion & Insights)
MonoScene은 단일 RGB 이미지만으로 실내·실외 양쪽에서 3D SSC를 처음으로 수행한 프레임워크입니다. LiDAR/depth 없이도 3D 입력 기반 일부 기법보다 높은 mIoU를 달성했으며, FLoSP·3D CRP·새 손실 함수라는 세 가지 독립적 기여가 각각 유효함을 ablation으로 검증했습니다.

- TPVFormer, SurroundOcc 등 카메라 전용 3D 점유 예측 계보의 직접적 선행 연구로, "깊이 센서 없이 3D 이해가 가능하다"는 패러다임을 처음 정립했습니다.
- 카메라 전용 3D 씬 이해라는 개념이 이후 Neural Simulation 연구들의 이론적 기반을 제공했습니다.
- 합성 데이터 생성이나 저비용 센서 구성 자율주행 스택에서, 깊이 센서 없는 3D 장면 이해의 실현 가능성을 처음으로 보여준 사례입니다.
- **한계점 및 아쉬운 점**: 세밀한 기하(fine-grained geometry) 추론에 어려움이 있어 작은 물체나 얇은 구조물을 놓치기 쉽습니다. 의미적으로 유사한 클래스(의자/소파, 창문/물체 등)를 혼동하는 경우도 있습니다. 실외 장면에서는 시선 방향을 따라 아티팩트가 발생하는 single viewpoint의 근본적 한계가 있고, 카메라 FOV가 훈련 데이터 설정과 다를 때 성능이 저하됩니다.

---

*관련 논문: [TPVFormer](/posts/papers/tpvformer-tri-perspective-view-3d-semantic-occupancy/), [SurroundOcc](/posts/papers/surroundocc/), [Occ3D](/posts/papers/occ3d-large-scale-3d-occupancy-prediction-benchmark/), [BEVFormer](/posts/papers/bevformer/), [GaussianWorld](/posts/papers/gaussianworld-gaussian-world-model-for-streaming-3d-occupancy-prediction/)*
