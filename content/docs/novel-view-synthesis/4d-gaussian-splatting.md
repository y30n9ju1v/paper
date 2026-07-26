---
title: "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering"
date: 2026-04-14T16:00:00+09:00
draft: false
categories: ["Papers", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "Dynamic Scene", "Novel View Synthesis", "Real-Time Rendering"]
year: 2023
references:
  - "3d-gaussian-splatting"
  - "nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis"
---

## 💡 한 줄 요약
하나의 정규(canonical) 3D 가우시안 집합에 시간에 따른 변형(deformation)을 예측하는 네트워크를 결합해, 동적 장면도 실시간(최대 82fps)으로 렌더링하는 4D Gaussian Splatting을 제안한다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, Xinggang Wang
- **소속**: Huazhong University of Science and Technology, Huawei Inc.
- **발행년도**: 2023 (arXiv:2310.08528, ICLR 2024)
- **주요 기여점**:
  1. 프레임마다 독립적인 3DGS를 두는 대신, 하나의 **Canonical 3D Gaussians**와 시간을 입력받는 **Gaussian Deformation Field Network**의 조합으로 동적 장면을 표현해 메모리 복잡도를 O(tN)에서 **O(N+F)**로 줄임
  2. HexPlane 구조에서 착안한 6개의 2D 시공간 플레인으로 인접 가우시안들의 공간·시간적 특징을 효율적으로 인코딩하는 **Spatial-Temporal Structure Encoder** 설계
  3. 위치·회전·크기 변형을 각각 독립된 헤드로 예측하는 **Multi-head Deformation Decoder**로 복잡한 동작을 정확하게 모델링

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: 동적 NeRF 계열(HyperNeRF의 canonical 매핑, TiNeuVox/K-Planes의 4D 볼륨 렌더링, NSFF의 광학 흐름 기반 합성)이 품질은 높였지만 느린 렌더링 문제를 안고 있던 와중, 3D Gaussian Splatting이 정적 장면에서 실시간 렌더링을 달성. 이를 동적 장면으로 확장하려는 DynamicGaussian은 각 가우시안을 프레임별로 독립 추적해 메모리가 선형으로 증가하는 문제가 있었고, 4D-GS는 이를 단일 deformation field로 해결
- **기존 한계점**:
  1. NeRF 기반 동적 방법들은 품질은 높지만 학습·렌더링 속도가 느려 실시간 응용에 부적합
  2. 3D-GS(정적)를 동적 장면에 그대로 적용하려면 각 타임스텝마다 별도의 가우시안 집합이 필요해 메모리/저장 비용이 프레임 수에 비례해 선형 증가
  3. Flow 기반 방법(NSFF)은 프레임 간 광학 흐름에 의존해 장거리 변형에 취약
- **이 논문의 접근 방식**: 객체의 "기본 형태"를 담는 단일 Canonical 3D Gaussians를 유지하고, 각 타임스텝에서는 이 기본 형태로부터의 변형량만 신경망(Deformation Field)으로 예측해 더함으로써, 프레임 수와 무관한 메모리로 동적 장면을 표현

## 📑 목차
- Chapter 1: 4D Gaussian Splatting 프레임워크
- Chapter 2: Gaussian Deformation Field Network
- Chapter 3: 최적화

## 🛠️ Chapter 1: 4D Gaussian Splatting 프레임워크

**요약**
4D-GS는 학습 파라미터인 Canonical 3D Gaussians에, 타임스텝 t에서 예측된 변형량을 더해 그 순간의 가우시안 집합을 만들고, 이를 3DGS와 동일한 미분 가능 래스터라이저로 렌더링합니다. 즉 "정적 장면 하나 + 시간에 따른 변형"이라는 구조로 동적 장면을 표현합니다.

**핵심 개념**
- **Canonical Space**: 모든 타임스텝의 기준이 되는 정규 공간 — 객체의 "기본 형태"가 여기에 저장되고, 각 타임스텝의 변형은 이 기본 형태로부터의 차이로 표현됨
- **메모리 복잡도 O(N+F)**: 가우시안 수 N과 변형 필드 파라미터 수 F만 저장하면 되므로, 프레임별로 3DGS를 복제하는 O(tN) 방식보다 훨씬 효율적

**수식 예제**

$$\tilde{I} = \mathcal{S}(M, \mathcal{G}'), \quad \mathcal{G}' = \Delta\mathcal{G} + \mathcal{G}$$

**수식 설명**
- **$\mathcal{S}$**: 미분 가능한 3DGS 래스터라이저 (뷰 행렬 $M$과 가우시안 집합을 받아 이미지를 렌더링)
- **$\mathcal{G}$**: Canonical 3D Gaussians — 학습되는 기본 가우시안 집합
- **$\Delta\mathcal{G}$**: Gaussian Deformation Field가 타임스텝 $t$에 대해 예측한 변형량
- **$\mathcal{G}'$**: 해당 타임스텝에서 실제로 렌더링에 쓰이는, 변형이 적용된 가우시안 집합

## 🛠️ Chapter 2: Gaussian Deformation Field Network

**요약**
변형량 $\Delta\mathcal{G}$는 두 단계로 예측됩니다. 먼저 Spatial-Temporal Structure Encoder가 인접한 가우시안들의 공간·시간적 특징을 HexPlane 방식으로 압축해서 인코딩하고, 이어서 Multi-head Decoder가 이 특징으로부터 위치·회전·크기 변형을 각각 별도로 예측합니다.

**핵심 개념**
- **HexPlane 인코딩**: 4D(x,y,z,t) 시공간을 (x,y),(x,z),(y,z) 공간 평면 3개와 (x,t),(y,t),(z,t) 시공간 평면 3개, 총 6개의 2D 플레인으로 분해해 메모리를 크게 아낌
- **Multi-head Decoder**: 위치(Position), 회전(Rotation), 크기(Scaling) 변형을 하나의 헤드가 아니라 세 개의 독립된 MLP 헤드가 각각 담당해 복잡한 동작을 더 정확히 모델링

**수식 예제**

$$f_h = \bigcup_l \prod_i \text{interp}(R_l(i,j)), \quad \{(i,j)\} \in \{(x,y), (x,z), (y,z), (x,t), (y,t), (z,t)\}$$

**수식 설명**
- **$R_l(i,j)$**: 두 축 $i,j$로 구성된 2D 플레인의 $l$번째 해상도 피처 맵 (총 6개 조합 × 여러 해상도)
- **$\text{interp}$**: 연속 좌표에서 플레인 피처를 bi-linear 보간으로 조회
- **$\prod_i$**: 서로 다른 축 쌍의 피처를 element-wise 곱으로 결합
- **$\bigcup_l$**: 모든 해상도 레벨의 피처를 이어붙임(concatenate)
- **직관**: 6개의 얇은 2D 판(plane)을 겹쳐서 4D 공간 전체를 근사하는 방식으로, 4D 그리드를 통째로 저장하는 것보다 메모리를 크게 절약

이후 Tiny MLP로 특징을 모으고, 세 개의 헤드가 각각 변형량을 예측합니다:

$$(\mathcal{X}', r', s') = (\mathcal{X} + \Delta\mathcal{X},\; r + \Delta r,\; s + \Delta s)$$

- **$\Delta\mathcal{X}, \Delta r, \Delta s$**: Position/Rotation/Scaling 헤드가 각각 출력하는 위치·회전·크기 변위

## 🛠️ Chapter 3: 최적화

**요약**
SfM 포인트 클라우드로 Canonical Gaussians를 초기화하고, 3000회 정적 워밍업 이후 Deformation Field 학습이 합류합니다. 손실 함수는 렌더링 이미지의 재구성 오차와 HexPlane 피처의 공간적 평활도를 함께 고려합니다.

**핵심 개념**
- **정적 워밍업**: Deformation Field가 처음부터 불안정한 변형을 학습하지 않도록, 초반에는 정적 3DGS처럼 학습한 뒤 변형 학습을 합류시킴
- **Total Variation 손실**: 그리드 기반 표현(HexPlane)이 인접 값 사이에서 급격히 변하지 않도록 정규화

**수식 예제**

$$\mathcal{L} = |\tilde{I} - I| + \mathcal{L}_{tv}$$

**수식 설명**
- **$|\tilde{I} - I|$**: 렌더링 이미지와 정답 이미지 간의 L1 재구성 손실
- **$\mathcal{L}_{tv}$**: HexPlane 피처의 공간적 평활도를 강제하는 Total Variation 손실

## 📊 주요 실험 및 결과 (Experiments & Results)
- **사용 데이터셋 / 벤치마크**: D-NeRF(합성, 단안 카메라), HyperNeRF(실제, 1~2대 카메라), Neu3D(실제, 15~20대 카메라). RTX 3090 단일 GPU 사용
- **주요 성과**:

| 모델 | PSNR↑ | FPS↑ | 학습시간↓ | 저장용량↓ |
|------|-------|------|---------|---------|
| TiNeuVox-B | 32.67 | 1.5 | 28분 | 48 MB |
| 3D-GS (정적) | 23.19 | 170 | 10분 | 10 MB |
| V4D | 31.34 | 2.08 | 6분 | 377 MB |
| **4D-GS (Ours)** | **34.05** | **82** | **8분** | **18 MB** |

  - 합성 데이터셋(D-NeRF)에서 품질(PSNR) 1위를 기록하면서 FPS도 2위 수준, 저장 용량도 최소 수준으로 세 마리 토끼를 동시에 잡음
  - 실제 데이터셋 Neu3D(1352×1014)에서는 30 FPS로 비교 방법 중 유일하게 실시간을 달성하면서 PSNR 31.15의 경쟁력 있는 품질 유지
  - Ablation: HexPlane 인코더를 제거하면 PSNR이 34.05 → 27.05로 크게 하락해, 시공간 인코딩이 품질에 핵심적임을 확인

## 💡 결론 및 시사점 (Conclusion & Insights)
Key Contributions가 논문 저자의 주장이라면, 여기서는 다 읽고 난 뒤의 평가와 실무 적용 관점을 담습니다.
- 3DGS(정적) → 4D-GS(단일 deformation field로 동적 물체 모델링) → DrivingGaussian(정적 배경과 동적 차량을 분리 모델링) 흐름의 핵심 연결 고리로, 이후 자율주행 장면 재구성 연구들이 이 deformation field 개념을 기반 기술로 사용
- HUGSIM 같은 포토리얼리스틱 폐루프 시뮬레이터가 동적 에이전트를 재현하려면 실시간(30+ FPS) 렌더링과 정확한 동작 모델링이 모두 필요한데, 4D-GS가 그 두 조건을 동시에 만족시키는 방법을 제시했다는 점에서 시뮬레이션 인프라 관점에서도 의미가 큼
- **한계점 및 아쉬운 점**:
  - 배경 포인트가 부족하거나 카메라 포즈가 부정확한 대규모 모션 장면에서는 최적화가 불안정
  - 모노큘러 입력에서는 정적/동적 가우시안을 분리하는 명시적 지도(supervision)가 없어 완전히 자동으로 분리되지 않음
  - 도시 규모로 확장할 경우 가우시안 수가 늘면서 HexPlane 쿼리 연산량도 함께 증가해, 자율주행처럼 넓은 장면에는 별도의 확장이 필요 (Street Gaussians, DrivingGaussian 등 후속 연구가 이 지점을 다룸)

---

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [Street Gaussians](/posts/papers/street-gaussians-modeling-dynamic-urban-scenes/), [DrivingGaussian](/posts/papers/driving-gaussian-composite-gaussian-splatting/), [HUGSIM](/posts/papers/hugsim-real-time-photorealistic-closed-loop-simulator/)*
