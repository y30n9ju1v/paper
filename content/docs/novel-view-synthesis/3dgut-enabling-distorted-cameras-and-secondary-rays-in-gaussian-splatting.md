---
title: "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting"
date: 2026-04-10T09:00:00+09:00
draft: false
categories: ["Papers", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "Novel View Synthesis", "Ray Tracing", "Autonomous Driving"]
year: 2025
references:
  - "3d-gaussian-splatting"
  - "3d-gaussian-ray-tracing"
---

## 💡 한 줄 요약
3D Gaussian Splatting의 EWA splatting 선형화를 Unscented Transform으로 대체해, 래스터화의 속도를 유지하면서 어안·롤링 셔터 같은 왜곡 카메라와 반사·굴절 같은 secondary ray 효과까지 지원한다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas Moenne-Loccoz, Zan Gojcic
- **소속**: NVIDIA, University of Toronto
- **발행년도**: 2025 (arXiv:2412.12507v2)
- **주요 기여점**:
  1. 3D 가우시안을 이미지 평면에 투영할 때 쓰이던 Jacobian 기반 EWA splatting 선형화를 **Unscented Transform(UT)**으로 대체해, 임의의 비선형 카메라 투영 함수를 코드 수정 없이 직접 지원
  2. 가우시안 응답을 2D 이미지 평면이 아니라 **3D 광선 위에서 직접 평가**하여 투영 함수 역전파 없이도 수치적으로 안정적인 정렬·최댓값 계산이 가능하게 함
  3. UT 기반 렌더링과 3DGRT(광선 추적)의 렌더링 공식을 일치시켜, 1차 광선은 래스터화로 2차 광선(반사/굴절)은 광선 추적으로 처리하는 **하이브리드 렌더링**을 가능하게 함

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: NeRF(느린 볼륨 렌더링) → 3D Gaussian Splatting(EWA splatting 기반 실시간 래스터화, 단 핀홀 카메라·1차 광선만 지원) → FisheyeGS(카메라별 Jacobian을 직접 유도해 어안 렌즈만 지원) / 3DGRT(광선 추적으로 왜곡 카메라와 secondary ray를 지원하지만 래스터화보다 3~4배 느림) → 이 논문(래스터화 속도와 광선 추적의 범용성을 UT로 동시에 확보)
- **기존 한계점**:
  1. 기존 3DGS의 EWA splatting은 투영 함수의 1차 테일러 근사(Jacobian)를 쓰기 때문에, 완벽한 핀홀 카메라에서도 근사 오차가 있고 왜곡이 클수록 오차가 커짐
  2. FisheyeGS처럼 특정 카메라 모델(어안 등)에 맞춰 Jacobian을 직접 유도하는 방식은 카메라 모델마다 새로 유도해야 해 범용성이 부족
  3. 3DGRT 같은 광선 추적 방식은 왜곡 카메라와 secondary ray를 정확히 지원하지만, 최적화된 래스터화 대비 3~4배 느림
- **이 논문의 접근 방식**: 선형화 대신 비선형 변환을 Sigma point로 근사하는 Unscented Transform을 도입해, 임의의 투영 함수 $g$만 정의하면 Jacobian 유도 없이도 래스터화 속도로 왜곡 카메라와 secondary ray를 함께 지원

## 📑 목차
- Chapter 1: 배경 — 3DGS와 EWA Splatting의 한계
- Chapter 2: Unscented Transform 기반 투영
- Chapter 3: 3D 파티클 응답 평가와 정렬
- Chapter 4: 응용 — 왜곡 카메라와 Secondary Ray

## 🛠️ Chapter 1: 배경 — 3DGS와 EWA Splatting의 한계

**요약**
3DGS는 장면을 N개의 3D 가우시안 파티클로 표현하고, 카메라 광선을 따라 색과 불투명도를 알파 합성해 렌더링합니다. 이때 3D 가우시안을 2D 이미지 평면에 투영하는 EWA splatting은 투영 함수를 Jacobian으로 1차 선형화하는데, 이 근사가 왜곡 카메라에서 오차의 근원이 됩니다.

**핵심 개념**
- **3D 가우시안 표현**: 위치 $\boldsymbol{\mu}$와 공분산 $\Sigma = RSS^TR^T$(회전 $R$ × 스케일 $S$)로 정의되는 타원체 밀도 분포
- **체적 파티클 렌더링**: 광선을 따라 가까운 가우시안부터 순서대로 $c_i\alpha_i\prod_{j<i}(1-\alpha_j)$ 형태로 알파 합성 (화가 알고리즘과 동일한 직관)
- **EWA splatting의 한계**: Jacobian $J$는 카메라 모델마다 새로 계산해야 하고, 왜곡이 큰 카메라(어안, 롤링 셔터)일수록 선형 근사 오차가 커짐

**수식 예제**

$$\Sigma' = J_{[2:3]} W \Sigma W^T J_{[2:3]}^T$$

**수식 설명**
- **$\Sigma' \in \mathbb{R}^{2\times2}$**: 이미지 평면에 투영된 2D 가우시안 공분산
- **$W \in SE(3)$**: 월드 → 카메라 좌표 변환 (외부 파라미터)
- **$J$**: 투영 함수를 아핀 근사하기 위한 Jacobian 행렬
- **$[2:3]$**: 3D → 2D로 줄이기 위해 행렬의 처음 두 행만 선택

## 🛠️ Chapter 2: Unscented Transform 기반 투영

**요약**
3DGUT의 핵심은 EWA splatting의 선형화를 Unscented Transform(UT)으로 바꾼 것입니다. UT는 가우시안 분포를 7개의 대표점(Sigma point)으로 근사한 뒤, 각 점을 실제 비선형 투영 함수에 그대로 통과시켜 변환 후 분포(평균·공분산)를 추정합니다. Jacobian이 필요 없으므로 어떤 카메라 모델이든 투영 함수 $g$만 정의하면 바로 지원됩니다.

**핵심 개념**
- **Sigma point**: 중심 1개 + 각 축 방향으로 ±3개, 총 7개의 대표점으로 3D 가우시안을 대표
- **직관**: 가우시안을 하나의 근사식으로 뭉개어 변환하는 대신, 대표점 몇 개를 실제 함수에 직접 통과시켜 "정답에 가까운" 변환 결과를 얻음
- **범용성**: 코드 수정 없이 어안 렌즈, 롤링 셔터, 핀홀 등 임의의 카메라 모델에 동일하게 적용 가능

**수식 예제**

$$\boldsymbol{v}_\mu = \sum_{i=0}^{6} w_i^m \boldsymbol{v}_{x_i}, \qquad \Sigma' = \sum_{i=0}^{6} w_i^\Sigma (\boldsymbol{v}_{x_i} - \boldsymbol{v}_\mu)(\boldsymbol{v}_{x_i} - \boldsymbol{v}_\mu)^T$$

**수식 설명**
- **$\boldsymbol{v}_{x_i} = g(\boldsymbol{x}_i)$**: i번째 Sigma point를 실제 비선형 투영 함수 $g$에 통과시킨 결과
- **$w_i^m, w_i^\Sigma$**: 평균/공분산 추정을 위한 가중치 (중심 점에 더 큰 가중치)
- **$\boldsymbol{v}_\mu, \Sigma'$**: 이미지 평면에서 추정된 2D 가우시안의 평균과 공분산
- **핵심 장점**: Jacobian 없이 $g$를 직접 적용하므로, 왜곡이 아무리 복잡해도 동일한 방식으로 처리 가능

## 🛠️ Chapter 3: 3D 파티클 응답 평가와 정렬

**요약**
3DGS는 2D 이미지 평면에서 가우시안 응답을 평가하지만, 3DGUT는 3DGRT(광선 추적)을 따라 광선 위에서 응답이 최대가 되는 지점을 3D에서 직접, 분석적으로 계산합니다. 이 덕분에 투영 함수의 역전파 없이도 안정적인 그라디언트 계산과, 3DGRT와 동일한 순서의 파티클 정렬(MLAB, k=16)이 가능해집니다.

**핵심 개념**
- **3D 평가의 안정성**: 비선형 투영을 역전파하지 않고 가우시안 좌표계에서 직접 최댓값을 구해 수치적으로 안정적
- **MLAB(Multi-Layer Alpha Blending)**: 픽셀당 k개의 가장 가까운 히트를 저장하고 순서대로 블렌딩해 정확한 정렬을 보장

**수식 예제**

$$\tau_{\max} = \frac{(\boldsymbol{\mu} - \boldsymbol{o})^T \Sigma^{-1} \boldsymbol{d}}{\boldsymbol{d}^T \Sigma^{-1} \boldsymbol{d}}$$

**수식 설명**
- **$\tau_{\max}$**: 광선 $\boldsymbol{r}(\tau)=\boldsymbol{o}+\tau\boldsymbol{d}$ 위에서 가우시안 응답이 최대가 되는 거리
- **$\boldsymbol{o}, \boldsymbol{d}$**: 광선의 원점과 방향
- **직관**: 가우시안 좌표계로 바꿔 보면, 광선에 가장 가까운(수직인) 지점이 곧 최대 응답점

## 🛠️ Chapter 4: 응용 — 왜곡 카메라와 Secondary Ray

**요약**
UT 기반 투영 덕분에 어안 렌즈나 롤링 셔터처럼 카메라 움직임/왜곡이 시간에 따라 달라지는 경우도 투영 함수 하나로 자연스럽게 처리됩니다. 또한 UT 렌더링 공식을 3DGRT와 맞춰, 1차 광선은 래스터화로 빠르게 처리하고 반사·굴절이 필요한 2차 광선만 광선 추적으로 넘기는 하이브리드 렌더링이 가능해집니다.

**핵심 개념**
- **롤링 셔터**: 센서가 위에서 아래로 순차적으로 노출되며 카메라가 계속 움직이는 경우 — 시간 의존적 투영 함수를 UT가 그대로 처리
- **Secondary ray**: 1차 광선이 표면에서 반사·굴절되어 생기는 2차 광선 — 거울, 유리 등 조명 효과 표현에 필요
- **하이브리드 렌더링**: 대부분의 픽셀은 래스터화 속도(200+ FPS)를 유지하면서, 반사/굴절이 필요한 곳만 광선 추적 정확도를 빌려옴

## 📊 주요 실험 및 결과 (Experiments & Results)
- **사용 데이터셋 / 벤치마크**: MipNeRF360·Tanks & Temples(표준 핀홀), Scannet++(어안 카메라), Waymo(자율주행, 롤링 셔터)
- **주요 성과**:
  - MipNeRF360(핀홀): 3DGS와 동등한 PSNR(27.26)을 유지하면서 265+ FPS 렌더링 — 가장 가까운 경쟁자인 3DGRT(52 FPS)보다 약 4배 빠름
  - Scannet++(어안): 카메라 전용 Jacobian을 직접 유도한 FisheyeGS보다도 PSNR(28.15→28.46)·SSIM 모두 우수하면서 가우시안 개수는 절반 이하(1.07M→0.38M)로 감소
  - Waymo(롤링 셔터): 3DGS(29.83) · 3DGRT(29.99)보다 높은 PSNR 30.16 달성 — 단일 구현으로 핀홀·어안·롤링 셔터를 모두 처리 가능함을 실증

## 💡 결론 및 시사점 (Conclusion & Insights)
Key Contributions가 논문 저자의 주장이라면, 여기서는 다 읽고 난 뒤의 평가와 실무 적용 관점을 담습니다.
- Jacobian을 Unscented Transform으로 바꾼다는 단순한 아이디어 하나로, 카메라 모델별 특수 처리 없이 왜곡 카메라와 secondary ray를 함께 푼 점이 실용적으로 크다 — 자율주행·로봇처럼 광각/어안 카메라가 흔한 도메인에서 특히 유용
- 새 카메라 모델을 지원할 때 Jacobian을 새로 유도할 필요 없이 투영 함수 $g$만 정의하면 되므로, 실제 파이프라인에 적용하는 비용이 크게 낮아짐
- **한계점 및 아쉬운 점**:
  - 3D에서 UT를 추가로 평가하는 비용 때문에 순수 3DGS보다는 여전히 느림
  - 왜곡이 매우 심한 경우 투영 결과가 실제 2D 가우시안 형태에서 벗어날 수 있음
  - 겹치는 가우시안들을 광선 위 단일 점(τ_max)으로만 평가하기 때문에, overlap이 많은 영역에서는 정확도가 떨어질 수 있음 (EVER 같은 완전 미분 가능 방식이 대안이 될 수 있음)

---

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [3D Gaussian Ray Tracing](/posts/papers/3d-gaussian-ray-tracing/), [HUGSIM](/posts/papers/hugsim-real-time-photorealistic-closed-loop-simulator/)*
