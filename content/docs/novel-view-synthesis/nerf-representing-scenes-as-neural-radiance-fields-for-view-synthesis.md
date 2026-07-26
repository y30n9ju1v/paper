---
title: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
date: 2026-04-10T09:00:00+09:00
draft: false
categories: ["Papers", "Novel View Synthesis"]
tags: ["NeRF", "Novel View Synthesis", "Neural Rendering", "3D Reconstruction"]
year: 2020
references: []
---

## 💡 한 줄 요약
장면을 위치·방향을 입력받아 색상·밀도를 출력하는 5D 연속 함수(MLP)로 표현하고, 미분 가능한 체적 렌더링으로 학습함으로써 희소한 입력 이미지만으로 사실적인 새 시점 이미지를 합성한다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng
- **소속**: UC Berkeley, Google Research, UC San Diego
- **발행년도**: 2020 (arXiv:2003.08934)
- **주요 기여점**:
  1. 장면을 위치 $(x,y,z)$와 시선 방향 $(\theta,\phi)$을 입력받아 색상·밀도를 출력하는 5D **Neural Radiance Field**로 표현
  2. 체적 렌더링을 미분 가능한 형태로 이산화하여, RGB 이미지만으로 MLP를 end-to-end 학습
  3. **Positional Encoding**과 **계층적(Coarse-to-Fine) 볼륨 샘플링**을 도입해 MLP의 저주파 편향을 극복하고 세밀한 디테일과 샘플링 효율을 동시에 확보

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: 암묵적 3D 형상 표현(SDF, Occupancy Field)과 이미지 기반 렌더링(라이트 필드 보간, 메쉬 미분 렌더러, 복셀 기반 심층 네트워크)이 각자 발전해오다, NeRF가 이 둘을 "연속 함수 기반 체적 렌더링"이라는 하나의 프레임워크로 통합
- **기존 한계점**:
  1. 암묵적 3D 형상 표현 방법들은 3D 정답 데이터가 필요하거나 단순한 형상에만 적용 가능
  2. 복셀 그리드처럼 이산적인 표현은 해상도를 높일수록 메모리·연산량이 폭발적으로 증가
  3. 라이트 필드 보간, 메쉬 기반 렌더러 등은 해상도 한계나 복잡한 기하 표현의 어려움, 대용량 저장 공간 필요 등의 문제가 있었음
- **이 논문의 접근 방식**: 장면 정보를 복셀이나 메쉬가 아니라 MLP 네트워크의 가중치 자체(~5MB)에 압축적으로 담아, 연속적이고 미분 가능한 5D 함수로 표현

## 📑 목차
- Chapter 1: Neural Radiance Field Scene Representation
- Chapter 2: Volume Rendering with Radiance Fields
- Chapter 3: Optimizing a Neural Radiance Field

## 🛠️ Chapter 1: Neural Radiance Field Scene Representation

**요약**
NeRF는 장면을 5D 연속 함수로 표현합니다: 3D 위치와 시선 방향을 입력받아 색상과 밀도를 출력하는 MLP입니다. 밀도는 시선 방향과 무관하게 위치만으로, 색상은 위치와 방향 모두로 결정되도록 설계해 정반사 같은 방향 의존적 효과(Non-Lambertian)까지 표현합니다.

**핵심 개념**
- **Implicit Neural Representation**: 좌표를 입력으로 받아 해당 위치의 물리량을 출력하는 신경망
- **$\sigma$는 위치만으로 결정**: 물체가 어디 있는지는 보는 방향과 무관
- **색상은 위치+방향으로 결정**: 같은 지점이라도 보는 방향에 따라 색이 달라질 수 있음 (정반사 등)
- **네트워크 구조**: $\mathbf{x}$를 8개 완전연결층(256채널)에 통과시켜 $\sigma$와 특징 벡터를 얻고, 여기에 방향 $\mathbf{d}$를 더해 1개 층(128채널)으로 RGB를 출력

**수식 예제**

$$F_\Theta : (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$$

**수식 설명**
- **$\mathbf{x} = (x, y, z)$**: 3D 공간에서의 위치
- **$\mathbf{d} = (\theta, \phi)$**: 시선 방향 (구면 좌표)
- **$\mathbf{c} = (r, g, b)$**: 해당 위치에서 방출되는 색상
- **$\sigma$**: 체적 밀도 — 그 위치에 물질이 있을 확률 (클수록 불투명)
- **$F_\Theta$**: 가중치 $\Theta$를 가진 MLP 네트워크

## 🛠️ Chapter 2: Volume Rendering with Radiance Fields

**요약**
카메라 레이를 따라 색상과 밀도를 적분해 픽셀 색을 계산합니다. 이 적분은 실제로는 층별 샘플링(stratified sampling)으로 이산화되며, 결과적으로 알파 합성(alpha compositing)과 수학적으로 동일한 형태가 되어 미분 가능성을 유지합니다.

**핵심 개념**
- **체적 렌더링의 미분 가능성**: 이산화된 색상 추정값이 MLP 파라미터에 대해 미분 가능해 역전파로 학습 가능
- **물리적 직관**: 카메라에서 출발한 레이가 각 지점에서 색을 수집하되, 앞의 불투명한 물체가 뒤를 가림

**수식 예제**

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt, \quad T(t) = \exp\!\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\,ds\right)$$

**수식 설명**
- **$t_n, t_f$**: 카메라 레이의 근거리, 원거리 바운드
- **$T(t)$**: 누적 투과율 — 레이가 $t_n$에서 $t$까지 아무 입자도 만나지 않고 통과할 확률. $\sigma$가 크면 빠르게 감소해 뒤의 물체가 가려짐
- **$\sigma(\mathbf{r}(t))$**: 위치 $t$에서의 밀도
- **$\mathbf{c}(\mathbf{r}(t), \mathbf{d})$**: 위치와 방향에 따른 색상

이산화된 형태는 다음과 같습니다:

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \bigl(1 - \exp(-\sigma_i \delta_i)\bigr)\, \mathbf{c}_i, \quad T_i = \exp\!\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)$$

- **$\delta_i = t_{i+1} - t_i$**: 인접 샘플 간 거리
- **$1 - \exp(-\sigma_i \delta_i)$**: i번째 구간의 불투명도 $\alpha_i$ — 밀도가 높거나 구간이 넓을수록 1에 가까워짐
- **$T_i$**: i번째 샘플까지의 누적 투과율

## 🛠️ Chapter 3: Optimizing a Neural Radiance Field

**요약**
기본 MLP만으로는 복잡한 장면의 고주파 세부 정보를 표현하기 어렵기 때문에, Positional Encoding과 계층적 볼륨 샘플링 두 가지를 추가로 도입합니다.

**핵심 개념**
- **Positional Encoding**: MLP는 저주파 함수를 선호하는 경향(spectral bias)이 있어 날카로운 경계나 세밀한 텍스처 표현이 어려움 → 입력 좌표를 사인/코사인의 다중 주파수로 매핑해 고주파 함수 학습을 도움
- **Hierarchical Volume Sampling**: 레이 위에 균일하게 샘플링하면 빈 공간에 샘플이 낭비되므로, Coarse 네트워크(64샘플)로 중요한 영역의 가중치를 먼저 파악하고 그 분포에서 Fine 네트워크용 128샘플을 추가로 뽑아 집중 샘플링
- **구현**: Adam 옵티마이저, 학습률 $5\times10^{-4}\to5\times10^{-5}$ 지수 감소, V100 GPU에서 장면당 100k~300k 이터레이션(1~2일)

**수식 예제**

$$\gamma(p) = \bigl(\sin(2^0 \pi p),\, \cos(2^0 \pi p),\, \cdots,\, \sin(2^{L-1} \pi p),\, \cos(2^{L-1} \pi p)\bigr)$$

**수식 설명**
- **$p$**: 원래 입력 좌표값 ([-1, 1] 정규화)
- **$L$**: 주파수 레벨 수 — 위치 $\mathbf{x}$에는 $L=10$, 방향 $\mathbf{d}$에는 $L=4$ 사용
- 이 인코딩으로 입력이 3차원에서 $2\times3\times L=60$차원으로 확장됨
- **직관**: 낮은 주파수는 거친 형상(전체 모양)을, 높은 주파수는 세밀한 디테일(텍스처, 날카로운 경계)을 표현

학습 손실은 Coarse/Fine 네트워크 예측을 모두 정답과 비교합니다:

$$\mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}} \left[\left\|\hat{C}_c(\mathbf{r}) - C(\mathbf{r})\right\|_2^2 + \left\|\hat{C}_f(\mathbf{r}) - C(\mathbf{r})\right\|_2^2\right]$$

- **$\mathcal{R}$**: 배치 내 카메라 레이 집합 (배치 크기 4096)
- **$\hat{C}_c, \hat{C}_f$**: Coarse/Fine 네트워크의 예측 색상 — Coarse 손실도 포함시켜 Coarse의 가중치 분포가 의미있게 학습되도록 함

## 📊 주요 실험 및 결과 (Experiments & Results)
- **사용 데이터셋 / 벤치마크**: Diffuse Synthetic 360°(DeepVoxels, 4개 물체), Realistic Synthetic 360°(비-Lambertian 8개 물체), Real Forward-Facing/LLFF(실제 촬영 8개 장면)
- **주요 성과**:

| 방법 | Diffuse Synthetic PSNR↑ | Realistic Synthetic PSNR↑ | Real Forward PSNR↑ |
|------|------------------------|--------------------------|-------------------|
| SRN  | 33.20                  | 22.26                    | 22.84             |
| NV   | 29.62                  | 26.05                    | -                 |
| LLFF | 34.38                  | 24.88                    | 24.13             |
| **NeRF (Ours)** | **40.15**     | **31.01**                | **26.50**         |

  - 세 데이터셋 모두에서 PSNR/SSIM/LPIPS 최고 성능을 달성하면서도 모델 크기는 ~5MB로, LLFF 대비 약 3000배 압축
  - Ablation(Realistic Synthetic 기준): 기본 MLP 26.67 → +View Dependence 28.77 → +Positional Encoding 27.66 → +Hierarchical Sampling 30.06 → 전체 31.01. Positional Encoding의 기여가 가장 큼

## 💡 결론 및 시사점 (Conclusion & Insights)
Key Contributions가 논문 저자의 주장이라면, 여기서는 다 읽고 난 뒤의 평가와 실무 적용 관점을 담습니다.
- 장면 전체를 ~5MB MLP 가중치로 압축하면서도 기존 방법을 능가하는 화질을 보여, 이후 10년 가까이 이어질 "신경 장면 표현" 연구 흐름의 시발점이 된 논문
- 카메라 포즈가 있는 RGB 이미지만으로 학습 가능해(3D 스캔·depth sensor 불필요) 데이터 수집 장벽이 낮다는 점이 실무적으로 큰 장점
- **한계점 및 아쉬운 점**:
  - 장면 하나를 학습하는 데 1~2일이 걸리고, 새 장면마다 처음부터 재학습해야 함 (이후 Instant-NGP, 3D Gaussian Splatting 등이 이 문제를 집중적으로 개선)
  - 정적 장면만 다룰 수 있어 동적 객체가 있는 장면(자율주행 등)에는 별도 확장이 필요
  - MLP 가중치에 인코딩된 표현은 직접 해석하거나 편집하기 어려움

---

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/)*
