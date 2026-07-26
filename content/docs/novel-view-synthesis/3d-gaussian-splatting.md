---
title: "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
date: 2026-04-10T10:00:00+09:00
draft: false
categories: ["Papers", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "Novel View Synthesis", "Real-Time Rendering", "Neural Rendering"]
year: 2023
references:
  - "nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis"
---

## 💡 한 줄 요약
3D 가우시안이라는 명시적이고 미분 가능한 장면 표현과 타일 기반 CUDA 래스터라이저를 결합해, NeRF급 화질을 유지하면서 최초로 실시간(≥30fps @ 1080p) 노벨 뷰 렌더링을 달성했다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
- **발행년도**: 2023 (SIGGRAPH / ACM Transactions on Graphics, arXiv:2308.04079)
- **주요 기여점**:
  1. 위치·공분산·불투명도·색상(구면조화)을 갖는 **3D 가우시안**을 장면의 명시적 표현으로 채택하고, 희소 SfM 포인트에서 초기화해 SGD로 직접 최적화
  2. 기울기 기반 **적응적 밀도 제어**(분할/복제/제거)로 가우시안 개수와 형태를 훈련 중 스스로 조정
  3. 가시성 순서를 보존하면서 임의 개수의 가우시안에 대해 역전파가 가능한 **타일 기반 GPU 래스터라이저**를 설계해 학습·렌더링 모두를 빠르게 만듦

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: NeRF(MLP 기반 연속 방사장, 고품질이지만 훈련 48시간·렌더링 <1fps) → Plenoxels/Instant-NGP(그리드·해시 기반으로 훈련은 수십 분대로 단축했지만 렌더링은 여전히 10fps 미만) → 이 논문(명시적 포인트 기반 표현 + 전용 래스터라이저로 훈련과 렌더링 속도를 동시에 해결)
- **기존 한계점**:
  1. NeRF 계열은 체적 광선 추적(volumetric ray-marching)과 확률적 샘플링에 의존해 렌더링이 근본적으로 느리고 노이즈가 생김
  2. 최근 빠른 방법들은 속도를 얻는 대신 화질을 희생하거나, 메시/포인트 기반의 명시적 표현은 최적화가 어려움
  3. 그리드 기반(voxel, hash grid) 방식도 여전히 렌더링 속도와 화질 사이에서 트레이드오프가 존재
- **이 논문의 접근 방식**: 장면을 이방성 3D 가우시안들의 집합으로 명시적으로 표현하여 미분 가능한 최적화가 가능하게 하고, 이를 위한 전용 타일 기반 래스터라이저를 직접 설계해 빈 공간 계산을 건너뛰면서 GPU를 최대로 활용

## 📑 목차
- Chapter 1: 3D 가우시안 표현
- Chapter 2: 최적화 및 적응적 밀도 제어
- Chapter 3: 빠른 미분 가능 래스터라이저
- Chapter 4: 구현 세부사항

## 🛠️ Chapter 1: 3D 가우시안 표현

**요약**
장면의 기본 단위를 신경망 가중치가 아니라, 위치·모양·색상·투명도를 갖는 3D 가우시안으로 정의합니다. 카메라 캘리브레이션 과정에서 얻는 희소 SfM 포인트로부터 가우시안들을 초기화한 뒤, 연속적인 볼륨 방사장의 장점(미분 가능성)은 유지하면서 빈 공간에서의 불필요한 계산은 피합니다.

**핵심 개념**
- **명시적 기하학**: 각 가우시안의 위치와 크기가 파라미터로 명확히 존재해 해석과 조작이 쉬움
- **미분 가능성**: 모든 파라미터(위치, 공분산, 불투명도, 색상)가 SGD로 직접 최적화 가능
- **이방성 공분산**: 방향성 있는 공분산을 통해 원판·바늘 모양 등 임의의 형태를 표현

**수식 예제**

$$G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$

**수식 설명**
- **$G(\mathbf{x})$**: 위치 $\mathbf{x}$에서의 가우시안 값 (중심에서 멀어질수록 작아짐)
- **$\mathbf{x}$**: 3D 공간의 좌표 (x, y, z)
- **$\boldsymbol{\mu}$**: 가우시안의 중심 위치 (mean)
- **$\Sigma$**: 공분산 행렬 — 가우시안의 크기와 형태를 결정
- **지수 함수**: 중심에서 먼 점일수록 값이 지수적으로 0에 가까워짐 — 연기가 중심에서 짙고 멀어질수록 옅어지는 모양과 비슷

공분산 행렬은 회전과 스케일의 곱으로 인수분해되어, 최적화 중에도 항상 유효한(양의 준정부호) 공분산을 보장합니다:

$$\Sigma = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T$$

- **$\mathbf{R}$**: 회전 행렬(쿼터니언으로 표현) — 가우시안의 방향을 결정
- **$\mathbf{S}$**: 스케일 대각 행렬 — 각 축 방향(x, y, z)의 크기를 결정

## 🛠️ Chapter 2: 최적화 및 적응적 밀도 제어

**요약**
초기 가우시안 집합만으로는 장면을 정확히 표현할 수 없기 때문에, 렌더링 손실을 기준으로 가우시안의 파라미터를 최적화하는 동시에 가우시안의 개수 자체도 훈련 중에 늘리거나 줄입니다.

**핵심 개념**
- **적응적 밀도 제어**: 기울기가 큰(under-reconstruction) 영역에서는 가우시안을 분할(split)하거나 복제(clone)하고, 기울기가 작은 영역에서는 제거(prune)
- **투명도 리셋**: 지나치게 불투명하고 큰 가우시안이 국소 최적점에 갇히지 않도록 주기적으로 불투명도를 낮춰 재조정
- **최종 제거**: 훈련이 끝날 때까지 투명도가 임계값 이하로 남는 가우시안은 삭제해 메모리 낭비를 방지

**수식 예제**

$$\mathcal{L} = (1-\lambda) \cdot L_1(\hat{C}, C) + \lambda \cdot L_{SSIM}(\hat{C}, C)$$

**수식 설명**
- **$\mathcal{L}$**: 총 손실값 (작을수록 좋음)
- **$L_1(\hat{C}, C)$**: 예측 색상 $\hat{C}$과 실제 색상 $C$의 픽셀별 절대 오차
- **$L_{SSIM}(\hat{C}, C)$**: 사람 눈이 인지하는 구조적 유사성(엣지, 질감 등) 오차
- **$\lambda$** (보통 0.2): 두 손실의 가중치 — 색상 오차 80% + 구조 유사성 20%

## 🛠️ Chapter 3: 빠른 미분 가능 래스터라이저

**요약**
가우시안 표현이 아무리 좋아도 렌더링이 느리면 의미가 없기 때문에, 저자들은 정확한 알파 블렌딩과 가시성 순서를 유지하면서도 임의 개수의 가우시안에 대해 역전파가 가능한 타일 기반 GPU 래스터라이저를 직접 설계했습니다.

**핵심 개념**
- **타일 기반 스플래팅**: 이미지를 16×16 픽셀 타일로 나눠 GPU 캐시 효율과 스레드 병렬성을 극대화
- **단일 Radix 정렬**: 각 가우시안의 키를 64비트(하위 32비트=타일 인덱스, 상위 32비트=깊이)로 인코딩해, 타일·깊이 정렬을 GPU Radix 정렬 한 번으로 처리
- **조기 종료(Early Termination)**: 픽셀의 누적 투명도가 0.9999를 넘으면 남은 가우시안 계산을 건너뜀
- **역방향 패스 최적화**: 순방향에서는 최종 누적 투명도만 저장하고, 역방향에서 뒤→앞 순서로 중간 투명도를 복원해 $O(1)$ 추가 메모리만 사용

**수식 예제**

$$C(\mathbf{p}) = \sum_{i=1}^{n} T_i c_i \alpha_i, \quad T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$$

**수식 설명**
- **$C(\mathbf{p})$**: 픽셀 $\mathbf{p}$의 최종 색상
- **$n$**: 이 픽셀에 영향을 주는 가우시안 개수
- **$c_i, \alpha_i$**: i번째 가우시안의 색상과 불투명도
- **$T_i$**: i번째 가우시안에 도달하기 전까지, 앞의 가우시안들을 통과하고 남은 빛의 투과율

**직관 예시**: 색유리 3장을 겹쳐서 보는 것과 같습니다 — 앞 유리를 통과하고 남은 빛의 비율만큼만 뒤 유리의 색이 더해집니다.

## 🛠️ Chapter 4: 구현 세부사항

**요약**
알고리즘 자체 외에도, 실제로 51분 훈련·93fps 렌더링을 달성하기까지는 최적화기 설정과 수치 안정성 처리가 중요한 역할을 합니다.

**핵심 개념**
- **파라미터별 학습률**: 위치는 $10^{-3}$에서 지수적으로 감소, 회전/스케일 등은 $10^{-2}$에서 시작하는 등 파라미터 종류별로 다른 학습률 스케줄 적용
- **수치 안정성**: 순방향에서는 $\alpha < 1/255$인 가우시안의 블렌딩을 건너뛰고, 역방향에서는 누적 투명도를 0.9999로 클램핑해 0으로 나누는 것을 방지
- **훈련 규모**: 30,000~50,000회 반복, 장면당 7~51분 소요

## 📊 주요 실험 및 결과 (Experiments & Results)
- **사용 데이터셋 / 벤치마크**: Mip-NeRF 360(야외 9개 장면), Tanks and Temples(Truck, Train), Deep Blending(Dr Johnson, Playroom)
- **주요 성과** (Mip-NeRF360 기준 비교):

| 방법 | 훈련 시간 | 렌더링 FPS | PSNR | SSIM | LPIPS |
|------|---------|----------|------|------|-------|
| Mip-NeRF360 | 48h | 0.071 | 24.37 | 0.692 | 0.192 |
| Plenoxels | 26min | 8.2 | 23.08 | 0.626 | 0.241 |
| InstantNGP | 7min | 9.2 | 24.16 | 0.657 | 0.180 |
| **제안 방법 (7K iter)** | **7min** | **134** | **23.6** | **0.668** | **0.183** |
| **제안 방법 (30K iter)** | **51min** | **93** | **27.21** | **0.815** | **0.214** |

  - PSNR/SSIM은 높을수록, LPIPS는 낮을수록 좋음. 48시간 걸리던 최고 화질 방법을 51분으로 단축(약 57배)하면서도 화질은 오히려 근소하게 앞섬
  - Ablation(Table 3, Truck/Garden/Bicycle 평균 PSNR)에서 이방성 공분산, 분할, 복제, 구면조화 항목을 하나씩 제거해보면 모두 화질에 기여함을 확인 (Full 23.90 vs 각 요소 제거 시 19~23점대로 하락)

## 💡 결론 및 시사점 (Conclusion & Insights)
Key Contributions가 논문 저자의 주장이라면, 여기서는 다 읽고 난 뒤의 평가와 실무 적용 관점을 담습니다.
- 명시적·미분가능한 가우시안 표현과 전용 GPU 래스터라이저의 조합만으로 NeRF 계열이 오랫동안 풀지 못한 "화질과 속도의 동시 달성" 문제를 해결했다는 점에서, 복잡한 신경망보다 잘 설계된 단순한 표현이 더 강력할 수 있음을 보여준 사례
- 이후 4D Gaussian Splatting, Street Gaussians, DrivingGaussian 등 자율주행/동적 장면 재구성 연구의 표준 백본이 된 만큼, 파급력이 큰 기반 기술
- **한계점 및 아쉬운 점**:
  - 정적 장면을 가정하므로 동적 객체(움직이는 차량 등)를 다루려면 별도의 확장이 필요 (후속 연구들이 이 지점을 메움)
  - 매우 큰 실외 장면에서는 가우시안 개수가 급증해 메모리 사용량이 커질 수 있음
  - 반투명하거나 뷰 의존적 반사가 강한 물체(유리, 금속) 표현은 여전히 어려움

---

## 참고 자료

**공개 자료**
- [공식 프로젝트 페이지](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [CUDA 구현](https://github.com/graphdeco-inria/gaussian-splatting)
- [WebGL 데모](https://huggingface.co/spaces/dylanebert/3dgaussiansplats)

*관련 논문: [NeRF](/posts/papers/nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis/), [4D Gaussian Splatting](/posts/papers/4d-gaussian-splatting/), [3D Gaussian Ray Tracing](/posts/papers/3d-gaussian-ray-tracing/), [3DGUT](/posts/papers/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting/), [Street Gaussians](/posts/papers/street-gaussians-modeling-dynamic-urban-scenes/), [DrivingGaussian](/posts/papers/driving-gaussian-composite-gaussian-splatting/), [OmniRe](/posts/papers/omnire-omni-urban-scene-reconstruction/), [HUGSIM](/posts/papers/hugsim-real-time-photorealistic-closed-loop-simulator/)*
