---
title: "DIFIX3D+: Improving 3D Reconstructions with Single-Step Diffusion Models"
date: 2026-04-10T09:00:00+09:00
draft: false
categories: ["Papers", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "NeRF", "Novel View Synthesis", "Diffusion", "3D Reconstruction"]
year: 2025
references:
  - "3d-gaussian-splatting"
---

## 💡 한 줄 요약
단일 단계(single-step) 확산 모델 DIFIX를 3D 재구성의 학습 데이터 정제와 추론 시 실시간 후처리 양쪽에 활용해, NeRF·3DGS 렌더링에서 발생하는 아티팩트를 10배 이상 빠르게 제거한다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, Huan Ling
- **소속**: NVIDIA, National University of Singapore, University of Toronto, Vector Institute
- **발행년도**: 2025 (arXiv:2503.01774)
- **주요 기여점**:
  1. SD-Turbo 기반 단일 단계 확산 모델 **DIFIX**를 설계해, 노이즈가 있는 NeRF/3DGS 렌더링 이미지를 한 번의 forward pass로 아티팩트 없는 이미지로 변환
  2. 여러 참조 뷰의 정보를 attention으로 섞는 **크로스-뷰 참조 혼합 레이어**로, 개별 뷰를 따로 보정할 때 생기는 3D 불일치를 줄임
  3. DIFIX를 (a) 3D 재구성 중 학습 데이터를 점진적으로 정제하는 용도와 (b) 추론 시 실시간 후처리 용도 양쪽에 재사용하는 **DIFIX3D+ 파이프라인**을 제안해, NeRF와 3DGS 모두에 적용 가능한 단일 범용 모델로 만듦

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: 노이즈 카메라 포즈 최적화·조명 변화 처리 같은 3D 재구성 자체의 불일치 개선 방법들 → 깊이/법선 맵 등 기하학적 사전(prior)을 활용해 희소 입력에서 품질을 높이는 방법들 → GAN·확산 모델 같은 생성 사전(generative prior)을 3D 재구성에 도입하는 방법들(예: Deceptive-NeRF, 매 학습 스텝마다 확산 모델을 호출해 느림) → 이 논문(단일 단계 확산 모델로 생성 사전의 이점은 유지하면서 속도 문제를 해결)
- **기존 한계점**:
  1. 카메라 포즈·조명 처리를 개선하는 방법들은 관측이 부족한 영역의 아티팩트 자체를 근본적으로 없애지는 못함
  2. 기하학적 사전 기반 방법은 희소 입력 설정에는 효과적이지만, 밀도 높은 캡처에서는 개선 효과가 미미함
  3. 확산 모델 같은 생성 사전은 강력하지만, 매 학습 스텝마다 확산 모델을 쿼리하면 학습 속도가 크게 저하됨
- **이 논문의 접근 방식**: 확산 모델을 매 스텝 호출하는 대신, 단일 단계로 증류된 DIFIX 하나를 (1) 학습 데이터를 미리 정제하는 데, (2) 추론 시 후처리하는 데 재사용함으로써 생성 사전의 이점과 실시간에 가까운 속도를 동시에 확보

## 📑 목차
- Chapter 1: 배경 — NeRF·3DGS의 공통 렌더링 공식
- Chapter 2: DIFIX — 단일 단계 확산 기반 Artifact Fixer
- Chapter 3: DIFIX3D+ — 점진적 3D 업데이트와 실시간 후처리

## 🛠️ Chapter 1: 배경 — NeRF·3DGS의 공통 렌더링 공식

**요약**
NeRF와 3DGS는 표현 방식(좌표 기반 MLP vs 명시적 가우시안 입자)은 다르지만, 둘 다 카메라 광선을 따라 색상과 불투명도를 알파 합성하는 동일한 볼륨 렌더링 공식을 공유합니다. DIFIX3D+는 이 공통 구조 덕분에 두 표현 모두에 동일하게 적용될 수 있습니다.

**핵심 개념**
- **공통 볼륨 렌더링 공식**: NeRF의 밀도 기반 불투명도와 3DGS의 가우시안 기반 불투명도는 형태만 다를 뿐, 둘 다 "앞의 투과율 × 현재 불투명도 × 색상"을 누적하는 동일한 알파 합성 구조
- **타일 기반 래스터화**: 3DGS는 미분 가능한 래스터화로 픽셀당 기여하는 가우시안 수를 결정하지만, 최종 색상 합성 방식은 NeRF와 동일

**수식 예제**

$$\mathcal{C}(\mathbf{p}) = \sum_{i=1}^{N} \alpha_i c_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

**수식 설명**
- **$\mathcal{C}(\mathbf{p})$**: 광선 $\mathbf{p}$의 최종 렌더링 색상
- **$c_i$**: i번째 샘플(또는 가우시안)의 색상
- **$\alpha_i$**: i번째 지점의 불투명도 — NeRF에서는 밀도 $\sigma_i$와 샘플 간격으로, 3DGS에서는 가우시안 중심으로부터의 거리로 계산
- **$\prod_{j=1}^{i-1}(1-\alpha_j)$**: 앞의 모든 지점을 통과하고 남은 빛의 투과율

## 🛠️ Chapter 2: DIFIX — 단일 단계 확산 기반 Artifact Fixer

**요약**
DIFIX는 SD-Turbo(Stable Diffusion의 단일 단계 증류 버전)를 기반으로, 노이즈가 있는 렌더링 이미지와 참조 뷰 이미지를 입력받아 한 번의 forward pass로 아티팩트가 제거된 이미지를 출력합니다. 여러 참조 뷰의 attention 정보를 시점 차원으로 재배열해 섞는 크로스-뷰 참조 혼합 레이어가 3D 일관성을 유지하는 핵심입니다.

**핵심 개념**
- **SD-Turbo + LoRA**: 동결된 VAE 인코더에 LoRA로 파인튜닝된 디코더를 붙여, 사전학습된 확산 모델의 지식은 유지하면서 적은 파라미터만 학습
- **노이즈 레벨 $\tau=200$**: 너무 높으면(예: 1000) 원본과 무관한 환각이 생기고, 너무 낮으면(예: 10) 아티팩트가 그대로 남음 — 실험적으로 200이 최적 균형점
- **4가지 학습 데이터 큐레이션 전략**: 희소 재구성, 사이클 재구성, 크로스 참조, 모델 언더피팅으로 "저하된 렌더링 → 깨끗한 이미지" 쌍을 다양하게 확보

**수식 예제**

$$\mathcal{L} = \mathcal{L}_{\text{Recon}} + \mathcal{L}_{\text{LPIPS}} + 0.5\,\mathcal{L}_{\text{Gram}}$$

**수식 설명**
- **$\mathcal{L}_{\text{Recon}}$**: 픽셀 단위 L2 재구성 손실
- **$\mathcal{L}_{\text{LPIPS}}$**: 인간의 지각에 가까운 perceptual loss
- **$\mathcal{L}_{\text{Gram}}$**: 특징 맵의 채널 간 상관관계(Gram 행렬)를 비교해 텍스처·스타일의 일관성을 유지하는 손실

## 🛠️ Chapter 3: DIFIX3D+ — 점진적 3D 업데이트와 실시간 후처리

**요약**
DIFIX를 렌더링 뷰에 그대로 적용하면 뷰마다 결과가 조금씩 달라져 3D 불일치가 생깁니다. 이를 막기 위해 DIFIX3D+는 두 단계로 DIFIX를 재사용합니다: 먼저 3D 표현을 학습하는 동안 향상된 뷰를 점진적으로 3D에 다시 증류(distill)해 아티팩트가 적은 3D 표현을 만들고, 이후 추론 시에는 렌더링 결과에 DIFIX를 실시간으로 적용하는 신경 향상기로 한 번 더 사용합니다.

**핵심 개념**
- **점진적 3D 업데이트**: 렌더링→DIFIX로 향상→향상된 이미지를 타겟 뷰로 재학습, 을 반복하며(카메라를 살짝 섭동시켜가며) 3D 표현 자체의 품질을 점점 끌어올림
- **실시간 후처리(신경 향상기)**: 이미 학습된 3D 표현의 렌더링 결과에 DIFIX를 한 번 더 적용 — A100 기준 약 76ms로, 매 스텝 확산 모델을 호출하는 방식보다 10배 이상 빠름
- **이중 역할**: 같은 DIFIX 모델이 학습 단계(데이터 정제)와 추론 단계(후처리) 모두에 재사용되어 별도 모델을 추가로 학습할 필요가 없음

## 📊 주요 실험 및 결과 (Experiments & Results)
- **사용 데이터셋 / 벤치마크**: Nerfbusters & DL3DV(in-the-wild 아티팩트 제거), RDS(인하우스 실제 주행 장면)
- **주요 성과**:

| 방법 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|---|---|---|---|---|
| Nerfacto | 17.29 | 0.6214 | 0.4021 | 134.65 |
| NeRFLiX | 17.91 | 0.6560 | 0.3458 | 113.59 |
| 3DGS | 17.66 | 0.6780 | 0.3265 | 113.84 |
| **DIFIX3D+ (Nerfacto)** | **18.32** | **0.6623** | **0.2789** | **49.44** |
| **DIFIX3D+ (3DGS)** | **18.51** | **0.6858** | **0.2637** | **41.77** |

  - Nerfbusters 벤치마크에서 PSNR/SSIM/LPIPS/FID 전 지표 최고 성능을 달성했고, 특히 FID는 비교 방법 대비 2배 이상 개선
  - 실제 주행 장면(RDS)에서도 Nerfacto+DIFIX3D+가 기존 대비 전 지표에서 우수 (PSNR 19.95→21.75)
  - Ablation: DIFIX만 추가(FID 63.77) → 점진적 3D 업데이트 추가 → 실시간 후처리까지 추가(DIFIX3D+, FID 49.44)로 갈수록 단계적으로 품질이 향상되어, 두 재사용 방식이 서로 보완적임을 확인

## 💡 결론 및 시사점 (Conclusion & Insights)
Key Contributions가 논문 저자의 주장이라면, 여기서는 다 읽고 난 뒤의 평가와 실무 적용 관점을 담습니다.
- 확산 모델을 매 학습 스텝 호출하는 대신 단일 단계로 증류해, 생성 사전의 화질 개선 효과는 유지하면서 실시간에 가까운 속도(76ms)를 확보했다는 점이 실무 적용 가능성을 크게 높임
- NeRF와 3DGS 어느 쪽에도 동일한 모델을 꽂아 쓸 수 있어, 기존 재구성 파이프라인에 플러그인 형태로 통합하기 쉬움 — 특히 자율주행처럼 실제 촬영 장면을 재구성해 회귀 테스트용 시뮬레이션 자산을 만드는 워크플로우에서, 극단적 시점의 아티팩트를 후처리로 줄이는 용도로 바로 활용 가능
- **한계점 및 아쉬운 점**:
  - 노이즈 레벨 $\tau$ 같은 하이퍼파라미터에 화질-환각 트레이드오프가 민감하게 반응해, 새로운 도메인에 적용할 때 재튜닝이 필요할 수 있음
  - 점진적 3D 업데이트 과정 자체가 반복적인 재학습을 요구해, 최초 3D 표현을 얻기까지의 전체 파이프라인은 여전히 여러 단계를 거쳐야 함
  - 확산 모델 기반이라 완전히 새로운 콘텐츠를 "만들어내는" 방향으로 실패할 경우(환각) 원본에 없던 구조가 생길 위험이 구조적으로 존재

---

*관련 논문: [NeRF](/posts/papers/nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis/), [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/)*
