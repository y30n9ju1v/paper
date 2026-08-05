---
title: "Evaluating Human Perception of Novel View Synthesis: Subjective Quality Assessment of Gaussian Splatting and NeRF in Dynamic Scenes"
date: 2026-07-26T17:10:00+09:00
draft: false
categories: ["Papers", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "NeRF", "Quality Assessment", "Perceptual Quality", "Human Study", "Dynamic Scene"]
year: 2025
references:
  - "3d-gaussian-splatting"
  - "nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis"
---

## 💡 한 줄 요약
13개 동적(Dynamic) 실제 재구성 장면과 5가지 NeRF/3DGS 알고리즘에 대해 34명의 사람이 직접 평가하는 **주관 품질 평가(SAMVIQ 프로토콜)**를 수행하고, 기존 PSNR·SSIM 등 20여 개 자동 지표가 사람의 지각적 실사 품질을 예측하지 못하는 한계와 **3DGS가 NeRF보다 사람 눈에 압도적으로 실사처럼 평가받음**을 정량 입증했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Yuhang Zhang, Joshua Maraval, Zhengyu Zhang, Nicolas Ramin, Shishun Tian, Lu Zhang (Shenzhen Univ., IRT b-com, INSA Rennes)
- **발행년도**: 2025 (arXiv:2501.08072)
- **주요 기여점**:
  1. **최초의 동적 장면 NeRF vs 3DGS 인간 지각 종합 주관 데이터셋**: 13개 동적 야외/실내 장면에 5가지 NVS 기술(NeRFacto, K-Planes, 3DGS 7k, 3DGS 30k, STGFS)을 적용하여 130개의 렌더링 궤적 비디오를 구축하고 SAMVIQ 주관 평가 수행.
  2. **두 가지 주관 평가 프로토콜 구축 (다시점 vs 단일시점)**: 정답(GT)이 없는 다시점(Multi-View, 360도 및 Panning) 조건과 정답이 명확한 단일시점(Single-View) 참조 조건을 분리 검증.
  3. **20여 개 자동 품질 평가 지표(FR & NR IQA/VQA) 대규모 벤치마킹**: MS-SSIM, LPIPS, DISTS, VMAF, BRISQUE 등의 한계를 상관계수(PLCC, SRCC)로 정밀 측정.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **정적 장면 중심 품질 데이터셋 (NeRF-QA, FFV 등)**: 정적 오브젝트나 NeRF 계열 렌더링 품질에 한정됨.
2. **3DGS 등장 이후 지각 연구의 부재**: 3DGS가 속도는 빠르나 동적 물체가 섞인 실제 환경에서 사람 눈에 NeRF보다 실제처럼 보이는지 주관적 평가가 전무했음.
3. **이 논문의 접근**: 국제 표준 SAMVIQ(ITU-T BT.1788) 지침을 따라 평가자 스크리닝 후 인지적 차이를 정량화.

---

## 📑 목차
- Chapter 1: 주관 품질 평가 실험 설계 및 SAMVIQ 프로토콜
- Chapter 2: 주관 평가 데이터 통계 처리 (Z-Score & MOS 및 이상치 제거)
- Chapter 3: 사람 인지 평가 결과 분석 (3DGS vs NeRF)
- Chapter 4: 20여 개 객관 지표 벤치마킹 및 MS-SSIM 계산
- Chapter 5: 결론 및 시사점

---

## 🛠️ Chapter 1: 실험 설계 및 SAMVIQ 프로토콜

### 1. 요약
13개 동적 장면(움직이는 사람, 차, 동물 포함)에 5가지 모델을 적용합니다. 34명의 피험자가 시력 및 색각 테스트를 통과한 후, SAMVIQ 방식으로 영상을 0~100 사이 슬라이더로 채점했습니다.

---

## 🛠️ Chapter 2: Z-Score 정규화 및 MOS 계산

### 1. 요약
평가자마다 채점 성향(매우 짜게 주거나 관대하게 줌)이 다르므로, Individual Z-Score 정규화와 ITU-R BT.500 아웃라이어 정제 알고리즘을 거쳐 최종 **MOS (Mean Opinion Score)**를 산출합니다.

### 2. 수식 및 파이썬 코드 설명

#### Z-Score 및 MOS 계산
$$z_{i, j, k} = \frac{s_{i, j, k} - \bar{s}_i}{\sigma_i}, \quad \text{MOS}_{j, k} = \frac{1}{M} \sum_{i=1}^M s_{i, j, k}$$

- **$s_{i, j, k}$**: $i$번째 평가자가 $k$번째 장면의 $j$번째 모델에 부여한 원시 점수 (0~100)
- **$\bar{s}_i, \sigma_i$**: $i$번째 평가자의 평균 점수 및 표준편차

```python
import torch

def process_subjective_scores_and_mos(
    raw_scores: torch.Tensor # Shape: (M_subjects, N_models, K_scenes)
) -> tuple:
    """
    평가자 편향 제거를 위한 Z-Score 정규화 및 MOS 계산
    """
    # 1. 평가자별 평균 및 표준편차 계산 (M, 1, 1)
    mean_per_subject = torch.mean(raw_scores, dim=(1, 2), keepdim=True)
    std_per_subject  = torch.std(raw_scores, dim=(1, 2), keepdim=True) + 1e-8
    
    # 2. Z-Score 정규화
    z_scores = (raw_scores - mean_per_subject) / std_per_subject
    
    # 3. 모델별 최종 MOS (Mean Opinion Score) 산출 (N_models, K_scenes)
    mos = torch.mean(raw_scores, dim=0)
    
    return z_scores, mos

# --- 사용 예시 ---
dummy_scores = torch.randint(20, 90, (30, 5, 13)).float() # 30명 평가자, 5개 모델, 13개 씬
z_sc, mos_res = process_subjective_scores_and_mos(dummy_scores)
print("모델 1번의 13개 씬 평균 MOS:", mos_res[0].mean().item())
```

---

## 🛠️ Chapter 3: 객관 지표 벤치마크 및 MS-SSIM 수식 해설

### 1. 요약
참조 지표(FR) 중에서는 **MS-SSIM(Multi-Scale SSIM)**이 PLCC **0.8941**로 사람 판단을 가장 잘 대변했으나, 무참조(NR) 상황에서는 모든 지표의 성능이 큰 폭으로 떨어졌습니다.

### 2. 수식 및 파이썬 코드 설명

$$\text{MS-SSIM}(\mathbf{x}, \mathbf{y}) = [l_M(\mathbf{x}, \mathbf{y})]^{\alpha_M} \prod_{j=1}^M [c_j(\mathbf{x}, \mathbf{y})]^{\beta_j} [s_j(\mathbf{x}, \mathbf{y})]^{\gamma_j}$$

- **$l_M$**: 가장 낮은 해상도 $M$에서의 휘도(Luminance) 비교
- **$c_j, s_j$**: $j$번째 스케일에서의 대비(Contrast) 및 구조(Structure) 비교

```python
import torch
import torch.nn.functional as F

def multi_scale_ssim_score(
    img1: torch.Tensor, # (1, 1, H, W)
    img2: torch.Tensor, # (1, 1, H, W)
    scales: int = 3
) -> torch.Tensor:
    """
    Multi-Scale SSIM 인지 지표 계산 예시
    """
    ms_score = 1.0
    weights = [0.4, 0.3, 0.3]
    
    curr_img1 = img1
    curr_img2 = img2
    
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    for s in range(scales):
        # 1. 국소 평균(휘도)과 분산/공분산(대비·구조) 계산
        mu1 = F.avg_pool2d(curr_img1, 3, stride=1, padding=1)
        mu2 = F.avg_pool2d(curr_img2, 3, stride=1, padding=1)
        sigma1_sq = F.avg_pool2d(curr_img1 * curr_img1, 3, stride=1, padding=1) - mu1**2
        sigma2_sq = F.avg_pool2d(curr_img2 * curr_img2, 3, stride=1, padding=1) - mu2**2
        sigma12 = F.avg_pool2d(curr_img1 * curr_img2, 3, stride=1, padding=1) - mu1 * mu2
        
        # 2. 해당 스케일의 SSIM(휘도 x 대비 x 구조)을 계산해 누적 곱
        ssim_scale = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                     ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        ms_score *= (ssim_scale.mean() ** weights[s])
        
        # 2. 다음 스케일 다운샘플링
        if s < scales - 1:
            curr_img1 = F.avg_pool2d(curr_img1, 2, stride=2)
            curr_img2 = F.avg_pool2d(curr_img2, 2, stride=2)
            
    return ms_score

# --- 사용 예시 ---
i1 = torch.rand(1, 1, 128, 128)
i2 = i1 + torch.randn(1, 1, 128, 128) * 0.05
print("MS-SSIM 산출 점수:", multi_scale_ssim_score(i1, i2).item())
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. NVS 모델별 최종 주관 평가 점수 (MOS)

| 알고리즘 (Method) | 대표 기술 | 다시점 MOS ↑ (Max 100) | 단일시점 MOS ↑ | 정순위 |
|---|---|---|---|---|
| **STGFS** | 동적 3DGS | **57.29** | **54.46** | **1위 (최고)** |
| **3DGS (30k iter)** | 정적 3DGS | **54.15** | **52.31** | **2위** |
| **3DGS (7k iter)** | 빠른 3DGS | 52.61 | 50.12 | 3위 |
| **NeRFacto** | NeRF (하이브리드) | 42.32 | 41.05 | 4위 |
| **K-Planes** | 동적 NeRF | 25.43 | 24.10 | 5위 (최저) |

- **핵심 발견**: 사람 눈에는 **3DGS 계열 모델들이 NeRF 계열보다 현격히 높은 주관 품질 점수(57.29 vs 42.32)**를 받음.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
동적 실제 환경에서 3DGS는 단순 렌더링 속도뿐 아니라 사람의 시각적 인지 품질(MOS) 면에서도 NeRF를 압도하며, PSNR 대신 인지적 품질 지표(MS-SSIM 등)를 3DGS 평가는 물론 시뮬레이션 회귀 테스트의 지표로 선택해야 함을 정량 검증했습니다.

### 2. 한계점 및 아쉬운 점
- **작은 규모**: 평가자 34명(최종 29명), 장면 13개로 규모가 크지 않아 통계적으로 더 다양한 조건에 대한 일반화는 제한적.
- **비교 대상의 한계**: 비교된 3DGS 방법이 정적(GS)과 동적(STGFS) 각 하나씩이라, 최신 3DGS 변형 전반(4D-GS, Street Gaussians 등)에 결론이 그대로 적용되는지는 별도 검증이 필요.
- **자율주행 도메인 미검증**: 실외 자율주행 특유의 조건(다중 카메라 리그, 롤링 셔터, 초원거리 장면)은 다루지 않아, 그 도메인으로의 일반화는 추가 연구 과제로 남음.

---

## 참고 자료
- [논문 arXiv 페이지 (arXiv:2501.08072)](https://arxiv.org/abs/2501.08072)

*관련 논문: [NeRF](/posts/papers/nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis/), [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/)*
