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
SD-Turbo 기반 단일 단계(Single-Step) 증류 확산 모델 **DIFIX**를 설계하여, 3DGS 및 NeRF 렌더링에서 발생하는 아티팩트를 76ms(실시간)에 제거하고, 학습 단계의 3D 데이터 정제와 추론 시 후처리 신경망으로 재사용하는 **DIFIX3D+ 파이프라인**을 제안한다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, Huan Ling (NVIDIA, NUS, U of Toronto, Vector Inst.)
- **발행년도**: 2025 (arXiv:2503.01774)
- **주요 기여점**:
  1. **단일 단계 확산 보정기 (DIFIX)**: 수백 번의 디노이징 스텝 대신 단 1회의 Forward Pass(Single-step)로 아티팩트(부유물, 블러, 노이즈)를 즉시 제거하는 SD-Turbo 기반 생성 모델 설계.
  2. **크로스-뷰 참조 혼합 메커니즘 (Cross-View Reference Attention)**: 여러 참조 뷰(Reference Views)의 가시적 특징을 Cross-Attention으로 상호 결합하여 3D 일관성(3D Consistency) 파괴를 최소화.
  3. **이중 재사용 DIFIX3D+ 파이프라인**: 동일한 DIFIX 확산 모델을 (a) 3D 최적화 시 향상된 타겟 이미지를 제공하는 점진적 3D 증류기(Distiller)와 (b) 추론 시 76ms 신경 향상 후처리기(Neural Enhancer)로 이중 활용.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **기하학적 기저 3D 표현 (NeRF & 3DGS)**: 뛰어난 화질을 보이나 관측 뷰가 부족하거나 입력 카메라 오차가 있을 시 극심한 플로터(Floaters) 아티팩트 발생.
2. **생성적 사전(Generative Prior) 결합 (Deceptive-NeRF 등)**: 일반 확산 모델(Diffusion)을 3D 최적화 루프에 연결했으나, 매 step마다 수십~수백 번의 디노이징으로 인해 훈련 시간이 며칠 이상으로 폭발함.
3. **DIFIX3D+**: 증류된 Single-step 확산 모델을 도입해 생성 사전의 이점을 유지하면서 10배 이상 가속화 달성.

---

## 📑 목차
- Chapter 1: SD-Turbo 기반 단일 단계 확산 보정기 (DIFIX)
- Chapter 2: 크로스-뷰 참조 혼합 (Cross-View Reference Attention)
- Chapter 3: 텍스처 및 정밀도 손실 함수 (Gram Matrix & LPIPS)
- Chapter 4: 점진적 3D 증류 및 실시간 추적 후처리 파이프라인
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: SD-Turbo 기반 단일 단계 확산 보정기 (DIFIX)

### 1. 요약
노이즈가 포함된 3DGS/NeRF 렌더링 이미지 $\hat{C}$를 VAE 잠재 공간(Latent Space)으로 인코딩한 후, 노이즈 레벨 $\tau=200$의 노이즈를 섞고 단 1회의 Forward Pass로 완전한 보정 이미지 $\tilde{C}$를 복원합니다.

### 2. 수식 및 파이썬 코드 설명

#### 단일 단계 잠재 디노이징 모델
$$\mathbf{z}_\tau = \text{Encoder}(\hat{C}) + \sigma_\tau \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

$$\hat{\mathbf{z}}_0 = D_\theta\left(\mathbf{z}_\tau, \tau, \mathbf{c}_{\text{ref}}\right)$$

$$\tilde{C} = \text{Decoder}(\hat{\mathbf{z}}_0)$$

- **$D_\theta$**: LoRA 파인튜닝이 적용된 SD-Turbo 단일 단계 디노이저 U-Net
- **$\mathbf{c}_{\text{ref}}$**: 참조 카메라 뷰에서 추출된 시각적 조건 피처 (Conditioning Feature)

```python
import torch
import torch.nn as nn

class SingleStepDiffusionFixer(nn.Module):
    """
    SD-Turbo 기반 단일 단계(Single-step) 렌더링 아티팩트 보정기
    """
    def __init__(self, latent_dim: int = 4, feature_dim: int = 64):
        super().__init__()
        # 가상의 U-Net LoRA 블록
        self.unet_lora = nn.Sequential(
            nn.Conv2d(latent_dim + feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, kernel_size=3, padding=1)
        )

    def forward(
        self,
        noisy_render_latent: torch.Tensor, # (B, 4, H, W) 렌더링 잠재 벡터
        ref_condition_feat: torch.Tensor   # (B, 64, H, W) 참조 뷰 조건 피처
    ) -> torch.Tensor:
        """
        단 1회의 Forward Pass로 깨끗한 3D 잠재 벡터 z_0 예측
        """
        # 1. 렌더링 잠재 입력과 참조 조건 피처 결합
        inp = torch.cat([noisy_render_latent, ref_condition_feat], dim=1)
        
        # 2. Single-step Denoising Forward
        cleaned_latent = self.unet_lora(inp)
        return cleaned_latent

# --- 사용 예시 ---
render_z = torch.randn(1, 4, 32, 32)
ref_cond = torch.randn(1, 64, 32, 32)
fixer = SingleStepDiffusionFixer()
print("보정된 3D Latent Shape:", fixer(render_z, ref_cond).shape)
```

---

## 🛠️ Chapter 2: 크로스-뷰 참조 혼합 (Cross-View Reference Attention)

### 1. 요약
각 시점을 독립적으로 보정하면 3D 시점 변경 시 깜빡임(Flickering)이 발생합니다. 이를 막기 위해 타겟 뷰의 Query $\mathbf{Q}$와 인접 참조 뷰들의 Key/Value ($\mathbf{K}_{\text{ref}}, \mathbf{V}_{\text{ref}}$)를 Cross-Attention으로 융합합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{A} = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}_{\text{ref}}^T}{\sqrt{d_k}} \right) \mathbf{V}_{\text{ref}}$$

```python
import torch
import torch.nn as nn

class CrossViewReferenceAttention(nn.Module):
    """
    시점 간 3D 일관성 유지를 위한 Cross-View Attention
    """
    def __init__(self, dim: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(
        self,
        target_feat: torch.Tensor, # (B, N, C) 타겟 렌더링 뷰 피처
        ref_feat: torch.Tensor     # (B, M, C) 인접 참조 뷰 피처
    ) -> torch.Tensor:
        Q = self.q_proj(target_feat)
        K = self.k_proj(ref_feat)
        V = self.v_proj(ref_feat)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        out = torch.matmul(attn_weights, V)
        return out

# --- 사용 예시 ---
target = torch.randn(1, 100, 64)
refs = torch.randn(1, 300, 64) # 3개 참조 뷰 융합
cross_attn = CrossViewReferenceAttention()
print("크로스 뷰 어텐션 결과 Shape:", cross_attn(target, refs).shape)
```

---

## 🛠️ Chapter 3: 텍스처 손실 함수 (Gram Matrix Loss)

### 1. 요약
단순 픽셀 오차 외에 디테일한 세부 텍스처 무결성을 유지하기 위해 VGG/DINO 피처의 Gram Matrix 오차를 제2의 학습 손실로 추가합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{G}(\mathbf{F}) = \frac{1}{C \cdot H \cdot W} \mathbf{F} \mathbf{F}^T, \quad \mathcal{L}_{\text{Gram}} = \|\mathbf{G}(\mathbf{F}_{\text{pred}}) - \mathbf{G}(\mathbf{F}_{\text{gt}})\|_\text{F}^2$$

```python
import torch

def compute_gram_matrix_loss(
    feat_pred: torch.Tensor, # (1, C, H, W)
    feat_gt: torch.Tensor    # (1, C, H, W)
) -> torch.Tensor:
    """
    세부 텍스처 일관성을 위한 Gram Matrix 손실 계산
    """
    B, C, H, W = feat_pred.shape
    f_pred = feat_pred.view(C, H * W)
    f_gt = feat_gt.view(C, H * W)
    
    gram_pred = torch.mm(f_pred, f_pred.t()) / (C * H * W)
    gram_gt   = torch.mm(f_gt, f_gt.t()) / (C * H * W)
    
    return torch.norm(gram_pred - gram_gt, p='fro')

# --- 사용 예시 ---
f1 = torch.randn(1, 64, 16, 16)
f2 = torch.randn(1, 64, 16, 16)
print("Gram Matrix 텍스처 손실:", compute_gram_matrix_loss(f1, f2).item())
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. 벤치마크 별 아티팩트 보정 성능 비교 (Nerfbusters 데이터셋)

| 모델 (Baseline) | 보정 기법 (Fixer) | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | 렌더링 지연시간 |
|---|---|---|---|---|---|---|
| **Nerfacto** | None | 17.29 | 0.6214 | 0.4021 | 134.65 | 10 ms |
| **Nerfacto** | NeRFLiX | 17.91 | 0.6560 | 0.3458 | 113.59 | 1500 ms (느림) |
| **Nerfacto** | **DIFIX3D+ (Ours)** | **18.32** | **0.6623** | **0.2789** | **49.44** | **76 ms (실시간)** |
| **3DGS** | None | 17.66 | 0.6780 | 0.3265 | 113.84 | 5 ms |
| **3DGS** | **DIFIX3D+ (Ours)** | **18.51** | **0.6858** | **0.2637** | **41.77** | **76 ms (실시간)** |

- **결과**: FID 지표가 기존 대비 **2배 이상(113 -> 41.77) 대폭 개선**되었으며, Single-step 확산 모델 덕분에 **76ms 후처리 시간**으로 실시간 응용 가능.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
DIFIX3D+는 무거운 확산 모델을 Single-step으로 증류하여 3DGS 및 NeRF의 아티팩트를 76ms에 실시간 정제하는 신경 향상기이자 3D 학습 데이터 증류 파이프라인입니다.

### 2. 실무적 시사점
- **자율주행 회귀 테스트 및 가상 자산 렌더링**: 자율주행 시뮬레이터 렌더링 시 시점 한계로 발생하는 플로터 아티팩트를 DIFIX3D+ 후처리 플러그인으로 즉시 정제 가능.

### 3. 한계점 및 아쉬운 점
- **하이퍼파라미터 민감도**: 노이즈 레벨 $\tau$ 같은 하이퍼파라미터에 화질-환각 트레이드오프가 민감하게 반응해, 새로운 도메인에 적용할 때 재튜닝이 필요할 수 있음.
- **여전히 여러 단계로 이루어진 파이프라인**: 점진적 3D 업데이트 과정 자체가 반복적인 재학습을 요구해, 최초 3D 표현을 얻기까지의 전체 파이프라인은 여러 단계를 거쳐야 함.
- **환각 위험**: 확산 모델 기반이라 완전히 새로운 콘텐츠를 "만들어내는" 방향으로 실패할 경우 원본에 없던 구조가 생길 위험이 구조적으로 존재.

---

## 참고 자료
- [DIFIX3D+ arXiv 논문 (arXiv:2503.01774)](https://arxiv.org/abs/2503.01774)

*관련 논문: [NeRF](/posts/papers/nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis/), [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/)*
