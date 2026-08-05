---
title: "ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
date: 2026-04-24T12:00:00+09:00
draft: false
categories: ["Papers", "Computer Vision", "Deep Learning"]
tags: ["Vision Transformer", "ViT", "Image Classification", "Self-Attention", "ICLR 2021"]
year: 2021
references:
  - "resnet-deep-residual-learning-for-image-recognition"
  - "attention-is-all-you-need"
---

## 💡 한 줄 요약
CNN 고유의 지역성(Locality) 귀납적 편향(Inductive Bias) 없이, $P \times P$ 패치 단위로 분할된 이미지를 토큰으로 직렬화하고 순수 Transformer Encoder를 대규모 데이터셋(JFT-300M)에 사전학습하여 CNN SOTA(ResNet/BiT)를 4배 낮은 연산 비용으로 압도했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (Google Brain)
- **발행년도**: 2021 (arXiv:2010.11929, ICLR 2021)
- **주요 기여점**:
  1. **이미지의 1D 패치 시퀀스 변환 (Patch Embedding)**: $H \times W \times C$ 이미지를 $P \times P$ 2D 패치로 분할 후 Linear Projection을 거쳐 1D 토큰 시퀀스로 변환.
  2. **[CLS] 토큰 & 1D Positional Embedding**: BERT의 `[CLS]` 토큰 개념을 적용하여 이미지 전역 표현을 집계하고, 학습 가능한 1D 위치 임베딩 주입.
  3. **Pre-LayerNorm Transformer Encoder**: MSA 및 MLP 레이어 전단에 Layer Normalization을 배치하여 대규모 네트워크 훈련 수렴성 확보.
  4. **고해상도 파인튜닝 시 2D 보간 (Bilinear Interpolation)**: 사전학습 대비 해상도가 증가할 때 1D 위치 임베딩을 2D Grid로 재배치하고 쌍선형 보간 적용.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **CNN (ResNet, EfficientNet)**: 인접 픽셀이 관련되어 있다는 귀납적 편향(Locality/Translation Equivariance)을 국소 커널에 하드코딩하여 소규모 데이터 학습에 강하나 데이터 규모 확장의 한계 존재.
2. **Vision Transformer (ViT)**: 귀납적 편향을 배제하고 대규모 데이터(JFT-300M) 사전학습을 통해 전역 패치 간 광범위 Self-Attention 패턴을 스스로 학습.

---

## 📑 목차
- Chapter 1: 패치 임베딩 (Patch Embedding) & [CLS] 토큰 수식
- Chapter 2: Pre-LayerNorm ViT Encoder 블록 수식
- Chapter 3: Multi-Head Self-Attention (MSA) 수식
- Chapter 4: 2D 위치 임베딩 보간 (Bilinear Interpolation)
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: 패치 임베딩 & [CLS] 토큰 수식

### 1. 요약
이미지 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$를 $N = \frac{HW}{P^2}$개 패치 $\mathbf{x}_p^i$로 나누어 Linear Projection Matrix $\mathbf{E}$에 곱한 후 `[CLS]` 토큰 $\mathbf{x}_{\text{class}}$ 및 위치 임베딩 $\mathbf{E}_{\text{pos}}$를 결합합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{z}_0 = [\mathbf{x}_{\text{class}} \;;\; \mathbf{x}_p^1\mathbf{E} \;;\; \mathbf{x}_p^2\mathbf{E} \;;\; \dots \;;\; \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{\text{pos}}$$

$$\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}, \quad \mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$$

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    ViT 2D 이미지 -> 1D Patch Embedding 토큰 변환 모듈 (Conv2d 1x1 stride=P 구현)
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 768, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Conv2d를 이용한 고속 Linear Patch Projection
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B = x.shape[0]
        patches = self.proj(x).flatten(2).transpose(1, 2) # (B, N_patches, D)
        
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, D)
        z0 = torch.cat([cls_tokens, patches], dim=1) # (B, N+1, D)
        z0 = z0 + self.pos_embed
        return z0

# --- 사용 예시 ---
patch_embed = PatchEmbedding(patch_size=16, embed_dim=768, img_size=224)
img = torch.randn(2, 3, 224, 224)
print("ViT z0 입력 토큰 시퀀스 Shape:", patch_embed(img).shape)
```

---

## 🛠️ Chapter 2: Pre-LayerNorm ViT Encoder 블록 수식

### 1. 요약
각 서브레이어 전단에 Layer Normalization을 적용(Pre-LN)하고 잔차 연결(Residual Connection)을 수행합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}, \quad \ell = 1 \dots L$$

$$\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell, \quad \ell = 1 \dots L$$

$$\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$$

```python
import torch
import torch.nn as nn

class ViTEncoderBlock(nn.Module):
    """
    Pre-LayerNorm ViT Encoder Block (MSA + MLP)
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. MSA with Residual
        norm_x1 = self.ln1(x)
        attn_out, _ = self.attn(norm_x1, norm_x1, norm_x1)
        x = x + attn_out
        
        # 2. MLP with Residual
        norm_x2 = self.ln2(x)
        x = x + self.mlp(norm_x2)
        return x

# --- 사용 예시 ---
encoder_block = ViTEncoderBlock(embed_dim=768)
seq_in = torch.randn(2, 197, 768)
print("ViT Encoder Block 출력 Shape:", encoder_block(seq_in).shape)
```

---

## 🛠️ Chapter 3: Multi-Head Self-Attention (MSA) 수식

### 1. 요약
시퀀스 $\mathbf{z}$에서 사영된 $[\mathbf{q}, \mathbf{k}, \mathbf{v}]$ 벡터 간 전역 어텐션 행렬 $A \in \mathbb{R}^{(N+1) \times (N+1)}$을 생성합니다.

### 2. 수식 및 파이썬 코드 설명

$$[\mathbf{q}, \mathbf{k}, \mathbf{v}] = \mathbf{z} \mathbf{U}_{qkv}, \quad \mathbf{U}_{qkv} \in \mathbb{R}^{D \times 3D}$$

$$A = \text{Softmax}\left( \frac{\mathbf{q} \mathbf{k}^T}{\sqrt{D_h}} \right), \quad \text{SA}(\mathbf{z}) = A \mathbf{v}$$

```python
import torch
import torch.nn.functional as F

def compute_patch_self_attention_matrix(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    ViT Single-Head Self-Attention 연산 구현
    q, k, v: (B, N+1, D_h)
    """
    d_h = q.size(-1)
    attn_matrix = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (d_h ** 0.5), dim=-1)
    output = torch.matmul(attn_matrix, v)
    return output, attn_matrix

# --- 사용 예시 ---
q_p = torch.randn(2, 197, 64)
k_p = torch.randn(2, 197, 64)
v_p = torch.randn(2, 197, 64)
out_sa, A_mat = compute_patch_self_attention_matrix(q_p, k_p, v_p)
print("Self-Attention 출력 Shape:", out_sa.shape, "어텐션 맵 Shape:", A_mat.shape)
```

---

## 🛠️ Chapter 4: 2D 위치 임베딩 보간 (Bilinear Interpolation)

### 1. 요약
사전학습 해상도 $224 \times 224$ ($14 \times 14$ Grid)에서 파인튜닝 해상도 $384 \times 384$ ($24 \times 24$ Grid)로 확대될 때 1D 위치 임베딩을 2D Grid로 변경한 후 Bilinear Interpolation 보간합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{E}_{\text{pos\_new}} = \text{BilinearInterpolate2D}\left( \mathbf{E}_{\text{pos\_orig\_2D}}, \ (H_{\text{new}}/P, W_{\text{new}}/P) \right)$$

```python
import torch
import torch.nn.functional as F

def interpolate_pos_embed_2d(
    pos_embed_orig: torch.Tensor, # (1, N_orig+1, D)
    orig_grid_size: int = 14,
    new_grid_size: int = 24
) -> torch.Tensor:
    """
    ViT 파인튜닝 고해상도 적용 시 Positional Embedding 2D Bilinear Interpolation
    """
    D = pos_embed_orig.shape[-1]
    cls_token_pos = pos_embed_orig[:, :1, :]
    patch_pos = pos_embed_orig[:, 1:, :] # (1, 196, D)
    
    # 2D Grid (1, D, 14, 14) 변환
    patch_pos_2d = patch_pos.reshape(1, orig_grid_size, orig_grid_size, D).permute(0, 3, 1, 2)
    
    # 2D Bilinear Interpolation -> (1, D, 24, 24)
    new_patch_pos_2d = F.interpolate(patch_pos_2d, size=(new_grid_size, new_grid_size), mode='bilinear', align_corners=False)
    
    # 1D 토큰 복원 (1, 576, D)
    new_patch_pos = new_patch_pos_2d.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, D)
    
    # [CLS] 위치 임베딩 합체
    return torch.cat([cls_token_pos, new_patch_pos], dim=1) # (1, 577, D)

# --- 사용 예시 ---
pos_orig = torch.randn(1, 197, 768) # 14x14 grid
pos_new = interpolate_pos_embed_2d(pos_orig, orig_grid_size=14, new_grid_size=24)
print("보간된 고해상도 Positional Embedding Shape:", pos_new.shape)
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. ImageNet-1K 탑-1 분류 정확도 및 학습 연산 비용 비교

| 알고리즘 (Method) | 사전학습 데이터셋 | ImageNet 탑-1 Accuracy ↑ | 학습 연산 비용 (TPUv3-days) ↓ |
|---|---|---|---|
| **ResNet-152x4 (BiT-L)** | JFT-300M | 87.54% | 9.9k TPU-days |
| **Noisy Student (EffNet)** | JFT-300M | 88.50% | 12.3k TPU-days |
| **ViT-L/16 (Ours)** | JFT-300M | 87.76% | 0.68k TPU-days |
| **ViT-H/14 (Ours)** | **JFT-300M** | **88.55% (SOTA)** | **2.5k TPU-days (4배 절감)** |

- **결과**: 대규모 JFT-300M 사전학습 시 CNN 기반 SOTA를 완전히 넘어서며 **학습 컴퓨팅 연산량을 4배 가량 절감**.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
ViT는 "An Image is Worth 16x16 Words"라는 표제 아래, 컴퓨터 비전 백본을 CNN에서 Vision Transformer 시대로 완전히 전환시킨 혁명적 논문입니다.

### 2. 한계점 및 아쉬운 점
- 소규모 데이터에서 CNN 대비 열세 — DeiT(데이터 효율 개선), MAE(자기지도 사전학습)로 해결됨.
- 고해상도 이미지에서 $O(N^2)$ 계산 비용 — Swin Transformer(윈도우 Attention)로 해결됨.
- 탐지·분할 등 밀집 예측 태스크 미지원 — ViTDet, DINOv2 등으로 확장됨.
- JFT-300M과 같은 비공개 초대규모 데이터셋에 의존한다는 점은 재현성·접근성 측면에서 아쉬운 부분.

---

## 참고 자료
- [Google Research ViT 공식 GitHub 저장소](https://github.com/google-research/vision_transformer)
- [ICLR 2021 논문 (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929)

*관련 논문: [Attention Is All You Need](/posts/papers/attention-is-all-you-need/), [ResNet](/posts/papers/resnet-deep-residual-learning-for-image-recognition/), [DETR](/posts/papers/detr-end-to-end-object-detection-with-transformers/), [BEVFormer](/posts/papers/bevformer/)*
