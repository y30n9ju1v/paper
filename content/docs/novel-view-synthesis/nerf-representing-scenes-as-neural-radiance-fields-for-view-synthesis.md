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
장면 전체를 3D 위치 $(x, y, z)$와 시선 방향 $(\theta, \phi)$을 입력받아 해당 지점의 체적 밀도 $\sigma$와 방출 색상 $(r, g, b)$을 출력하는 **5D 연속 함수(MLP)**로 표현하고, 미분 가능한 체적 렌더링(Volume Rendering)으로 학습함으로써 희소한 RGB 이미지만으로 극도로 사실적인 새 시점 합성(Novel View Synthesis)을 달성했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng (UC Berkeley, Google Research, UCSD)
- **발행년도**: 2020 (arXiv:2003.08934, ECCV 2020 Best Paper Honorable Mention)
- **주요 기여점**:
  1. **5D Neural Radiance Field 표현**: 명시적 메시(Mesh)나 이산 복셀(Voxel) 없이 단 5MB 분량의 연속 신경망(MLP) 가중치 자체로 3D 장면의 기하와 광학적 특성(정반사, 광택 포함)을 암묵적(Implicit) 인코딩.
  2. **미분 가능한 이산 체적 렌더링 (Differentiable Volume Rendering)**: 광선을 따른 이산 알파 블렌딩 적분 방정식을 유도하여 RGB 오차만으로 MLP 파라미터 $\Theta$를 End-to-End 역전파 학습.
  3. **Positional Encoding & 계층적 볼륨 샘플링 (Hierarchical Sampling)**: 고주파 스펙트럼 편향(Spectral Bias)을 극복하는 위치 인코딩과 Coarse-to-Fine 2단계 광선 샘플링으로 디테일과 렌더링 효율을 동시 확보.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **명시적 3D 표현 (복셀, 메쉬, 포인트 클라우드)**: 복셀 해상도 증가 시 메모리가 $\mathcal{O}(N^3)$으로 폭발하고, 미분 가능한 메시 렌더러는 복잡한 형상 최적화가 어려움.
2. **암묵적 표면 표현 (SDF, Occupancy Networks)**: 3D 표면 기하 표현에는 우수하나 밀도가 연속적인 시점 의존적 색상 표현에는 한계 존재.
3. **NeRF**: 체적 방사장(Volume Radiance Field)이라는 물리학적 고전 개념을 연속 신경망과 미분 렌더링으로 통합하여 현대 신경 렌더링의 패러다임 전환을 이끎.

---

## 📑 목차
- Chapter 1: 5D Neural Radiance Field 장면 표현 구조
- Chapter 2: Positional Encoding (위치 인코딩)
- Chapter 3: 연속 및 이산 체적 렌더링 (Volume Rendering)
- Chapter 4: 계층적 볼륨 샘플링 (Coarse-to-Fine Hierarchical Sampling)
- Chapter 5: NeRF 손실 함수 및 파이프라인 최적화
- Chapter 6: 주요 실험 및 결과
- Chapter 7: 결론 및 시사점

---

## 🛠️ Chapter 1: 5D Neural Radiance Field 장면 표현 구조

### 1. 요약
3D 공간 좌표 $\mathbf{x} = (x, y, z)$와 2D 시선 방향 단위 벡터 $\mathbf{d} = (\theta, \phi)$를 입력으로 받아, 해당 지점의 체적 밀도 $\sigma$와 뷰 의존적 RGB 색상 $\mathbf{c} = (r, g, b)$를 출력하는 함수 $F_\Theta$를 MLP로 구성합니다. 물체의 존재 유무인 밀도 $\sigma$는 방향과 무관하되, 반사광을 위해 색상 $\mathbf{c}$는 방향 $\mathbf{d}$의 영향을 받도록 차단 설계합니다.

### 2. 수식 및 파이썬 코드 설명

$$F_\Theta : (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$$

```python
import torch
import torch.nn as nn

class NeRFMLP(nn.Module):
    """
    NeRF 5D 연속 장면 표현 MLP 구조 (8개 FC 층 + Skip Connection)
    """
    def __init__(self, pos_dim: int = 60, dir_dim: int = 24, net_depth: int = 8, net_width: int = 256):
        super().__init__()
        self.pos_dim = pos_dim
        self.dir_dim = dir_dim
        
        # 1. 3D 위치 좌표 처리 백본 (8개 층, 5번째 층에 Skip Connection)
        layers = []
        for i in range(net_depth):
            in_channels = pos_dim if i == 0 else (net_width + pos_dim if i == 5 else net_width)
            layers.append(nn.Linear(in_channels, net_width))
        self.pts_linears = nn.ModuleList(layers)
        
        # 2. 체적 밀도 (Sigma) 출력 헤드
        self.density_linear = nn.Linear(net_width, 1)
        
        # 3. 뷰 의존적 색상 (RGB) 출력 헤드
        self.feature_linear = nn.Linear(net_width, net_width)
        self.views_linear = nn.Linear(net_width + dir_dim, net_width // 2)
        self.rgb_linear = nn.Linear(net_width // 2, 3)

    def forward(self, pos_enc: torch.Tensor, dir_enc: torch.Tensor) -> tuple:
        """
        pos_enc: (N_rays * N_samples, pos_dim)
        dir_enc: (N_rays * N_samples, dir_dim)
        """
        h = pos_enc
        for i, layer in enumerate(self.pts_linears):
            if i == 5:
                h = torch.cat([pos_enc, h], dim=-1) # Skip connection
            h = torch.relu(layer(h))
            
        # 1. 체적 밀도 sigma 출력 (ReLU로 양수 보장)
        sigma = torch.relu(self.density_linear(h))
        
        # 2. 특징 벡터 + 방향 인코딩 결합 -> RGB 출력 (Sigmoid로 [0, 1] 보장)
        feature = self.feature_linear(h)
        h_dir = torch.cat([feature, dir_enc], dim=-1)
        h_dir = torch.relu(self.views_linear(h_dir))
        rgb = torch.sigmoid(self.rgb_linear(h_dir))
        
        return rgb, sigma

# --- 사용 예시 ---
p_enc = torch.randn(1024, 60) # 1024개 샘플점 위치 인코딩
d_enc = torch.randn(1024, 24) # 1024개 샘플점 방향 인코딩
nerf = NeRFMLP()
rgb_out, sigma_out = nerf(p_enc, d_enc)
print("출력 RGB Shape:", rgb_out.shape, "Sigma Shape:", sigma_out.shape)
```

---

## 🛠️ Chapter 2: Positional Encoding (위치 인코딩)

### 1. 요약
MLP 신경망은 입력 데이터의 저주파(Low-frequency) 성분을 선호하는 저주파 편향(Spectral Bias)을 갖습니다. 세밀한 고주파 텍스처와 날카로운 경계를 표현하기 위해 입력 스칼라 좌표 $p$를 다중 주파수 사인/코사인 기저로 인코딩합니다.

### 2. 수식 및 파이썬 코드 설명

$$\gamma(p) = \left( \sin(2^0 \pi p), \ \cos(2^0 \pi p), \ \sin(2^1 \pi p), \ \cos(2^1 \pi p), \ \dots, \ \sin(2^{L-1} \pi p), \ \cos(2^{L-1} \pi p) \right)$$

- **$L$**: 주파수 밴드 차수 ($3\text{D}$ 위치 $\mathbf{x}$에는 $L=10$ 적용으로 60차원, $2\text{D}/3\text{D}$ 시선 방향 $\mathbf{d}$에는 $L=4$ 적용으로 24차원으로 확장)

```python
import torch

class PositionalEncoding(torch.nn.Module):
    """
    입력 좌표 p를 2 * L 차원의 고주파 사인/코사인 삼각함수로 임베딩
    """
    def __init__(self, num_freqs: int = 10, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x Shape: (..., D)
        Return Shape: (..., D * 2 * num_freqs + (D if include_input else 0))
        """
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out.append(torch.sin(torch.pi * freq * x))
            out.append(torch.cos(torch.pi * freq * x))
        return torch.cat(out, dim=-1)

# --- 사용 예시 ---
pos_3d = torch.tensor([[0.5, -0.2, 0.9]]) # (1, 3)
pe = PositionalEncoding(num_freqs=10, include_input=False) # 3 * 2 * 10 = 60 차원
print("3D 위치 인코딩 결과 차원:", pe(pos_3d).shape)
```

---

## 🛠️ Chapter 3: 체적 렌더링 (Volume Rendering)

### 1. 요약
카메라 레이 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ 상에서 근거리 $t_n$부터 원거리 $t_f$까지 탐색하며 연속체적 적분 방정식을 이산화된 알파 블렌딩(Alpha-compositing) 형태로 복원합니다.

### 2. 수식 및 파이썬 코드 설명

#### (1) 연속 체적 적분 방정식
$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt, \quad T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\,ds\right)$$

#### (2) 이산화된 체적 렌더링 방정식
$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \left(1 - \exp(-\sigma_i \delta_i)\right) \mathbf{c}_i, \quad T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)$$

- **$\delta_i = t_{i+1} - t_i$**: 인접 샘플 간 구간 거리
- **$\alpha_i = 1 - \exp(-\sigma_i \delta_i)$**: $i$번째 샘플 구간의 이산 불투명도
- **$T_i$**: $i$번째 샘플 위치까지 도달하는 빛의 누적 투과율

```python
import torch

def raw2outputs(
    raw_rgb: torch.Tensor,     # (N_rays, N_samples, 3)
    raw_sigma: torch.Tensor,   # (N_rays, N_samples, 1)
    z_vals: torch.Tensor,      # (N_rays, N_samples)
    rays_d: torch.Tensor       # (N_rays, 3)
) -> tuple:
    """
    NeRF 이산 체적 렌더링 방정식 구현
    """
    # 1. 샘플 간 거리 delta_i 계산
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # 마지막 샘플 후속 거리 무한대 연장
    dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], dim=-1)
    
    # 광선 방향 크기에 따른 실제 거리 보정
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    # 2. 알파 불투명도 alpha_i = 1 - exp(-sigma_i * dist_i)
    alpha = 1.0 - torch.exp(-raw_sigma.squeeze(-1) * dists) # (N_rays, N_samples)
    
    # 3. 누적 투과율 T_i = exp(-sum_{j<i} sigma_j * dist_j)
    # T_i = cumprod(1 - alpha_j)
    transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
    
    # 4. 가중치 weight_i = T_i * alpha_i
    weights = transmittance * alpha # (N_rays, N_samples)
    
    # 5. 픽셀 최종 색상 적분
    rgb_map = torch.sum(weights[..., None] * raw_rgb, dim=-2) # (N_rays, 3)
    depth_map = torch.sum(weights * z_vals, dim=-1)           # (N_rays,)
    
    return rgb_map, depth_map, weights

# --- 사용 예시 ---
rgb_dummy = torch.rand(10, 64, 3)
sig_dummy = torch.relu(torch.randn(10, 64, 1))
z_dummy = torch.linspace(2.0, 6.0, 64).repeat(10, 1)
ray_dir = torch.tensor([[0.0, 0.0, 1.0]]).repeat(10, 1)

rgb_res, depth_res, w_res = raw2outputs(rgb_dummy, sig_dummy, z_dummy, ray_dir)
print("렌더링된 RGB 이미지 색상 Shape:", rgb_res.shape)
```

---

## 🛠️ Chapter 4: 계층적 볼륨 샘플링 (Coarse-to-Fine Sampling)

### 1. 요약
광선 전체를 균일하게 샘플링하면 공기나 빈 공간(Empty space)에 연산이 낭비됩니다. 먼저 **Coarse 네트워크**($N_c=64$)로 가중치 분포 $w_i$를 탐색한 후, 가중치가 높은 핵심 표면 영역 주변에서 누적분포함수(CDF) 역변환 샘플링(Inverse Transform Sampling)으로 **Fine 네트워크**($N_f=128$)용 정밀 샘플점을 고밀도 채집합니다.

### 2. 수식 및 파이썬 코드 설명

#### CDF 역변환 샘플링 (Inverse Transform Sampling)
$$w_i = T_i (1 - \exp(-\sigma_i \delta_i)), \quad \hat{w}_i = \frac{w_i}{\sum_{j=1}^{N_c} w_j}$$

$$\text{CDF}(k) = \sum_{j=1}^k \hat{w}_j$$

```python
import torch

def sample_pdf(
    bins: torch.Tensor,    # (N_rays, N_samples-1) 구간 경계점
    weights: torch.Tensor, # (N_rays, N_samples-2) Coarse 가중치 w_i
    N_fine_samples: int = 128
) -> torch.Tensor:
    """
    Coarse 가중치 PDF 기반 CDF 역변환 샘플링 (Fine 집중 샘플 생성)
    """
    # 1. 가중치 정규화 (PDF)
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    
    # 2. 누적분포함수 (CDF) 구성
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) # (N_rays, N_samples)
    
    # 3. 균등 난수 u ~ Uniform(0, 1) 샘플링
    u = torch.rand(list(cdf.shape[:-1]) + [N_fine_samples])
    
    # 4. CDF 역변환 (Invert CDF via searchsorted)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    
    # 구간 선형 보간 샘플링
    cdf_g0 = torch.gather(cdf, -1, below)
    cdf_g1 = torch.gather(cdf, -1, above)
    bins_g0 = torch.gather(bins, -1, below)
    bins_g1 = torch.gather(bins, -1, above)
    
    denom = cdf_g1 - cdf_g0
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g0) / denom
    fine_samples = bins_g0 + t * (bins_g1 - bins_g0)
    
    return fine_samples

# --- 사용 예시 ---
bins_ex = torch.linspace(2.0, 6.0, 63).repeat(2, 1)
w_ex = torch.rand(2, 62)
fine_pts = sample_pdf(bins_ex, w_ex, N_fine_samples=128)
print("Fine 고밀도 추가 샘플 Shape:", fine_pts.shape)
```

---

## 🛠️ Chapter 5: NeRF 손실 함수 (Coarse & Fine Loss)

### 1. 요약
Coarse 네트워크와 Fine 네트워크가 각각 예측한 픽셀 색상 $\hat{C}_c(\mathbf{r})$와 $\hat{C}_f(\mathbf{r})$를 Ground Truth 픽셀 $C(\mathbf{r})$와 동시 비교하여 $L_2$ 제곱 오차 손실을 줄입니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}} \left[ \left\| \hat{C}_c(\mathbf{r}) - C(\mathbf{r}) \right\|_2^2 + \left\| \hat{C}_f(\mathbf{r}) - C(\mathbf{r}) \right\|_2^2 \right]$$

```python
import torch
import torch.nn.functional as F

def compute_nerf_loss(
    coarse_rgb: torch.Tensor, # (N_rays, 3) Coarse 예측 색상
    fine_rgb: torch.Tensor,   # (N_rays, 3) Fine 예측 색상
    gt_rgb: torch.Tensor      # (N_rays, 3) 정답 RGB 색상
) -> torch.Tensor:
    """
    Coarse 및 Fine 네트워크 공동 최적화를 위한 MSE 손실 함수
    """
    loss_coarse = F.mse_loss(coarse_rgb, gt_rgb)
    loss_fine   = F.mse_loss(fine_rgb, gt_rgb)
    return loss_coarse + loss_fine

# --- 사용 예시 ---
c_rgb = torch.rand(4096, 3)
f_rgb = torch.rand(4096, 3)
target_rgb = torch.rand(4096, 3)
print("NeRF 총 손실 (Loss):", compute_nerf_loss(c_rgb, f_rgb, target_rgb).item())
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. 대표 데이터셋 화질 벤치마크 (Realistic Synthetic 360°)

| 모델 (Method) | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 용량/메모리 | 렌더링 속도 |
|---|---|---|---|---|---|
| **SRN (Scene Representation Networks)** | 22.26 | 0.846 | 0.170 | 15 MB | 1 FPS 이하 |
| **Neural Volumes (NV)** | 26.05 | 0.893 | 0.160 | 1.5 GB | 3 FPS |
| **LLFF (Local Light Field Fusion)** | 24.88 | 0.901 | 0.114 | 15 GB | 0.1 FPS |
| **NeRF (Ours)** | **31.01** | **0.947** | **0.081** | **5 MB (압축)** | **~0.5 FPS (느림)** |

- **화질 성과**: SOTA 기존 기법 대비 **PSNR +5 dB 이상 압도적 우위**를 점했으며, 3D 장면 전체를 단 5MB 가중치로 보관하는 암묵적 표현의 장점 입증.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
NeRF는 신경망 가중치 자체로 3D 장면의 물리적 방사장(Radiance Field)을 연속 표현하고 미분 체적 렌더링으로 훈련하는 패러다임을 확립하여, 현대 3D 신경 렌더링(Novel View Synthesis) 분야의 시발점이 된 불후의 연구입니다.

### 2. 한계점
- **느린 학습 및 렌더링 속도**: 픽셀당 수백 번의 MLP 평가로 인해 장면당 훈련에 1~2일, 렌더링에 수 초 이상 소요 (이후 Instant-NGP, 3DGS에 의해 극복됨).
- **정적 장면 전용**: 동적 객체가 있는 장면(자율주행 등)에는 별도 확장이 필요.
- **낮은 해석·편집 가능성**: MLP 가중치에 암묵적으로 인코딩된 표현은 직접 해석하거나 편집하기 어려움.

---

## 참고 자료
- [NeRF 공식 프로젝트 페이지](https://www.matthewtancik.com/nerf)
- [ECCV 2020 논문 (arXiv:2003.08934)](https://arxiv.org/abs/2003.08934)

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/)*
