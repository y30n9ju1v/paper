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
3D 가우시안이라는 명시적(Explicit)이고 미분 가능한(Differentiable) 장면 표현과 타일 기반 CUDA 래스터라이저를 결합하여, NeRF급 최고 화질을 유지하면서도 최초로 고해상도(1080p) 실시간(≥30fps) 노벨 뷰 렌더링 및 빠른 학습 속도를 동시에 달성했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis (Inria, Université Côte d'Azur)
- **발행년도**: 2023 (SIGGRAPH / ACM Transactions on Graphics, arXiv:2308.04079)
- **주요 기여점**:
  1. **명시적 3D 가우시안 표현 (3D Gaussian Representation)**: 위치, 3D 공분산(회전·스케일), 불투명도, 색상(구면조화 계수)을 지닌 3D 가우시안을 장면의 기본 단위로 채택하고, 희소 SfM 포인트 cloud에서 초기화하여 미분 가능한 최적화 수행
  2. **적응적 밀도 제어 (Adaptive Density Control)**: 최적화 과정에서 위치 기울기(Position Gradient)를 모니터링하여 복잡하거나 디테일이 부족한 영역의 가우시안을 자동으로 분할(Split), 복제(Clone), 제거(Prune)함으로써 장면의 복잡도에 맞게 가우시안 분포를 동적으로 구성
  3. **빠른 미분 가능 타일 기반 래스터라이저 (Fast Differentiable Tile-based Rasterizer)**: 타일(Tile) 단위 스플래팅과 64비트 Radix 정렬, 역방향 패스 오버헤드 최소화를 통해 임의 개수의 가우시안에 대해 고속 알파 블렌딩 및 역전파를 지원하는 GPU 전용 래스터라이저 개발

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **고전적 3D 재구성 & 포인트 렌더링**: SfM/MVS를 통한 희소/조밀 포인트 클라우드 렌더링. 속도는 빠르지만 구멍(Hole)이나 앨리어싱이 발생하고 세밀한 시점 의존적 조명 효과 표현이 어려움.
2. **NeRF (Neural Radiance Fields)**: 연속적 체적(Volume) 표현과 MLP를 사용해 사실적인 복원 화질을 달성했으나, 광선 추적(Ray-marching) 및 확률적 샘플링으로 인해 학습에 수십 시간, 렌더링에 초당 수 프레임 이하의 막대한 연산 비용 발생.
3. **가속화 연구 (Instant-NGP, Plenoxels 등)**: 복셀 그리드, 해시 테이블 구조로 학습 시간을 수 분~수십 분대로 축소했으나, 여전히 렌더링 속도는 실시간(30fps)에 못 미치거나 화질 트레이드오프가 존재함.
4. **3D Gaussian Splatting**: 명시적 포인트 표현의 **빠른 렌더링 속도**와 체적 방사장의 **미분 가능성 및 고화질 표현력**을 융합하여 두 마리 토끼를 모두 잡음.

### 기존 한계점
- **체적 샘플링의 느린 속도**: 광선을 따라 수십~수백 번 평가하는 Ray-marching 방식은 빈 공간(Empty space) 연산 낭비가 큼.
- **불연속적 최적화의 어려움**: 기존 명시적 메시(Mesh)나 포인트 기반 모델은 경계면 미분 불가능성으로 인해 SGD 기반 경량 최적화가 어려움.

---

## 📑 목차
- Chapter 1: 3D 가우시안 장면 표현
- Chapter 2: 3D 가우시안의 2D 투영 및 스플래팅 (EWA Splatting)
- Chapter 3: 구면조화함수(SH)를 이용한 방향 의존적 색상 표현
- Chapter 4: 최적화 및 적응적 밀도 제어 (Adaptive Density Control)
- Chapter 5: 타일 기반 고속 미분 가능 래스터라이저 (Tile-based Rasterizer)
- Chapter 6: 손실 함수 및 파이프라인 구현 세부사항
- Chapter 7: 주요 실험 및 결과
- Chapter 8: 결론 및 시사점

---

## 🛠️ Chapter 1: 3D 가우시안 장면 표현 (3D Gaussian Representation)

### 1. 요약
장면을 공간상의 연속적인 타원체(Ellipsoid)들의 집합으로 나타냅니다. 각 3D 가우시안은 중심 위치 $\boldsymbol{\mu}$, 3D 공분산 행렬 $\Sigma$, 불투명도(Opacity) $\alpha$, 그리고 색상을 나타내는 구면조화(Spherical Harmonics, SH) 계수를 가집니다.

### 2. 핵심 개념
- **명시적 타원체 표현**: 3D 공간 상에서 밀도 분포가 중심 $\boldsymbol{\mu}$로부터 멀어질수록 지수적으로 감소하는 3D 타원체입니다.
- **공분산 인수분해**: 공분산 행렬 $\Sigma$는 항상 양의 준정부호(Positive Semi-Definite)여야 하므로, 회전 행렬 $\mathbf{R}$과 스케일 행렬 $\mathbf{S}$의 곱 $\Sigma = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$로 분해하여 최적화합니다.

### 3. 수식 및 파이썬 코드 설명

#### (1) 3D 가우시안 밀도 함수
3D 공간 좌표 $\mathbf{x} \in \mathbb{R}^3$에서의 가우시안 값 $G(\mathbf{x})$는 다음과 같이 정의됩니다.

$$G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$

- **$\mathbf{x}$**: 밀도를 평가할 3D 점의 좌표 $(x, y, z)$
- **$\boldsymbol{\mu}$**: 가우시안의 중심 위치 $( \mu_x, \mu_y, \mu_z )$
- **$\Sigma$**: $3 \times 3$ 이방성(Anisotropic) 공분산 행렬 (가우시안의 3D 크기 및 방향 결정)

```python
import torch

def compute_3d_gaussian_density(
    x: torch.Tensor,       # Shape: (N, 3) - 평가할 3D 샘플 좌표들
    mu: torch.Tensor,      # Shape: (3,)   - 가우시안 중심 위치
    sigma: torch.Tensor    # Shape: (3, 3) - 3D 공분산 행렬
) -> torch.Tensor:
    """
    3D 가우시안 밀도 G(x) = exp(-0.5 * (x - mu)^T @ Sigma^{-1} @ (x - mu)) 계산
    """
    diff = x - mu.unsqueeze(0)             # (N, 3): 중심과의 거리 벡터
    sigma_inv = torch.inverse(sigma)       # (3, 3): 공분산 행렬의 역행렬
    
    # 이차 형식 (Quadratic form) 계산: (x - mu)^T @ Sigma^{-1} @ (x - mu)
    quad_form = torch.sum((diff @ sigma_inv) * diff, dim=-1)  # (N,)
    
    # 지수 함수를 적용하여 밀도값 계산 (0 ~ 1 사이)
    density = torch.exp(-0.5 * quad_form)
    return density

# --- 사용 예시 ---
mu_example = torch.tensor([0.0, 0.0, 5.0])
sigma_example = torch.diag(torch.tensor([1.0, 2.0, 0.5]))
eval_pts = torch.tensor([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]])
print("중심점 밀도:", compute_3d_gaussian_density(eval_pts, mu_example, sigma_example))
```

---

#### (2) 공분산 행렬의 회전-스케일 분해 (Covariance Factorization)
경사하강법으로 공분산 행렬 $\Sigma$를 직접 최적화하면 물리적으로 불가능한 행렬(양의 준정부호 조건 위반)이 될 수 있습니다. 이를 방지하기 위해 3D 스케일 벡터 $\mathbf{s} \in \mathbb{R}^3$와 단위 쿼터니언(Quaternion) $\mathbf{q} \in \mathbb{R}^4$로 분해합니다.

$$\Sigma = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T$$

- **$\mathbf{S} = \text{diag}(s_x, s_y, s_z)$**: 각 주축(x, y, z) 방향의 크기를 나타내는 대각 스케일 행렬
- **$\mathbf{R}$**: 쿼터니언 $\mathbf{q} = (w, x, y, z)$로부터 유도된 $3 \times 3$ 3D 회전 행렬

```python
import torch

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    정규화된 쿼터니언 (w, x, y, z)를 3x3 회전 행렬 R로 변환
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    # torch.stack으로 구성해야 q에 대한 autograd 그래프가 끊기지 않음
    # (torch.tensor([...])로 감싸면 역전파가 불가능해짐)
    R = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)]),
        torch.stack([    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)]),
        torch.stack([    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)])
    ])
    return R

def compute_covariance_3d(s: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    스케일 s=(sx, sy, sz) 및 쿼터니언 q로부터 유효한 3D 공분산 행렬 Sigma 생성
    """
    # 1. 스케일 대각 행렬 S 구성
    S = torch.diag(s)
    
    # 2. 쿼터니언 정규화 및 회전 행렬 R 생성
    q_norm = q / torch.norm(q)
    R = quaternion_to_rotation_matrix(q_norm)
    
    # 3. M = R @ S
    M = R @ S
    
    # 4. Sigma = M @ M^T (항상 대칭이며 양의 준정부호 보장)
    sigma = M @ M.T
    return sigma

# --- 사용 예시 ---
s = torch.tensor([0.1, 0.5, 0.2])  # 축별 스케일
q = torch.tensor([1.0, 0.0, 0.0, 0.0])  # 회전 없음 (Identity)
print("3D 공분산 행렬 Sigma:\n", compute_covariance_3d(s, q))
```

---

## 🛠️ Chapter 2: 2D 투영 및 스플래팅 (EWA Splatting)

### 1. 요약
3D 공간상의 가우시안을 2D 카메라 이미지 평면으로 투영하여 2D 가우시안(Splat)으로 만듭니다. 저자들은 Zwicker et al.의 EWA(Ellipsoidal Splatting) 기법을 사용하여 3D 공분산 행렬을 2D 화면 공간 공분산 행렬 $\Sigma_{2D}$로 선형 변환합니다.

### 2. 수식 및 파이썬 코드 설명

#### (1) 2D 투영 공분산 행렬 ($\Sigma_{2D}$)
월드 좌표계의 3D 가우시안을 카메라 좌표계로 변환한 후 원근 투영(Perspective Projection)의 자코비안(Jacobian) 행렬 $\mathbf{J}$를 적용합니다.

$$\Sigma_{2D} = \mathbf{J} \mathbf{W} \Sigma \mathbf{W}^T \mathbf{J}^T$$

- **$\mathbf{W}$**: 월드 좌표계에서 카메라 좌표계로의 $3 \times 3$ 회전 변환 행렬
- **$\mathbf{J}$**: 카메라 좌표계 지점 $(t_x, t_y, t_z)$에서의 원근 투영 아핀 근사 자코비안 행렬

$$\mathbf{J} = \begin{bmatrix} \frac{f_x}{t_z} & 0 & -\frac{f_x \cdot t_x}{t_z^2} \\ 0 & \frac{f_y}{t_z} & -\frac{f_y \cdot t_y}{t_z^2} \end{bmatrix}$$

```python
import torch

def project_cov3d_to_cov2d(
    mean3d: torch.Tensor,    # (3,) 월드 좌표계 3D 중심점
    cov3d: torch.Tensor,     # (3, 3) 3D 공분산 행렬
    W: torch.Tensor,         # (4, 4) 월드-카메라 변환 행렬
    focal_x: float,          # 카메라 초점거리 fx
    focal_y: float           # 카메라 초점거리 fy
) -> torch.Tensor:
    """
    3D 공분산을 2D 화면 공간 공분산 cov2d로 투영
    """
    # 1. 3D 중심점을 카메라 좌표계로 변환: t = R_w2c @ mean3d + t_w2c
    R_w2c = W[:3, :3]
    t_w2c = W[:3, 3]
    t = R_w2c @ mean3d + t_w2c
    tx, ty, tz = t[0], t[1], t[2]
    
    # 수치 안정성을 위한 수직 거리 tz 클램핑
    tz = torch.clamp(tz, min=0.001)
    
    # 2. 원근 투영의 자코비안 행렬 J 구성
    J = torch.tensor([
        [focal_x / tz,            0.0, -(focal_x * tx) / (tz ** 2)],
        [         0.0,   focal_y / tz, -(focal_y * ty) / (tz ** 2)]
    ], dtype=torch.float32)
    
    # 3. 누적 투영 변환 행렬 T = J @ R_w2c (2x3)
    T = J @ R_w2c
    
    # 4. 2D 공분산 계산: Sigma_2D = T @ Sigma_3D @ T^T (2x2)
    cov2d = T @ cov3d @ T.T
    
    # 5. Anti-aliasing 저주파 필터 적용 (0.3 픽셀 대각 성분 추가)
    cov2d[0, 0] += 0.3
    cov2d[1, 1] += 0.3
    return cov2d

# --- 사용 예시 ---
W_dummy = torch.eye(4)
W_dummy[2, 3] = 2.0  # z 방향 2.0 이동
mean_3d = torch.tensor([0.5, 0.2, 0.0])
cov_3d = torch.eye(3) * 0.1
print("투영된 2D 공분산:\n", project_cov3d_to_cov2d(mean_3d, cov_3d, W_dummy, 800.0, 800.0))
```

---

#### (2) 픽셀 단위 2D 가우시안 불투명도 ($\alpha$)
화면 픽셀 좌표 $\mathbf{x}'$에서 $i$번째 2D 가우시안의 기여 불투명도 $\alpha_i$는 가우시안 기본 불투명도 $o_i$와 2D 밀도값의 곱으로 계산됩니다.

$$\alpha_i = o_i \cdot \exp\left(-\frac{1}{2}(\mathbf{x}' - \boldsymbol{\mu}'_i)^T \Sigma_{2D}^{-1}(\mathbf{x}' - \boldsymbol{\mu}'_i)\right)$$

- **$\mathbf{x}'$**: 렌더링 대상 픽셀의 2D 화면 좌표 $(u, v)$
- **$\boldsymbol{\mu}'_i$**: $i$번째 가우시안이 화면에 투영된 2D 중심 좌표
- **$o_i \in [0, 1]$**: 최적화 대상인 해당 가우시안의 고유 불투명도 파라미터

```python
import torch

def compute_2d_gaussian_alpha(
    pixel_coord: torch.Tensor,  # (2,) 픽셀 좌표 (u, v)
    mean2d: torch.Tensor,       # (2,) 2D 가우시안 투영 중심 (u_i, v_i)
    cov2d: torch.Tensor,        # (2, 2) 2D 공분산 행렬
    opacity: float              # o_i (기본 불투명도)
) -> torch.Tensor:
    """
    특정 픽셀 위치에서의 2D 가우시안 실효 불투명도 alpha_i 계산
    """
    diff = pixel_coord - mean2d
    cov2d_inv = torch.inverse(cov2d)
    
    # 지수 지수부 계산
    power = -0.5 * (diff @ cov2d_inv @ diff)
    
    # 알파 값 산출
    alpha = opacity * torch.exp(power)
    return torch.clamp(alpha, max=0.99)

# --- 사용 예시 ---
pixel = torch.tensor([400.0, 300.0])
mean_2d = torch.tensor([400.5, 299.8])
cov_2d = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
print("픽셀 알파값:", compute_2d_gaussian_alpha(pixel, mean_2d, cov_2d, opacity=0.9).item())
```

---

## 🛠️ Chapter 3: 구면조화함수(SH)를 이용한 색상 표현

### 1. 요약
물체의 표면은 바라보는 각도에 따라 빛 반사(Specualrity)나 광택이 달라집니다. 이를 표현하기 위해 가우시안의 색상을 고정된 RGB 값이 아니라 **구면조화함수(Spherical Harmonics, SH)** 계수로 유지하고, 시선 방향(View Direction)에 따라 색상을 동적으로 계산합니다.

### 2. 수식 및 파이썬 코드 설명

$$c(\mathbf{v}) = \sum_{l=0}^{D} \sum_{m=-l}^{l} k_l^m Y_l^m(\mathbf{v})$$

- **$\mathbf{v}$**: 가우시안 중심에서 카메라를 바라보는 정규화된 3D 시선 방향 단위 벡터 $(v_x, v_y, v_z)$
- **$Y_l^m$**: $l$차수 $m$모드의 구면조화 기저 함수(Spherical Harmonics Basis)
- **$k_l^m$**: 학습 대상인 RGB 각 채널별 SH 계수 (Degree $D=3$일 때 차원당 16개 계수 사용)

```python
import torch

# SH Degree 0 및 1 기저 상수 정의
C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))
C1 = 0.4886025119029199   # sqrt(3 / (4 * pi))

def eval_sh_degree_1(
    dir_vec: torch.Tensor,   # (3,) 정규화된 시선 방향 벡터 (dx, dy, dz)
    sh_coeffs: torch.Tensor  # (4, 3) 4개 기저에 대한 RGB SH 계수 Matrix
) -> torch.Tensor:
    """
    Degree L=1 구면조화함수를 이용한 시점 의존적 RGB 색상 계산
    """
    dx, dy, dz = dir_vec[0], dir_vec[1], dir_vec[2]
    
    # L=0 (1개) 및 L=1 (3개) 기저 함수 값 계산
    Y_0_0  = C0
    Y_1_m1 = -C1 * dy
    Y_1_0  =  C1 * dz
    Y_1_p1 = -C1 * dx
    
    # 기저 함수와 SH 계수의 선형 결합
    color = (
        sh_coeffs[0] * Y_0_0 +
        sh_coeffs[1] * Y_1_m1 +
        sh_coeffs[2] * Y_1_0 +
        sh_coeffs[3] * Y_1_p1
    )
    
    # RGB 범위 [0, 1]에 적절히 렌더링되도록 바이어스 조정 및 클램핑
    return torch.clamp(color + 0.5, min=0.0, max=1.0)

# --- 사용 예시 ---
view_dir = torch.tensor([0.0, 0.0, 1.0])  # 정면에서 바라봄
sh_coeffs_dummy = torch.randn(4, 3) * 0.1
print("시점에 따른 최종 색상 (RGB):", eval_sh_degree_1(view_dir, sh_coeffs_dummy))
```

---

## 🛠️ Chapter 4: 최적화 및 적응적 밀도 제어 (Adaptive Density Control)

### 1. 요약
희소한 SfM 포인트로 초기화된 가우시안들은 장면 전체의 디테일(미세한 나뭇잎, 얇은 구조 등)을 모두 담지 못합니다. 따라서 학습 중 위치 기울기 $\nabla_{\boldsymbol{\mu}} \mathcal{L}$를 모니터링하여, 정보가 모자란 곳은 가우시안을 **복제(Clone)**하거나 **분할(Split)**하고, 불필요한 가우시안은 **제거(Prune)**합니다.

### 2. 핵심 메커니즘
1. **복제 (Cloning)**: 얇거나 작아서 디테일을 복원하지 못하는 미정밀 영역(Under-reconstruction)의 가우시안을 복제해 위치 이동.
2. **분할 (Splitting)**: 공간을 너무 크게 덮고 있는 과대 영역(Over-reconstruction)의 대형 가우시안 1개를 2개의 소형 가우시안으로 분할(스케일 1.6배 축소).
3. **불투명도 리셋 (Opacity Reset)**: 3,000 Iteration마다 모든 가우시안의 $\alpha$를 0에 가깝게 리셋하여, 공간에 무분별하게 누적된 불투명 가우시안을 걸러냄.

```python
import torch

def adaptive_density_control(
    positions: torch.Tensor,     # (N, 3) 현재 가우시안 중심 좌표들
    scales: torch.Tensor,        # (N, 3) 현재 가우시안 스케일값들
    pos_grads: torch.Tensor,     # (N,) 위치 기울기 크기 ||grad_mu||
    grad_threshold: float = 0.0002,
    scale_threshold: float = 0.01
):
    """
    기울기와 스케일에 따른 가우시안 복제(Clone) 및 분할(Split) 마스크 계산
    """
    # 1. 기울기가 임계값보다 큰 위치 제어 대상 선별
    high_grad_mask = pos_grads > grad_threshold
    max_scale = torch.max(scales, dim=-1).values
    
    # 2. 복제 조건: 기울기가 크면서 크기(scale)가 작은 경우 (Under-reconstruction)
    clone_mask = high_grad_mask & (max_scale <= scale_threshold)
    
    # 3. 분할 조건: 기울기가 크면서 크기(scale)가 큰 경우 (Over-reconstruction)
    split_mask = high_grad_mask & (max_scale > scale_threshold)
    
    # 복제 로직: 동일 가우시안 추가
    cloned_pos = positions[clone_mask]
    
    # 분할 로직: 1개 가우시안을 2개로 분할하며 스케일을 1.6 감소
    split_pos1 = positions[split_mask] + torch.randn_like(positions[split_mask]) * scales[split_mask]
    split_pos2 = positions[split_mask] - torch.randn_like(positions[split_mask]) * scales[split_mask]
    split_scales = scales[split_mask] / 1.6
    
    return {
        "cloned_count": clone_mask.sum().item(),
        "split_count": split_mask.sum().item(),
        "new_split_positions": torch.cat([split_pos1, split_pos2], dim=0)
    }

# --- 사용 예시 ---
pos_dummy = torch.randn(10, 3)
scale_dummy = torch.rand(10, 3) * 0.02
grad_dummy = torch.rand(10) * 0.0005
result = adaptive_density_control(pos_dummy, scale_dummy, grad_dummy)
print(f"복제 개수: {result['cloned_count']}, 분할 개수: {result['split_count']}")
```

---

## 🛠️ Chapter 5: 타일 기반 고속 미분 가능 래스터라이저 (Tile-based Rasterizer)

### 1. 요약
가우시안 표현이 아무리 뛰어나도 초당 수십 프레임으로 그려내지 못하면 의미가 없습니다. 저자들은 화면을 $16 \times 16$ 픽셀 타일로 나누고, GPU 가속 **Radix Sort**로 타일별 깊이 정렬을 한 번에 처리하는 타일 기반 스플래팅 래스터라이저를 C++/CUDA로 직접 구현했습니다.

### 2. 렌더링 방정식 및 파이썬 코드 설명

#### 체적 렌더링 / 알파 합성 (Alpha Compositing)
앞에서 뒤(Front-to-Back) 순서로 정렬된 $N$개의 2D 가우시안을 합성하여 최종 픽셀 색상 $C(\mathbf{p})$를 얻습니다.

$$C(\mathbf{p}) = \sum_{i=1}^{N} T_i c_i \alpha_i, \quad T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$$

- **$C(\mathbf{p})$**: 최종 픽셀 색상 (RGB)
- **$c_i$**: $i$번째 가우시안의 SH로부터 계산된 색상
- **$\alpha_i$**: $i$번째 가우시안의 해당 픽셀에서의 불투명도
- **$T_i$**: $i$번째 가우시안 이전까지 앞의 가우시안들을 통과하고 남은 빛의 **누적 투과율 (Transmittance)**
- **조기 종료 (Early Termination)**: $T_i < 0.0001$ (누적 불투명도가 $0.9999$ 초과)가 되면 뒤쪽 가우시안 계산을 생략하여 연산 가속.

```python
import torch

def rasterize_pixel_colors(
    colors: torch.Tensor,    # Shape: (N, 3) 깊이순으로 정렬된 가우시안 색상들
    alphas: torch.Tensor,    # Shape: (N,) 각 가우시안의 픽셀 알파값 alpha_i
    bg_color: torch.Tensor   # Shape: (3,) 배경 색상 (RGB)
) -> torch.Tensor:
    """
    Front-to-Back 알파 블렌딩 및 Early Termination 구동
    """
    pixel_color = torch.zeros(3)
    transmittance = 1.0  # 초기 투과율 T_1 = 1.0
    
    for i in range(len(colors)):
        alpha_i = alphas[i]
        c_i = colors[i]
        
        # i번째 가우시안의 실효 기여도 weight = T_i * alpha_i
        weight = transmittance * alpha_i
        pixel_color += weight * c_i
        
        # 다음 가우시안을 위한 투과율 갱신: T_{i+1} = T_i * (1 - alpha_i)
        transmittance *= (1.0 - alpha_i)
        
        # 조기 종료 (Early Termination): 투과율이 0.0001 미만이면 더 이상 뒤가 보이지 않음
        if transmittance < 1e-4:
            break
            
    # 남은 투과율만큼 배경 색상 합성
    pixel_color += transmittance * bg_color
    return pixel_color

# --- 사용 예시 ---
colors_dummy = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # 빨, 녹, 파
alphas_dummy = torch.tensor([0.5, 0.8, 0.9])  # 알파값들
bg = torch.tensor([0.0, 0.0, 0.0])  # 검은색 배경
print("최종 알파 합성 픽셀 색상:", rasterize_pixel_colors(colors_dummy, alphas_dummy, bg))
```

---

## 🛠️ Chapter 6: 손실 함수 및 파이프라인 구현 세부사항

### 1. 손실 함수 (Loss Function)
픽셀 단위의 $L_1$ 오차와 인지적 구조 유사도인 $D_{SSIM}$ 손실을 결합하여 최적화를 진행합니다.

$$\mathcal{L} = (1-\lambda) \mathcal{L}_1(\hat{C}, C) + \lambda \mathcal{L}_{\text{SSIM}}(\hat{C}, C)$$

- **$\hat{C}$**: 렌더링된 예측 이미지
- **$C$**: 실제 Ground Truth 이미지
- **$\lambda = 0.2$**: SSIM 손실의 가중치 비율

```python
import torch
import torch.nn.functional as F

def ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    렌더링 이미지와 정답 이미지 간의 SSIM 손실 (1.0 - SSIM) 계산
    img1, img2 Shape: (1, C, H, W)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12   = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1.0 - ssim_map.mean()

def compute_total_loss(
    rendered_img: torch.Tensor, # (1, 3, H, W) 렌더링 결과
    target_img: torch.Tensor,   # (1, 3, H, W) 정답 이미지
    lambda_ssim: float = 0.2
) -> torch.Tensor:
    """
    최종 3DGS 손실 함수 L = (1 - lambda) * L1 + lambda * L_SSIM
    """
    l1 = F.l1_loss(rendered_img, target_img)
    l_ssim = ssim_loss(rendered_img, target_img)
    return (1.0 - lambda_ssim) * l1 + lambda_ssim * l_ssim
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. 대표 데이터셋 성능 비교 (Mip-NeRF 360 데이터셋)

| 방법 (Method) | 훈련 시간 (Train Time) | 렌더링 속도 (FPS) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| **Mip-NeRF 360** | 48 시간 | 0.07 FPS | 27.69 | 0.792 | 0.237 |
| **Plenoxels** | 24 분 | 6.79 FPS | 23.08 | 0.626 | 0.379 |
| **Instant-NGP** | 6 분 | 15.9 FPS | 25.59 | 0.681 | 0.297 |
| **3DGS (7k iter)** | **7 분** | **134 FPS** | **25.60** | **0.770** | **0.210** |
| **3DGS (30k iter)** | **50 분** | **93 FPS** | **27.21** | **0.815** | **0.185** |

- **화질**: Mip-NeRF 360과 동등 이상의 최고 수준 화질(PSNR, SSIM, LPIPS) 달성.
- **속도**: 30k iter 기준 렌더링 속도 **93 FPS**(가우시안 개수가 늘어 7k iter 대비 다소 느려지지만)로도 Instant-NGP 대비 약 6배, NeRF 대비 수천 배 이상 가속하여 1080p 해상도에서 완벽한 실시간 렌더링 성공.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
3D Gaussian Splatting은 기존 NeRF 계열이 오랜 기간 풀지 못했던 **"고화질 렌더링"**, **"빠른 학습 속도"**, **"실시간 렌더링"**이라는 세 가지 토끼를 명시적 가우시안 표현과 타일 기반 GPU 래스터라이저의 결합으로 완벽히 달성한 3D 신경 재구성 분야의 게임 체인저 논문입니다.

### 2. 실무적 시사점
- **자율주행 및 디지털 트윈**: 대규모 도시 장면 시뮬레이터(DrivingGaussian, Street Gaussians 등)의 백본 엔진으로 대세 정착.
- **게임 및 XR**: 별도의 거대한 신경망 평가 없이 CUDA 래스터라이저만으로 동작하므로 모바일/웹(WebGL) 및 VR 헤드셋에 직접 이식 가능.

### 3. 한계점 및 아쉬운 점
- **메모리 폭발 문제**: 장면이 복잡해질수록 가우시안 개수가 수백만 개로 급증하여 GPU VRAM 사양이 커짐 (이후 3DGS 압축 연구 등장).
- **동적 객체 표현의 부재**: 정적(Static) 장면만 지원하여 동적 물체 표현을 위해서는 4D Gaussian Splatting 등의 후속 확장이 필수적임.

---

## 참고 자료
- [공식 프로젝트 페이지](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [공식 GitHub 저장소 (CUDA C++)](https://github.com/graphdeco-inria/gaussian-splatting)
- [WebGL 3DGS Viewer](https://huggingface.co/spaces/dylanebert/3dgaussiansplats)

*관련 논문: [NeRF](/posts/papers/nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis/), [4D Gaussian Splatting](/posts/papers/4d-gaussian-splatting/), [3D Gaussian Ray Tracing](/posts/papers/3d-gaussian-ray-tracing/), [Street Gaussians](/posts/papers/street-gaussians-modeling-dynamic-urban-scenes/), [DrivingGaussian](/posts/papers/driving-gaussian-composite-gaussian-splatting/), [OmniRe](/posts/papers/omnire-omni-urban-scene-reconstruction/), [HUGSIM](/posts/papers/hugsim-real-time-photorealistic-closed-loop-simulator/)*

