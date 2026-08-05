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
3D Gaussian Splatting의 기존 EWA Splatting(Jacobian 선형 근사)을 **Unscented Transform(UT)** 기법으로 교체함으로써, 래스터화 특유의 압도적 속도를 유지하면서도 어안(Fisheye)·롤링 셔터(Rolling Shutter) 등 임의의 비선형 왜곡 카메라와 반사·굴절 2차 광선(Secondary Rays)을 통합 지원한다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas Moenne-Loccoz, Zan Gojcic (NVIDIA, University of Toronto)
- **발행년도**: 2025 (arXiv:2412.12507v2)
- **주요 기여점**:
  1. **Unscented Transform(UT) 기반 3D 가우시안 2D 투영**: 1차 테일러 선형 근사(Jacobian)의 한계를 극복하기 위해 7개의 Sigma Point를 비선형 카메라 투영 함수에 통과시켜 2D 평균 및 공분산을 추정하는 UT 알고리즘 도입.
  2. **3D 광선 상의 분석적 파티클 응답 평가**: 투영 함수의 역전파 오차 없이 수치적으로 안정적인 3D 파티클 중심 거리 $\tau_{\max}$ 계산 및 Multi-Layer Alpha Blending(MLAB) 정렬 수립.
  3. **하이브리드 렌더링 (Hybrid Rendering)**: 1차 광선(Primary Ray)은 UT 기반 고속 래스터화로, 2차 반사/굴절 광선은 3DGRT 레이 트레이싱 파이프라인으로 처리해 화질과 실시간 속도를 동시 달성.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **3D Gaussian Splatting (3DGS)**: 핀홀 카메라 모델 기반 EWA Splatting으로 실시간 렌더링 달성.
2. **FisheyeGS**: 특수 어안 카메라 모델 전용 Jacobian을 직접 유도해 적용했으나, 카메라 모델이 바뀔 때마다 복잡한 분석적 미분을 새로 유도해야 함.
3. **3D Gaussian Ray Tracing (3DGRT)**: 왜곡 카메라와 2차 광선을 모두 지원하나 래스터화 대비 3~4배 느림.
4. **3DGUT**: 임의의 카메라 모델 함수 $g$만 정의하면 Jacobian 유도 없이 UT로 즉시 투영할 수 있어 래스터화의 초고속(200+ FPS)과 레이 트레이싱의 범용성을 결합.

### 기존 EWA Splatting의 한계점
- **선형화 근사 오차**: 투영 자코비안 $\mathbf{J}$는 1차 아핀 근사이므로, 광각 어안 렌즈나 경계 영역에서 2D 타원체 왜곡 오차가 심각함.
- **범용성 결여**: 카메라 렌즈 모델이 바뀔 때마다(어안, 롤링 셔터, 렌즈 왜곡 계수 등) $\mathbf{J}$를 수학적으로 재유도해야 함.

---

## 📑 목차
- Chapter 1: Unscented Transform(UT)을 이용한 비선형 2D 가우시안 투영
- Chapter 2: 3D 공간 상의 분석적 파티클 평가 ($\tau_{\max}$)
- Chapter 3: 1차/2차 광선 하이브리드 렌더링 파이프라인
- Chapter 4: 복합 비선형 카메라 모델 지원 (Fisheye & Rolling Shutter)
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: Unscented Transform(UT) 기반 비선형 2D 투영

### 1. 요약
3D 가우시안의 3D 평균 $\boldsymbol{\mu} \in \mathbb{R}^3$와 3D 공분산 $\Sigma \in \mathbb{R}^{3 \times 3}$를 정밀하게 표현하는 **7개의 대표점(Sigma Points)**을 생성합니다. 각 시그마 포인트를 실제 비선형 카메라 투영 함수 $g(\mathbf{x})$에 직접 통과시킨 후, 결과 2D 점들의 가중 평균과 가중 공분산을 구해 투영 오차 없는 2D 가우시안 $(\boldsymbol{\mu}_{2D}, \Sigma_{2D})$을 복원합니다.

### 2. 수식 및 파이썬 코드 설명

#### (1) 3D Sigma Points 생성
$$\mathbf{X}_0 = \boldsymbol{\mu}$$

$$\mathbf{X}_i = \boldsymbol{\mu} + \sqrt{3 + \kappa} \ [\sqrt{\Sigma}]_i, \quad \mathbf{X}_{i+3} = \boldsymbol{\mu} - \sqrt{3 + \kappa} \ [\sqrt{\Sigma}]_i \quad (i=1, 2, 3)$$

- **$\sqrt{\Sigma}$**: Cholesky 분해로 구한 하삼각 행렬 $\mathbf{L}$ ($\Sigma = \mathbf{L}\mathbf{L}^T$)
- **$\kappa$**: 척도 파라미터 (Scale Parameter, 보통 0 사용)

#### (2) 2D 비선형 변환 및 2D 평균/공분산 재구성
각 시그마 포인트를 투영 함수 $g$에 통과시킵니다: $\mathbf{Y}_i = g(\mathbf{X}_i) \in \mathbb{R}^2$

$$\boldsymbol{\mu}_{2D} = \sum_{i=0}^6 w_i^m \mathbf{Y}_i, \quad \Sigma_{2D} = \sum_{i=0}^6 w_i^c (\mathbf{Y}_i - \boldsymbol{\mu}_{2D})(\mathbf{Y}_i - \boldsymbol{\mu}_{2D})^T + \mathbf{I} \cdot 0.3$$

```python
import torch

def generate_3d_sigma_points(
    mu: torch.Tensor,    # (3,) 3D 중심점
    cov3d: torch.Tensor, # (3, 3) 3D 공분산 행렬
    kappa: float = 0.0
) -> torch.Tensor:
    """
    3D 가우시안 분포를 대표하는 7개의 Sigma Points (3, 7) 생성
    """
    n = 3
    gamma = torch.sqrt(torch.tensor(n + kappa))
    
    # Cholesky 분해 L @ L^T = cov3d
    L = torch.linalg.cholesky(cov3d)
    
    sigma_points = torch.zeros((7, 3))
    sigma_points[0] = mu
    
    for i in range(3):
        sigma_points[i + 1]     = mu + gamma * L[:, i]
        sigma_points[i + 1 + 3] = mu - gamma * L[:, i]
        
    return sigma_points # Shape: (7, 3)

def unscented_transform_2d_projection(
    mu3d: torch.Tensor,
    cov3d: torch.Tensor,
    camera_proj_fn  # 임의의 비선형 카메라 투영 콜백 함수 g(X)
) -> tuple:
    """
    Unscented Transform(UT)을 이용한 비선형 2D 평균 및 공분산 추정
    """
    # 1. 7개 Sigma Points 생성
    sigma_pts_3d = generate_3d_sigma_points(mu3d, cov3d)
    
    # 2. 비선형 카메라 투영 함수에 통과 (7, 2)
    sigma_pts_2d = camera_proj_fn(sigma_pts_3d)
    
    # 3. UT 가중치 설정 (n=3)
    w_m0 = 0.0  # kappa / (n + kappa)
    w_c0 = 0.0
    w_i = 1.0 / (2 * (3 + 0.0))
    
    weights_m = torch.tensor([w_m0] + [w_i] * 6)
    weights_c = torch.tensor([w_c0] + [w_i] * 6)
    
    # 4. 2D 평균 계산
    mu2d = torch.sum(weights_m.unsqueeze(-1) * sigma_pts_2d, dim=0)
    
    # 5. 2D 공분산 계산
    diff = sigma_pts_2d - mu2d.unsqueeze(0) # (7, 2)
    cov2d = torch.zeros((2, 2))
    for i in range(7):
        cov2d += weights_c[i] * torch.outer(diff[i], diff[i])
        
    # Anti-aliasing 저주파 필터
    cov2d[0, 0] += 0.3
    cov2d[1, 1] += 0.3
    
    return mu2d, cov2d

# --- 사용 예시 ---
def dummy_fisheye_camera(pts_3d: torch.Tensor) -> torch.Tensor:
    # 예시 어안 카메라 투영 함수 (x/z, y/z 및 렌더 왜곡)
    x, y, z = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]
    r = torch.sqrt(x**2 + y**2) / z
    theta = torch.atan(r)
    u = (x / (r + 1e-6)) * theta * 500.0 + 400.0
    v = (y / (r + 1e-6)) * theta * 500.0 + 300.0
    return torch.stack([u, v], dim=-1)

mu = torch.tensor([0.5, 0.2, 3.0])
cov = torch.eye(3) * 0.1
mu_2d, cov_2d = unscented_transform_2d_projection(mu, cov, dummy_fisheye_camera)
print("UT 추정 2D 중심:", mu_2d)
print("UT 추정 2D 공분산:\n", cov_2d)
```

---

## 🛠️ Chapter 2: 3D 공간 상의 분석적 파티클 평가 ($\tau_{\max}$)

### 1. 요약
2D 투영 오차를 줄이기 위해 파티클 응답을 2D 평면이 아니라 **3D 광선 위에서 분석적으로 직접 평가**합니다. 광선 $\mathbf{r}(\tau) = \mathbf{o} + \tau \mathbf{d}$ 상에서 가우시안 밀도가 가장 높은 깊이 $\tau_{\max}$를 산출합니다.

### 2. 수식 및 파이썬 코드 설명

$$\tau_{\max} = \frac{(\boldsymbol{\mu} - \mathbf{o})^T \Sigma^{-1} \mathbf{d}}{\mathbf{d}^T \Sigma^{-1} \mathbf{d}}$$

$$\rho_{3D}(\tau_{\max}) = \exp\left(-\frac{1}{2} (\mathbf{o} + \tau_{\max}\mathbf{d} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{o} + \tau_{\max}\mathbf{d} - \boldsymbol{\mu})\right)$$

```python
import torch

def evaluate_3d_ray_gaussian_peak(
    ray_o: torch.Tensor,   # (3,) 광선 시작점
    ray_d: torch.Tensor,   # (3,) 광선 방향
    mu: torch.Tensor,      # (3,) 가우시안 중심
    sigma_inv: torch.Tensor# (3, 3) 역공분산
) -> torch.Tensor:
    """
    3D 광선 상의 정밀한 Peak 거리 tau_max 및 3D 밀도 평가
    """
    diff = mu - ray_o
    tau_max = torch.dot(diff @ sigma_inv, ray_d) / torch.dot(ray_d @ sigma_inv, ray_d)
    
    # tau_max 위치에서의 3D 좌표
    p_peak = ray_o + tau_max * ray_d
    
    # 3D 가우시안 밀도 값
    d_vec = p_peak - mu
    density = torch.exp(-0.5 * torch.dot(d_vec @ sigma_inv, d_vec))
    return tau_max, density

# --- 사용 예시 ---
o = torch.tensor([0.0, 0.0, 0.0])
d = torch.tensor([0.0, 0.0, 1.0])
mu_pt = torch.tensor([0.1, 0.0, 2.0])
sig_inv = torch.eye(3)
tau, dens = evaluate_3d_ray_gaussian_peak(o, d, mu_pt, sig_inv)
print(f"3D Peak 거리 tau: {tau.item():.4f}, 밀도: {dens.item():.4f}")
```

---

## 🛠️ Chapter 3: 1차/2차 광선 하이브리드 렌더링 (Hybrid Rendering)

### 1. 요약
대부분의 픽셀에 대한 1차 광선(Primary Ray)은 **UT 래스터라이저**를 통해 200+ FPS의 초고속으로 처리하고, 반사 유리나 거울 등 정반사 성분이 강한 영역의 2차 광선(Secondary Ray)만 선택적으로 레이 트레이싱 호스트로 넘깁니다.

```python
import torch

def hybrid_rendering_dispatch(
    pixel_coords: torch.Tensor,     # (N, 2) 화면 픽셀 좌표
    is_specular_mask: torch.Tensor, # (N,) 거울/반사 반사도 마스크 (Bool)
    rasterizer_fn,                  # UT 래스터라이저 콜백
    raytracer_fn                    # 3DGRT 레이트레이서 콜백
) -> torch.Tensor:
    """
    1차 광선은 UT 래스터라이저로, 2차 반사 광선은 레이트레이서로 분기 처리
    """
    rendered_image = torch.zeros((len(pixel_coords), 3))
    
    # 1. 확산 영역: UT 래스터라이저 가속
    diffuse_mask = ~is_specular_mask
    if diffuse_mask.any():
        rendered_image[diffuse_mask] = rasterizer_fn(pixel_coords[diffuse_mask])
        
    # 2. 반사/굴절 영역: 2차 광선 레이 트레이싱
    if is_specular_mask.any():
        rendered_image[is_specular_mask] = raytracer_fn(pixel_coords[is_specular_mask])
        
    return rendered_image
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. 카메라 종류별 성능 정량 비교

| 데이터셋 (Dataset) | 카메라 왜곡 조건 | 모델 (Method) | PSNR ↑ | SSIM ↑ | 렌더링 FPS | 특징 |
|---|---|---|---|---|---|---|
| **Scannet++** | 어안 렌즈 (Fisheye) | FisheyeGS | 28.15 | 0.880 | 120 FPS | 어안 전용 자코비안 유도 |
| **Scannet++** | 어안 렌즈 (Fisheye) | **3DGUT (Ours)** | **28.46** | **0.892** | **265 FPS** | **UT 래스터화로 최고속 및 고화질** |
| **Waymo** | 롤링 셔터 | 3DGS (Standard) | 29.83 | 0.891 | 180 FPS | 롤링 셔터 왜곡 오차 발생 |
| **Waymo** | 롤링 셔터 | **3DGUT (Ours)** | **30.16** | **0.902** | **210 FPS** | **시간 의존적 UT로 왜곡 완전 복원** |

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
3DGUT는 Jacobian 선형 근사를 Unscented Transform으로 대체하는 혁신적 아이디어를 통해, 복잡한 비선형 카메라 미분 유도 없이도 래스터화의 압도적 속도와 레이 트레이싱 수준의 왜곡 복원 및 2차 조명 기능을 완성한 기술입니다.

### 2. 실무적 시사점
- **자율주행 센서 융합**: 어안 렌즈, 롤링 셔터 등 실제 차재 카메라 센서 데이터를 왜곡 보정 없이 전용 투영 함수 $g$ 등록만으로 200 FPS 이상에서 직접 실시간 렌더링 가능.

### 3. 한계점 및 아쉬운 점
- **UT 평가 비용**: 3D에서 여러 Sigma point를 추가로 평가해야 하므로 순수 3DGS보다는 여전히 느림.
- **극심한 왜곡에서의 근사 오차**: 왜곡이 매우 심한 경우 투영 결과가 실제 2D 가우시안 형태에서 벗어날 수 있음.
- **겹치는 가우시안 처리 한계**: 겹치는 가우시안들을 광선 위 단일 점($\tau_{\max}$)으로만 평가하기 때문에, overlap이 많은 영역에서는 정확도가 떨어질 수 있음.

---

## 참고 자료
- [3DGUT arXiv 논문 (arXiv:2412.12507)](https://arxiv.org/abs/2412.12507)

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [3D Gaussian Ray Tracing](/posts/papers/3d-gaussian-ray-tracing/)*
