---
title: "3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes"
date: 2026-04-10T09:00:00+09:00
draft: false
categories: ["Papers", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "Ray Tracing", "Novel View Synthesis", "Neural Rendering"]
year: 2024
references:
  - "3d-gaussian-splatting"
---

## 💡 한 줄 요약
각 3D 가우시안 파티클을 바운딩 볼륨 계층 구조(BVH)에 삽입 가능한 프록시 기하(Stretched Icosahedron)로 감싸 GPU 레이 트레이서(OptiX)로 추적함으로써, 기존 래스터화 방식에서는 불가능했던 반사·굴절·그림자·어안/롤링 셔터 카메라 등 이차 조명 효과 및 복합 레이 추적을 실시간(실용적 속도)에 가능케 했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp, Zan Gojcic (NVIDIA)
- **발행년도**: 2024 (arXiv:2407.07090, SIGGRAPH Asia 2024)
- **주요 기여점**:
  1. **하드웨어 가속 파티클 레이 트레이싱**: 각 가우시안을 3D 공분산에 맞춰 변형된 정이십면체(Stretched Icosahedron) 프록시로 감싸 BVH에 구축하고, NVIDIA OptiX를 활용해 GPU 하드웨어 가속 광선-파티클 교차를 구현.
  2. **동적 k-Hit 추적 및 미분 가능 최적화**: 한 번에 정렬된 k개의 교차점을 순차 추적하여 투과율을 계산하는 forward pass와, 동일한 광선을 재추적하며 역전파 그래디언트를 정밀하게 누적하는 backward pass 파이프라인 개발.
  3. **일반화 파티클 커널 (Generalized Gaussian Kernels)**: 커널 차수 $n=2$인 일반화 가우시안 및 코사인 파동 변조 커널을 도입하여 파티클 간 불필요한 교차 횟수를 획기적으로 줄이고 렌더링 속도를 2배 가량 향상.
  4. **비선형 카메라 및 2차 조명 지원**: 어안 렌즈(Fisheye), 롤링 셔터(Rolling Shutter), 피사계 심도(DoF), 거울 반사/굴절 및 메쉬 결합 렌더링 지원.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **NeRF (Neural Radiance Fields)**: 연속적 좌표계 기반 체적 렌더링으로 우수한 화질을 제공하나, 광선 추적 시 밀도 평가 연산이 매우 느림.
2. **3D Gaussian Splatting (3DGS)**: 이방성 가우시안과 타일 기반 래스터라이저로 실시간 렌더링을 가능케 했으나, 핀홀(Pinhole) 카메라의 1차 광선(Primary Ray)으로 렌더링 구조가 제한됨.
3. **볼류메트릭 레이 트레이싱 (Volumetric Ray Tracing)**: Knoll et al. 등의 이전 연구는 소규모 파티클이나 저해상도 렌더링에 한정됨.
4. **3D Gaussian Ray Tracing**: 수백만 개의 3DGS 파티클 장면 전체를 GPU RT 코어(OptiX BVH)로 실시간에 추적할 수 있도록 기하학적 캡슐화 및 광선 추적 알고리즘을 완성.

### 기존 래스터화(Splatting)의 한계점
- **비선형 카메라 모델 지원 불가**: 래스터화는 핀홀 투영 선형 자코비안 근사에 의존하므로, 자율주행의 어안 렌즈나 롤링 셔터 카메라에서 원근 투영 오차가 급증함.
- **2차 광선(Secondary Rays) 시뮬레이션 불가**: 빛이 표면에 부딪혀 튕겨 나가는 반사, 굴절, 굴곡 광선, 그림자 추적이 구조적으로 불가능함.
- **확률적 광선 샘플링의 제한**: 타일 기반 스플래팅은 픽셀 전체를 격자 형태로 순회해야 하므로 훈련 중 임의 픽셀/광선의 독립 샘플링이 어려움.

---

## 📑 목차
- Chapter 1: 3D 가우시안의 캡슐화 프록시 및 광선 교차
- Chapter 2: 광선 상의 파티클 최대 응답점 분석적 계산
- Chapter 3: 동적 k-Hit 레이 트레이싱 및 미분 가능 역전파
- Chapter 4: 일반화 가우시안 커널 (Generalized Exponent Kernel)
- Chapter 5: 복합 카메라 모델 (Fisheye & Rolling Shutter)
- Chapter 6: 2차 조명 효과 (반사, 굴절, 그림자)
- Chapter 7: 주요 실험 및 결과
- Chapter 8: 결론 및 시사점

---

## 🛠️ Chapter 1: 3D 가우시안 캡슐화 프록시 (Stretched Polyhedron Proxy)

### 1. 요약
NVIDIA OptiX와 같은 GPU 레이 트레이싱 엔진은 삼각 메쉬(Triangle Mesh)나 바운딩 박스(AABB) 형태의 BVH 입력을 필요로 합니다. 3D 가우시안은 무한히 퍼지는 밀도 함수이므로, 저자들은 밀도가 유의미한 수준(예: $3\sigma$) 이상인 영역을 감싸는 **변형된 정이십면체(Stretched Icosahedron)**를 프록시 기하(Proxy Geometry)로 생성해 BVH에 등록합니다.

### 2. 핵심 개념
- **Stretched Polyhedron**: 정이십면체의 각 정점에 가우시안의 3D 회전 행렬 $\mathbf{R}$과 스케일 행렬 $\mathbf{S}$를 곱하여 가우시안 타원체 표면에 밀착시킨 프리미티브.
- **적응적 클램핑 (Adaptive Clamping)**: 불투명도가 높은 가우시안은 크게, 불투명도가 낮은 가우시안은 범위를 좁게 클램핑하여 레이 교차 테스트 횟수를 최적화.

---

## 🛠️ Chapter 2: 광선 상의 파티클 최대 응답점 분석적 계산

### 1. 요약
광선 $\mathbf{r}(\tau) = \mathbf{o} + \tau \mathbf{d}$가 가우시안 바운딩 프록시 내부를 통과할 때, 광선 위의 수많은 지점 중 파티클 밀도가 **최대가 되는 파라미터 $\tau_{\max}$**를 해석적으로 구합니다.

### 2. 수식 및 파이썬 코드 설명

#### (1) 광선 상의 최대 밀도 지점 $\tau_{\max}$
광선 원점 $\mathbf{o}$, 방향 $\mathbf{d}$일 때 마할라노비스 거리(Mahalanobis Distance)를 최소화하는 광선 상의 거리 $\tau_{\max}$는 다음과 같이 닫힌 형태(Closed-form)로 유도됩니다.

$$\tau_{\max} = \underset{\tau}{\text{argmax}} \ \rho(\mathbf{o} + \tau\mathbf{d}) = \frac{(\boldsymbol{\mu} - \mathbf{o})^T \Sigma^{-1} \mathbf{d}}{\mathbf{d}^T \Sigma^{-1} \mathbf{d}}$$

- **$\mathbf{o}$**: 광선 원점 (Ray Origin)
- **$\mathbf{d}$**: 광선 방향 단위 벡터 (Ray Direction)
- **$\boldsymbol{\mu}$**: 가우시안 파티클의 중심 위치
- **$\Sigma^{-1}$**: 3D 가우시안 공분산 행렬의 역행렬

```python
import torch

def compute_ray_gaussian_max_tau(
    ray_origin: torch.Tensor,     # (3,) 광선 시작점 o
    ray_dir: torch.Tensor,        # (3,) 광선 방향 d
    mu: torch.Tensor,             # (3,) 가우시안 중심 mu
    sigma_inv: torch.Tensor       # (3, 3) 가우시안 공분산 역행렬 Sigma^{-1}
) -> torch.Tensor:
    """
    광선 r(tau) = o + tau * d 위에서 가우시안 밀도가 최대가 되는 tau_max 계산
    tau_max = ((mu - o)^T @ Sigma^{-1} @ d) / (d^T @ Sigma^{-1} @ d)
    """
    diff = mu - ray_origin         # (3,) mu - o Vector
    
    # 분자: (mu - o)^T @ Sigma^{-1} @ d
    numerator = torch.dot(diff @ sigma_inv, ray_dir)
    
    # 분모: d^T @ Sigma^{-1} @ d
    denominator = torch.dot(ray_dir @ sigma_inv, ray_dir)
    
    # 수치 안정성 처리
    denominator = torch.clamp(denominator, min=1e-8)
    
    tau_max = numerator / denominator
    return tau_max

# --- 사용 예시 ---
o = torch.tensor([0.0, 0.0, 0.0])
d = torch.tensor([0.0, 0.0, 1.0])  # +Z 방향으로 쏘는 광선
mu = torch.tensor([0.2, -0.1, 4.0])
sigma_inv = torch.eye(3) * 2.0
tau_peak = compute_ray_gaussian_max_tau(o, d, mu, sigma_inv)
print("광선 상 최대 밀도 위치 tau_max:", tau_peak.item())
```

---

## 🛠️ Chapter 3: 동적 k-Hit 레이 트레이싱 및 미분 가능 최적화

### 1. 요약
광선이 장면을 지날 때 수십~수백 개의 가우시안 프록시와 교차할 수 있습니다. 메모리 폭발을 막기 위해 렌더러는 한 번의 쿼리에서 깊이 순으로 **가장 가까운 $k$개의 교차점(Hits)**만 동적으로 추적(Dynamic $k$-hits)하여 알파 합성을 진행하고, 투과율 $T$가 0에 도달할 때까지 연속적으로 추적을 재개합니다.

### 2. 수식 및 파이썬 코드 설명

#### 광선 단위 알파 체적 합성
광선 $\mathbf{r}$을 따라 탐지된 $N$개의 파티클 교차점들의 색상 $c_i$와 실효 불투명도 $\alpha_i$를 정렬하여 합성합니다.

$$L(\mathbf{o}, \mathbf{d}) = \sum_{i=1}^{N} c_i(\mathbf{d}) \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

$$\alpha_i = 1 - \exp\left(-\sigma_i \rho_i(\mathbf{r}(\tau_{i,\max})) \cdot \Delta\tau_i\right)$$

- **$\sigma_i$**: 파티클의 기본 밀도 스케일
- **$\rho_i$**: $\tau_{i,\max}$ 위치에서의 가우시안 밀도 평가값
- **$\Delta\tau_i$**: 파티클 바운딩 프록시를 광선이 통과하는 유효 구간 길이

```python
import torch

def trace_k_hits_compositing(
    hit_colors: torch.Tensor,   # (N, 3) 정렬된 Hit 파티클들의 색상
    hit_densities: torch.Tensor,# (N,) 각 Hit 파티클의 평가된 밀도 rho_i
    hit_deltas: torch.Tensor,   # (N,) 광선 통과 구간 길이 delta_tau_i
    base_opacities: torch.Tensor# (N,) 파티클 기본 불투명도 sigma_i
) -> torch.Tensor:
    """
    광선 레이 트레이싱 결과 k-hits 수집 후 Front-to-Back 알파 합성
    """
    accumulated_color = torch.zeros(3)
    transmittance = 1.0
    
    for i in range(len(hit_colors)):
        # 1. 구간 실효 불투명도 alpha_i 계산
        alpha_i = 1.0 - torch.exp(-base_opacities[i] * hit_densities[i] * hit_deltas[i])
        alpha_i = torch.clamp(alpha_i, max=0.99)
        
        # 2. 색상 누적
        weight = transmittance * alpha_i
        accumulated_color += weight * hit_colors[i]
        
        # 3. 투과율 갱신
        transmittance *= (1.0 - alpha_i)
        
        # Early termination
        if transmittance < 1e-4:
            break
            
    return accumulated_color

# --- 사용 예시 ---
colors = torch.tensor([[1.0, 0.2, 0.2], [0.2, 1.0, 0.2]])
densities = torch.tensor([0.8, 0.9])
deltas = torch.tensor([0.1, 0.15])
opacities = torch.tensor([2.0, 3.0])
print("광선 최종 추적 색상:", trace_k_hits_compositing(colors, densities, deltas, opacities))
```

---

## 🛠️ Chapter 4: 일반화 가우시안 커널 (Generalized Exponent Kernel)

### 1. 요약
표준 3D 가우시안($n=1$)은 꼬리(Tail) 부분이 길게 늘어져 있어 수많은 불필요한 광선 교차(Hit)를 발생시킵니다. 저자들은 커널 지수 $n=2$인 **일반화 가우시안(Generalized Gaussian)**을 도입해 경계를 더욱 선명하게 다듬음으로써 교차 연산 횟수를 절반으로 단축시킵니다.

### 2. 수식 및 파이썬 코드 설명

$$\tilde{\rho}_n(\mathbf{x}) = \sigma \exp\left( - \left( (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)^n \right)$$

- **$n=1$**: 표준 3D 가우시안 커널 (부드러운 전이)
- **$n=2$**: 일반화 가우시안 커널 (중심부는 평평하고 경계가 급격히 차단되어 교차 횟수 감소)

```python
import torch

def compute_generalized_gaussian_kernel(
    x: torch.Tensor,           # (N, 3) 평가 좌표
    mu: torch.Tensor,          # (3,) 중심
    sigma_inv: torch.Tensor,   # (3, 3) 역공분산
    n_order: float = 2.0       # 커널 차수 n (기본 2.0)
) -> torch.Tensor:
    """
    일반화 가우시안 커널 rho_n(x) = exp(- ( (x - mu)^T @ Sigma^{-1} @ (x - mu) )^n ) 계산
    """
    diff = x - mu.unsqueeze(0)
    quad_form = torch.sum((diff @ sigma_inv) * diff, dim=-1) # (N,)
    
    # n_order 지수 적용
    powered_quad = torch.pow(torch.clamp(quad_form, min=1e-8), n_order)
    
    density = torch.exp(-powered_quad)
    return density

# --- 사용 예시 ---
pts = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
mu_val = torch.tensor([0.0, 0.0, 0.0])
sig_inv = torch.eye(3)
print("n=1 (표준 가우시안) 밀도:", compute_generalized_gaussian_kernel(pts, mu_val, sig_inv, n_order=1.0))
print("n=2 (일반화 가우시안) 밀도:", compute_generalized_gaussian_kernel(pts, mu_val, sig_inv, n_order=2.0))
```

---

## 🛠️ Chapter 5: 복합 카메라 모델 (Fisheye & Rolling Shutter)

### 1. 요약
광선 추적 기반 렌더러의 가장 큰 장점은 각 픽셀마다 **임의의 시점과 방향을 갖는 3D 광선을 독립적으로 생성**할 수 있다는 점입니다. 이를 통해 선형 핀홀 투영법으로 표현할 수 없는 광각 어안 렌즈(Fisheye)와 롤링 셔터(Rolling Shutter) 왜곡을 수학적으로 정확하게 시뮬레이션할 수 있습니다.

### 2. 수식 및 파이썬 코드 설명

#### (1) 어안 렌즈(Equidistant Fisheye) 광선 생성
화면 중심으로부터의 거리 $r = \sqrt{u^2 + v^2}$와 초점거리 $f$에 대해 광선 방사각 $\theta = r / f$를 직접 구해 3D 방향 벡터를 생성합니다.

$$\mathbf{d}_{\text{fisheye}} = \begin{bmatrix} \frac{u}{r} \sin\theta \\ \frac{v}{r} \sin\theta \\ \cos\theta \end{bmatrix}$$

```python
import torch

def generate_fisheye_ray_direction(
    u: float, v: float, focal_length: float
) -> torch.Tensor:
    """
    어안 렌즈(Equidistant Fisheye) 픽셀 좌표 (u, v)로부터 3D 광선 방향 d 생성
    """
    r = torch.sqrt(torch.tensor(u**2 + v**2))
    if r < 1e-6:
        return torch.tensor([0.0, 0.0, 1.0])
        
    theta = r / focal_length
    
    dx = (u / r) * torch.sin(theta)
    dy = (v / r) * torch.sin(theta)
    dz = torch.cos(theta)
    
    # torch.stack 사용: torch.tensor([...])로 감싸면 u, v에 대한 autograd 그래프가 끊김
    ray_dir = torch.stack([dx, dy, dz])
    return ray_dir / torch.norm(ray_dir)

# --- 사용 예시 ---
print("어안 렌즈 외곽 픽셀 광선 방향:", generate_fisheye_ray_direction(300.0, 400.0, 500.0))
```

---

## 🛠️ Chapter 6: 2차 조명 효과 (반사, 굴절, 그림자)

### 1. 요약
1차 카메라 광선이 파티클이나 혼합 메쉬(Mesh) 표면에 부딪혔을 때, 해당 교차점에서 반사/굴절 광선을 새로 생성(Secondary Ray Spawning)하여 레이 트레이서에 다시 반사 광선을 쏘아 보냄으로써 거울 반사, 반투명 굴절, 광원 방향 그림자(Shadow Ray)를 완벽히 구현합니다.

### 2. 수식 및 파이썬 코드 설명

#### 2차 반사 광선 (Reflection Ray)
입사 광선 방향 $\mathbf{d}$와 표면 법선 벡터 $\mathbf{n}$에 대해 반사 광선 방향 $\mathbf{d}_{\text{refl}}$은 반사 법칙을 따릅니다.

$$\mathbf{d}_{\text{refl}} = \mathbf{d} - 2(\mathbf{d} \cdot \mathbf{n})\mathbf{n}$$

```python
import torch

def compute_reflection_ray(
    ray_dir: torch.Tensor,    # (3,) 입사 광선 방향 d
    normal: torch.Tensor      # (3,) 표면 법선 벡터 n (정규화됨)
) -> torch.Tensor:
    """
    표면 충돌 시 2차 반사 광선 방향 d_refl 계산
    """
    dot_prod = torch.dot(ray_dir, normal)
    refl_dir = ray_dir - 2.0 * dot_prod * normal
    return refl_dir / torch.norm(refl_dir)

# --- 사용 예시 ---
d_in = torch.tensor([1.0, -1.0, 0.0]) / torch.sqrt(torch.tensor(2.0))
n_surf = torch.tensor([0.0, 1.0, 0.0])  # 바닥면 법선
print("2차 반사 광선 방향:", compute_reflection_ray(d_in, n_surf))
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. 벤치마크 별 성능 비교

| 데이터셋 / 조건 | 모델 (Method) | PSNR ↑ | 렌더링 FPS | 특징 |
|---|---|---|---|---|
| **MipNeRF360 (야외)** | 3DGS (Splatting) | 28.69 | **387 FPS** | 1차 핀홀 광선 전용 |
| **MipNeRF360 (야외)** | **3DGS RT (Ours)** | **28.71** | **363 FPS** | 하드웨어 RT 사용, 2차 광선 가능 |
| **Waymo (자율주행)** | 3DGS (Splatting) | 29.83 | - | 어안 렌즈 투영 오차 발생 |
| **Waymo (자율주행)** | **3DGS RT (Ours)** | **29.99** | - | **어안/롤링셔터 정밀 복원** |
| **Tanks & Temples** | 3DGS RT ($n=1$) | - | 143 FPS | 표준 가우시안 커널 |
| **Tanks & Temples** | **3DGS RT ($n=2$)** | **화질 유지** | **277 FPS** | **일반화 커널로 속도 2배 가속** |

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
3D Gaussian Ray Tracing은 가우시안 파티클을 변형 정이십면체 프록시로 감싸 BVH에 구축하고 GPU RT 코어(OptiX)로 추적함으로써, **3DGS의 속도감과 파티클 표현력**을 유지하면서도 **레이 트레이싱의 범용성(반사, 굴절, 비선형 카메라)**을 완전히 접목한 연구입니다.

### 2. 실무적 시사점
- **자율주행 및 로봇공학**: 카메라 센서의 어안 렌즈 왜곡 및 롤링 셔터 현상을 왜곡 보정(Undistortion) 전처리 없이 raw 데이터 그대로 고품질 재구성에 활용 가능.
- **VFX 및 3D 그래픽스**: 3DGS 표현과 고전적인 메시 물체(가구, 자동차 등)를 하나의 BVH 공간에 통합해 거울 반사 및 광체 효과 렌더링 가능.

### 3. 한계점 및 아쉬운 점
- **학습 중 BVH 재구성 오버헤드**: 파티클 위치가 매 step 최적화되므로 매 반복마다 BVH를 새로 빌드해야 하여 학습 속도가 래스터화 대비 2~3배 느림.
- **메모리 전송 비용**: 광선당 $k$-hits 수집을 위한 GPU 전역 메모리 접근 비용 존재.

---

## 참고 자료
- [공식 프로젝트 페이지](https://nicolas-ml.github.io/3d-gaussian-ray-tracing/)
- [SIGGRAPH Asia 2024 논문 (arXiv)](https://arxiv.org/abs/2407.07090)

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [3DGUT](/posts/papers/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting/), [Street Gaussians](/posts/papers/street-gaussians-modeling-dynamic-urban-scenes/)*
