---
title: "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering"
date: 2026-04-14T16:00:00+09:00
draft: false
categories: ["Papers", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "Dynamic Scene", "Novel View Synthesis", "Real-Time Rendering"]
year: 2023
references:
  - "3d-gaussian-splatting"
  - "nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis"
---

## 💡 한 줄 요약
하나의 고정된 정규 공간(Canonical Space) 3D 가우시안 집합에 시간에 따른 3D 변형(Deformation)을 예측하는 **Gaussian Deformation Field Network**를 결합하여, 동적(Dynamic) 장면에서도 실시간(최대 82fps) 렌더링 및 프레임 수와 무관한 효율적 저장 용량을 달성했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, Xinggang Wang (Huazhong Univ. of Science and Tech., Huawei)
- **발행년도**: 2023 (arXiv:2310.08528, ICLR 2024)
- **주요 기여점**:
  1. **Canonical 3D Gaussians + Deformation Field 축조**: 프레임별로 가우시안을 새로 만드는 대신, 단일 정규 공간 3DGS $\mathcal{G}$와 시간 $t$를 입력받아 변형량 $\Delta\mathcal{G}$를 출력하는 미분 가능 변형 신경망을 연결해 저장 복잡도를 $\mathcal{O}(TN)$에서 $\mathcal{O}(N+F)$로 절감.
  2. **HexPlane 기반 시공간 구조 인코더 (Spatial-Temporal Encoder)**: 4D 시공간 $(x, y, z, t)$을 6개의 2D 특징 평면(Plane)으로 분해하여 시공간 이웃 특징을 매우 빠르게 조회하는 HexPlane 인코더 개발.
  3. **Multi-Head Deformation Decoder**: 3D 중심 이동량 $\Delta\boldsymbol{\mu}$, 회전 쿼터니언 변위 $\Delta\mathbf{q}$, 스케일 변위 $\Delta\mathbf{s}$를 각각 독립된 MLP 헤드로 예측하여 복잡한 비선형 변형을 정밀하게 복원.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **Dynamic NeRF (HyperNeRF, TiNeuVox 등)**: 연속 4D 신경 방사장을 체적 적분하여 동적 장면을 재구성했으나 렌더링 속도가 1~2 FPS로 극히 느림.
2. **DynamicGaussian**: 3DGS를 동적 장면에 도입했으나 프레임마다 별도의 가우시안 파티클을 독자 추적해 프레임 수 $T$에 비례하여 메모리가 폭발적으로 증가함.
3. **4D-GS**: 단일 Canonical 3DGS + HexPlane 4D 변형 신경망 기법으로 화질, 실시간 속도, 메모리 효율의 결합 성공.

### 기존 동적 렌더링의 한계점
- **메모리 선형 폭발**: 프레임당 가우시안을 보관하면 1,000 프레임 동영상 렌더링 시 수 GB 단위의 용량이 필요함.
- **장거리 모션 표현 한계**: 단순 광학 흐름(Optical Flow) 선형 이송은 물체의 3D 회전이나 유기적 변형을 포착하기 어려움.

---

## 📑 목차
- Chapter 1: 4D Gaussian Splatting 파이프라인
- Chapter 2: HexPlane (6-Plane) 시공간 인코딩 구조
- Chapter 3: Multi-Head Deformation Decoder 및 변형 방정식
- Chapter 4: 시공간 평활화 정규화 (Total Variation Loss)
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: 4D Gaussian Splatting 파이프라인

### 1. 요약
정적 장면을 표현하는 **Canonical 3D Gaussians** $\mathcal{G} = \{\boldsymbol{\mu}_i, \mathbf{q}_i, \mathbf{s}_i, \alpha_i, c_i\}_{i=1}^N$를 정규 공간에 정의합니다. 임의의 타임스텝 $t$가 주어지면 변형 신경망이 변위량 $\Delta\mathcal{G}_i(t) = \{\Delta\boldsymbol{\mu}_i, \Delta\mathbf{q}_i, \Delta\mathbf{s}_i\}$를 출력하며, 시간에 맞춰 변형된 가우시안 $\mathcal{G}'(t)$를 미분 가능 타일 래스터라이저로 전달해 실시간 렌더링합니다.

### 2. 수식 및 파이썬 코드 설명

#### 4D 가우시안 변형 렌더링 방정식
$$\mathcal{G}'(t) = \{ \boldsymbol{\mu}_i + \Delta\boldsymbol{\mu}_i(t), \ \mathbf{q}_i + \Delta\mathbf{q}_i(t), \ \mathbf{s}_i + \Delta\mathbf{s}_i(t), \ \alpha_i, \ c_i \}$$

$$\hat{\mathbf{I}}(t) = \text{Rasterize}\big( \mathbf{M}, \ \mathcal{G}'(t) \big)$$

- **$\mathbf{M}$**: 타임스텝 $t$에서의 카메라 외부/내부 포즈 행렬
- **$\Delta\boldsymbol{\mu}_i, \Delta\mathbf{q}_i, \Delta\mathbf{s}_i$**: 시간 $t$에 따른 $i$번째 가우시안의 위치, 회전, 크기 변형량

---

## 🛠️ Chapter 2: HexPlane 시공간 인코딩 구조

### 1. 요약
4D 공간 $(x, y, z, t)$을 3차원 그리드로 바로 저장하면 메모리가 폭발하므로, 6개의 2D 평면인 공간 평면 3개 $(xy, xz, yz)$와 시공간 평면 3개 $(xt, yt, zt)$로 분해합니다. 4D 위치 좌표를 각 2D 평면에 투영하여 얻은 특징들을 요소별 곱(Element-wise Multiplication)으로 결합합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{v}(x,y,z,t) = \prod_{p \in \{(xy),(xz),(yz),(xt),(yt),(zt)\}} \text{Interp}\Big( \mathbf{P}_p, \ \pi_p(x,y,z,t) \Big)$$

- **$\mathbf{P}_p$**: $p$번째 2D 평면 피처 그리드 (Feature Plane)
- **$\pi_p$**: 4D 좌표를 해당 2D 평면 축으로 선택 투영하는 함수
- **$\text{Interp}$**: 2D 피처 평면 상의 쌍선형 보간 (Bilinear Interpolation)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HexPlaneFeatureEncoder(nn.Module):
    """
    4D (x,y,z,t) 좌표를 6개의 2D HexPlane 피처 그리드에 투영하여 피처 벡터 추출
    """
    def __init__(self, plane_res: int = 64, feature_dim: int = 16):
        super().__init__()
        # 6개 2D 평면 피처 그리드 파라미터 (Spatial: xy, xz, yz / Temporal: xt, yt, zt)
        self.planes = nn.ParameterList([
            nn.Parameter(torch.randn(1, feature_dim, plane_res, plane_res) * 0.1)
            for _ in range(6)
        ])
        self.plane_pairs = [
            (0, 1), (0, 2), (1, 2), # xy, xz, yz
            (0, 3), (1, 3), (2, 3)  # xt, yt, zt
        ]

    def forward(self, coords_4d: torch.Tensor) -> torch.Tensor:
        """
        coords_4d Shape: (B, 4) -> (x, y, z, t) in [-1, 1]
        """
        B = coords_4d.shape[0]
        feature_product = 1.0
        
        for idx, (ax1, ax2) in enumerate(self.plane_pairs):
            # 1. 2D 좌표 선택 (B, 1, 1, 2)
            grid_coords = coords_4d[:, [ax1, ax2]].view(B, 1, 1, 2)
            
            # 2. 2D 쌍선형 보간 Sample (B, feature_dim, 1, 1)
            sampled_feat = F.grid_sample(
                self.planes[idx].expand(B, -1, -1, -1),
                grid_coords,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            ).squeeze(-1).squeeze(-1) # (B, feature_dim)
            
            # 3. Element-wise Product
            feature_product = feature_product * sampled_feat
            
        return feature_product

# --- 사용 예시 ---
coords_4d_example = torch.tensor([[0.2, -0.5, 0.1, 0.8]]) # (x, y, z, t)
encoder = HexPlaneFeatureEncoder()
print("HexPlane 추출 4D 피처 차원:", encoder(coords_4d_example).shape)
```

---

## 🛠️ Chapter 3: Multi-Head Deformation Decoder 및 변형 모델링

### 1. 요약
HexPlane에서 추출된 4D 시공간 특징 $\mathbf{v}$를 디코더 신경망에 전달하여 중심 이동, 회전 변화, 스케일 변화량을 다중 헤드(Multi-head) 구조로 독립 예측합니다.

### 2. 수식 및 파이썬 코드 설명

$$\Delta\boldsymbol{\mu} = \text{Head}_{\boldsymbol{\mu}}(\mathbf{v}), \quad \Delta\mathbf{q} = \text{Head}_{\mathbf{q}}(\mathbf{v}), \quad \Delta\mathbf{s} = \text{Head}_{\mathbf{s}}(\mathbf{v})$$

```python
import torch
import torch.nn as nn

class MultiHeadDeformationDecoder(nn.Module):
    """
    HexPlane 특징을 받아 가우시안 위치, 회전, 스케일 변위량을 예측하는 Multi-Head MLP
    """
    def __init__(self, in_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 3개 변형 예측 독립 헤드
        self.head_pos = nn.Linear(hidden_dim, 3) # delta_mu (dx, dy, dz)
        self.head_rot = nn.Linear(hidden_dim, 4) # delta_q (dw, dx, dy, dz)
        self.head_scale = nn.Linear(hidden_dim, 3) # delta_s (dsx, dsy, dsz)

    def forward(self, feature_v: torch.Tensor) -> tuple:
        feat = self.backbone(feature_v)
        
        delta_pos = self.head_pos(feat)
        delta_rot = self.head_rot(feat)
        delta_scale = self.head_scale(feat)
        
        return delta_pos, delta_rot, delta_scale

# --- 사용 예시 ---
v_feat = torch.randn(1, 16)
decoder = MultiHeadDeformationDecoder()
d_p, d_r, d_s = decoder(v_feat)
print("예측된 위치 변위 d_pos:", d_p)
print("예측된 회전 변위 d_rot:", d_r)
print("예측된 스케일 변위 d_scale:", d_s)
```

---

## 🛠️ Chapter 4: 시공간 Total Variation 정규화 손실 (TV Loss)

### 1. 요약
HexPlane 피처 평면의 인접 그리드 간 급격한 불연속성을 억제하고 시공간 모션의 부드러움(Smoothness)을 강제하기 위해 Total Variation 손실을 추가합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{L}_{TV}(\mathbf{P}_p) = \frac{1}{H \cdot W} \sum_{i, j} \left( \|\mathbf{P}_p[i+1, j] - \mathbf{P}_p[i, j]\|_1 + \|\mathbf{P}_p[i, j+1] - \mathbf{P}_p[i, j]\|_1 \right)$$

```python
import torch

def compute_hexplane_tv_loss(planes: list) -> torch.Tensor:
    """
    6개 HexPlane 피처 그리드에 대한 Total Variation 손실 계산
    """
    total_tv = 0.0
    for plane in planes:
        # plane Shape: (1, C, H, W)
        dh = torch.abs(plane[:, :, 1:, :] - plane[:, :, :-1, :])
        dw = torch.abs(plane[:, :, :, 1:] - plane[:, :, :, :-1])
        total_tv += dh.mean() + dw.mean()
    return total_tv

# --- 사용 예시 ---
dummy_planes = [torch.randn(1, 16, 64, 64) for _ in range(6)]
print("HexPlane TV Loss:", compute_hexplane_tv_loss(dummy_planes).item())
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. 대표 동적 벤치마크 성능 비교 (D-NeRF 데이터셋)

| 모델 (Method) | PSNR ↑ | SSIM ↑ | 렌더링 속도 (FPS) ↑ | 학습 시간 ↓ | 메모리/용량 |
|---|---|---|---|---|---|
| **TiNeuVox-B** | 32.67 | 0.971 | 1.5 FPS | 28 분 | 48 MB |
| **V4D** | 33.72 | 0.980 | 4.4 FPS | 6.9 시간 | - |
| **DynamicGaussian** | 31.20 | 0.960 | 60 FPS | 15 분 | 380 MB (선형 증가) |
| **4D-GS (Ours)** | **34.05** | **0.982** | **82 FPS** | **8 분** | **18 MB (최소 저장)** |

- **결과**: 기존 동적 NeRF 대비 렌더링 속도 **80배 가속 (82 FPS)** 및 PSNR +1.4 dB 향상. 프레임별 복제 방식 대비 용량을 **18 MB** 수준으로 압축.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
4D-GS는 정적 3DGS를 Canonical 공간에 배치하고 HexPlane 시공간 인코더 기반 변형 신경망을 유기적으로 결합하여, 동적 장면 렌더링에서 **최고 화질, 실시간 속도(80+ FPS), 최소 저장 용량**을 동시에 달성한 대표적인 4D 신경 표현 논문입니다.

### 2. 실무적 시사점
- **자율주행 시뮬레이터**: 보행자 및 차량 등 시간에 따라 움직이는 동적 객체들을 4D-GS 변형 필드로 표현해 클로즈드 루프 시뮬레이션(HUGSIM 등) 백본으로 적용 가능.

---

## 참고 자료
- [4D-GS 공식 프로젝트 페이지](https://fudan-zvg.github.io/4d-gaussian-splatting/)
- [ICLR 2024 논문 (arXiv:2310.08528)](https://arxiv.org/abs/2310.08528)

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [Street Gaussians](/posts/papers/street-gaussians-modeling-dynamic-urban-scenes/), [DrivingGaussian](/posts/papers/driving-gaussian-composite-gaussian-splatting/)*
