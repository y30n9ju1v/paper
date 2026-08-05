---
title: "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers"
date: 2026-04-14T00:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving"]
tags: ["Autonomous Driving", "BEV", "Transformer", "3D Object Detection", "Multi-Camera"]
year: 2022
references:
  - "attention-is-all-you-need"
  - "detr3d-3d-object-detection-multi-view-images"
  - "lift-splat-shoot"
  - "nuscenes-multimodal-dataset-autonomous-driving"
---

## 💡 한 줄 요약
BEVFormer는 시공간 트랜스포머(Spatial-Temporal Transformer) 구조를 이용해 명시적 깊이 추정 없이 다중 카메라 2D 이미지에서 BEV(Bird's-Eye-View) 3D 표현을 직접 구축하고, 이전 프레임 BEV의 ego-motion 정렬 및 재귀적 융합을 통해 nuScenes 테스트셋 56.9% NDS로 LiDAR급 3D 객체 탐지 성능을 달성했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, Jifeng Dai (Nanjing Univ., Shanghai AI Lab, HKU)
- **발행년도**: 2022 (arXiv:2203.17270, ECCV 2022)
- **주요 기여점**:
  1. **Spatial Cross-Attention (SCA)**: 깊이 예측 오차를 피하기 위해 각 BEV 쿼리 픽셀에서 $N_z$개 높이 방향의 3D Pillar Anchor 점을 샘플링한 후, 2D 카메라 평면으로 투영하여 Deformable Cross-Attention으로 다중 카메라 특징을 통합.
  2. **Temporal Self-Attention (TSA)**: 과거 프레임의 BEV 특징을 자차 이동(Ego-motion) 행렬로 보정·정렬(Warping)한 후 현재 BEV 쿼리와 융합하여 속도 추정 및 사물 가림(Occlusion) 해결.
  3. **통합 멀티태스크 3D 헤드**: 하나의 BEV 인코더 출력으로 3D 객체 탐지(Bounding Box)와 HD 맵 분할(Map Segmentation)을 정밀 수행.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **DETR3D**: 3D Object Query를 이미지에 역투영하여 특징을 획득하나 BEV 평면 표현이 불완전함.
2. **LSS (Lift-Splat-Shoot)**: 깊이 분포를 추정하여 3D Frustum으로 사영하나, 깊이 예측의 오차가 BEV 특징 왜곡으로 누적됨.
3. **BEVFormer**: 명시적 깊이 추정 단계를 제거하고, 3D Pillar Anchor 샘플링과 시공간 Transformer 어텐션으로 깊이 오차 및 연속 프레임 정보 누락 문제를 완벽히 해결.

---

## 📑 목차
- Chapter 1: BEV Query 및 3D Pillar 좌표계 정의
- Chapter 2: Spatial Cross-Attention (SCA) 및 카메라 역투영 수식
- Chapter 3: Temporal Self-Attention (TSA) 및 Ego-Motion 보정 정렬
- Chapter 4: 2D Deformable Attention 수식 및 구현
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: BEV Query 및 3D Pillar 좌표계 정의

### 1. 요약
자차(Ego) 기준 2D BEV 그리드를 $H \times W$ 격자로 분할하고, 각 셀마다 학습 가능한 쿼리 $Q_{x, y} \in \mathbb{R}^C$를 배치합니다. 2D 쿼리 셀 $(x, y)$는 높이 축 방향 $z \in [z_{\min}, z_{\max}]$로 확장되어 3D Pillar 형태의 앵커 점들을 형성합니다.

---

## 🛠️ Chapter 2: Spatial Cross-Attention (SCA)

### 1. 요약
BEV 쿼리 $Q_p$에 대해 $N_z$개의 높이 앵커 점을 3D 상에서 정의하고, 이를 6개 카메라의 2D 시야각으로 투영합니다. 실제로 투영된 히트 카메라 뷰 $\mathcal{V}_{\text{hit}}$의 이미지 특징 $F_t^i$에 대해 Deformable Attention을 수행합니다.

### 2. 수식 및 파이썬 코드 설명

#### 3D Pillar $\to$ 2D 카메라 평면 투영 공식
$$z_{i,j} \begin{bmatrix} u_{i,j} \\ v_{i,j} \\ 1 \end{bmatrix} = \mathbf{P}_i \begin{bmatrix} x' \\ y' \\ z'_j \\ 1 \end{bmatrix} = \mathbf{K}_i \left( \mathbf{R}_i \begin{bmatrix} x' \\ y' \\ z'_j \end{bmatrix} + \mathbf{t}_i \right)$$

$$\text{SCA}(Q_p, F_t) = \frac{1}{|\mathcal{V}_{\text{hit}}|} \sum_{i \in \mathcal{V}_{\text{hit}}} \sum_{j=1}^{N_z} \text{DeformAttn}\Big( Q_p, \ \mathcal{P}(p, i, j), \ F_t^i \Big)$$

- **$\mathcal{P}(p, i, j) = (u_{i,j}, v_{i,j})$**: BEV 셀 $p=(x, y)$의 $j$번째 높이 앵커 점이 $i$번째 카메라에 투영된 2D 픽셀 좌표
- **$\mathbf{P}_i$**: $i$번째 카메라의 $3 \times 4$ 투영 행렬

```python
import torch

def project_bev_pillar_to_camera(
    bev_grid_coords: torch.Tensor, # (H, W, 2) BEV 2D 그리드 좌표 (x, y)
    anchor_heights: torch.Tensor,  # (Nz,) 높이 앵커 값들 (예: -5m ~ 3m 사이 4개 점)
    P_cam: torch.Tensor,           # (3, 4) 3D World to 2D Image 투영 행렬
    img_shape: tuple               # (H_img, W_img)
) -> tuple:
    """
    BEV 쿼리 픽셀의 3D Pillar 높이 앵커점들을 2D 카메라 픽셀 좌표로 투영
    """
    H, W, _ = bev_grid_coords.shape
    Nz = len(anchor_heights)
    H_img, W_img = img_shape
    
    # 1. (H, W, Nz, 3) 3D Pillar 좌표 생성 (x, y, z)
    bev_x = bev_grid_coords[..., 0].unsqueeze(-1).repeat(1, 1, Nz)
    bev_y = bev_grid_coords[..., 1].unsqueeze(-1).repeat(1, 1, Nz)
    bev_z = anchor_heights.view(1, 1, Nz).repeat(H, W, 1)
    
    pts_3d_homo = torch.stack([bev_x, bev_y, bev_z, torch.ones_like(bev_x)], dim=-1) # (H, W, Nz, 4)
    
    # 2. 2D 카메라 투영 p_2d = P_cam @ p_3d
    pts_2d_homo = torch.matmul(pts_3d_homo, P_cam.T) # (H, W, Nz, 3)
    
    depths = pts_2d_homo[..., 2]
    u = pts_2d_homo[..., 0] / (depths + 1e-6)
    v = pts_2d_homo[..., 1] / (depths + 1e-6)
    
    # 3. Valid Mask (전방 depth > 0 및 이미지 픽셀 내 유효성)
    valid_mask = (depths > 0.1) & (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
    
    # [0, 1] 범위로 정규화된 2D Reference Coordinates
    norm_u = u / W_img
    norm_v = v / H_img
    
    return torch.stack([norm_u, norm_v], dim=-1), valid_mask

# --- 사용 예시 ---
grid = torch.meshgrid(torch.linspace(-50, 50, 10), torch.linspace(-50, 50, 10), indexing='ij')
grid_coords = torch.stack(grid, dim=-1) # (10, 10, 2)
heights = torch.tensor([-2.0, 0.0, 2.0])
P_dummy = torch.eye(4)[:3, :]
uv_norm, mask = project_bev_pillar_to_camera(grid_coords, heights, P_dummy, (480, 640))
print("투영된 정규화 2D 좌표 Shape:", uv_norm.shape)
```

---

## 🛠️ Chapter 3: Temporal Self-Attention (TSA) 및 Ego-Motion 정렬

### 1. 요약
과거 프레임의 BEV 특징 $B_{t-1}$을 자차 이동 변환 행렬 $\mathbf{T}_{t-1 \to t}$를 사용하여 현재 프레임 자차 좌표계로 워핑(Warping)한 후, 현재 BEV 쿼리 $Q$와 Deformable Self-Attention으로 합성하여 속도 추정을 정밀화합니다.

### 2. 수식 및 파이썬 코드 설명

$$p_{t-1} = \mathbf{T}_{t-1 \to t}^{-1} \cdot p_t$$

$$B'_{t-1} = \text{GridSample}\Big( B_{t-1}, \ p_{t-1} \Big)$$

$$\text{TSA}(Q_p, \{Q, B'_{t-1}\}) = \sum_{V \in \{Q, B'_{t-1}\}} \text{DeformAttn}\Big( Q_p, p, V \Big)$$

```python
import torch
import torch.nn.functional as F

def align_previous_bev_feature(
    prev_bev: torch.Tensor,         # (B, C, H, W) 이전 프레임 BEV 특징
    T_prev2curr: torch.Tensor,      # (B, 4, 4) 이전->현재 자차 Ego-Motion 변환 행렬
    grid_resolution: float = 0.5    # 그리드 1셀당 실제 거리 (m)
) -> torch.Tensor:
    """
    Ego-Motion 보정을 통한 과거 BEV 특징맵의 현재 좌표계 공간 워핑(Warping)
    """
    B, C, H, W = prev_bev.shape
    
    # 1. 현재 BEV 그리드 2D 좌표 생성
    xs = torch.linspace(-W/2 * grid_resolution, W/2 * grid_resolution, W)
    ys = torch.linspace(-H/2 * grid_resolution, H/2 * grid_resolution, H)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    
    # Homo 3D 좌표 (X, Y, 0, 1)
    pts_curr = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x), torch.ones_like(grid_x)], dim=-1) # (H, W, 4)
    pts_curr = pts_curr.unsqueeze(0).repeat(B, 1, 1, 1) # (B, H, W, 4)
    
    # 2. 역변환으로 이전 좌표계에서의 위치 계산: p_prev = T_{curr->prev} @ p_curr
    T_curr2prev = torch.linalg.inv(T_prev2curr)
    pts_prev = torch.matmul(pts_curr, T_curr2prev.transpose(-2, -1)) # (B, H, W, 4)
    
    # 3. [-1, 1] 범위로 Grid Sample 정규화
    norm_x = (pts_prev[..., 0] / (W/2 * grid_resolution))
    norm_y = (pts_prev[..., 1] / (H/2 * grid_resolution))
    grid_sample_norm = torch.stack([norm_x, norm_y], dim=-1) # (B, H, W, 2)
    
    # 4. 이전 BEV 특징 워핑 샘플링
    aligned_prev_bev = F.grid_sample(prev_bev, grid_sample_norm, mode='bilinear', align_corners=True)
    return aligned_prev_bev

# --- 사용 예시 ---
prev_b = torch.randn(1, 256, 50, 50)
T_dummy = torch.eye(4)
T_dummy[0, 3] = 1.5 # X 방향 1.5m 차량 이동
print("Ego-Motion 정렬된 과거 BEV Shape:", align_previous_bev_feature(prev_b, T_dummy.unsqueeze(0)).shape)
```

---

## 🛠️ Chapter 4: Deformable Attention 연산 구조

### 1. 요약
글로벌 Attention의 연산량 폭발을 방지하기 위해 쿼리 지점 주변의 $K$개 오프셋 위치만 어텐션하는 Deformable Attention을 사용합니다.

### 2. 수식 및 파이썬 코드 설명

$$\text{DeformAttn}(q, p, x) = \sum_{m=1}^{M} \mathbf{W}_m \sum_{k=1}^{K} A_{m,k} \cdot \mathbf{W}'_m x(p + \Delta p_{m,k})$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableAttention2D(nn.Module):
    """
    소수 K개 오프셋 지점만 어텐션하는 고속 2D Deformable Attention
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_keys: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.num_keys = num_keys
        self.scale = (embed_dim // num_heads) ** -0.5
        
        # 오프셋 delta_p 및 가중치 A 생성 헤드
        self.offset_head = nn.Linear(embed_dim, num_heads * num_keys * 2)
        self.weight_head = nn.Linear(embed_dim, num_heads * num_keys)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, value_feat: torch.Tensor) -> torch.Tensor:
        """
        query: (B, N_q, C)
        value_feat: (B, C, H, W)
        """
        B, N_q, C = query.shape
        head_dim = C // self.num_heads
        offsets = self.offset_head(query).view(B, N_q, self.num_heads, self.num_keys, 2)
        weights = torch.softmax(self.weight_head(query).view(B, N_q, self.num_heads, self.num_keys), dim=-1)

        # 헤드별로 value_feat을 (B*num_heads, head_dim, H, W)로 재배치해 grid_sample로 실제 오프셋 위치를 샘플링
        v = self.value_proj(value_feat.flatten(2).transpose(1, 2))  # (B, H*W, C)
        H, W = value_feat.shape[-2:]
        v = v.transpose(1, 2).reshape(B, self.num_heads, head_dim, H, W)
        v = v.reshape(B * self.num_heads, head_dim, H, W)

        grid = offsets.clamp(-1, 1).permute(0, 2, 1, 3, 4).reshape(B * self.num_heads, N_q, self.num_keys, 2)
        sampled = F.grid_sample(v, grid, align_corners=False)  # (B*num_heads, head_dim, N_q, num_keys)
        sampled = sampled.view(B, self.num_heads, head_dim, N_q, self.num_keys).permute(0, 3, 1, 4, 2)

        # 예측된 어텐션 가중치(weights)로 K개 샘플 지점을 가중합
        out = (sampled * weights.unsqueeze(-1)).sum(dim=3)  # (B, N_q, num_heads, head_dim)
        return out.reshape(B, N_q, C)

# --- 사용 예시 ---
q = torch.randn(1, 100, 256)
val = torch.randn(1, 256, 32, 32)
attn = DeformableAttention2D()
print("Deformable Attention 출력 Shape:", attn(q, val).shape)
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. nuScenes 3D 객체 탐지 성능 비교

| 방법 (Method) | 입력 센서 모달리티 | NDS ↑ | mAP ↑ | mAVE (속도 오차) ↓ |
|---|---|---|---|---|
| **FCOS3D** | Camera | 0.428 | 0.358 | 1.14 |
| **DETR3D** | Camera | 0.479 | 0.412 | 0.84 |
| **BEVDet** | Camera | 0.488 | 0.424 | 0.328 |
| **BEVFormer-S (단일 프레임)** | Camera | 0.448 | 0.375 | 0.802 |
| **BEVFormer (시공간 융합)** | **Camera** | **0.569** | **0.481** | **0.394 (절반 감소)** |
| **SSN** | **LiDAR** | 0.569 | 0.463 | 0.280 |

- **결과**: 시간 정보(TSA) 적용 시 속도 오차(mAVE)가 **0.802 $\to$ 0.394로 절반 이상 감소**하였으며, 카메라 전용 최초로 LiDAR 기반 SSN과 동등한 **56.9% NDS** 달성.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
BEVFormer는 명시적 깊이 추정 오차를 3D Pillar Anchor 샘플링으로 우회하고 과거 프레임 BEV 정렬로 시공간을 통합한 modern 자율주행 BEV 트랜스포머 표준 아키텍처입니다.

### 2. 한계점 및 아쉬운 점
- 카메라 기반 방법은 여전히 LiDAR 대비 효율성과 정확도에서 격차가 있으며, 2D 이미지에서 정확한 3D 위치를 추론하는 문제는 근본적 과제로 남아있다.
- 6개 인코더 레이어와 다중 anchor height 샘플링으로 인한 연산 비용이 적지 않아, 임베디드 환경에서의 실시간성 확보는 별도 검증이 필요하다.

---

## 참고 자료
- [BEVFormer 공식 GitHub 저장소](https://github.com/fundamentalvision/BEVFormer)
- [ECCV 2022 논문 (arXiv:2203.17270)](https://arxiv.org/abs/2203.17270)

*관련 논문: [Attention Is All You Need](/posts/papers/attention-is-all-you-need/), [Lift Splat Shoot](/posts/papers/lift-splat-shoot/), [BEVDepth](/posts/papers/bevdepth/), [DETR3D](/posts/papers/detr3d-3d-object-detection-multi-view-images/), [BEVFusion](/posts/papers/bevfusion-multi-task-multi-sensor-fusion/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
