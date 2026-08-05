---
title: "StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction"
date: 2026-04-20T20:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "HD Map"]
tags: ["Autonomous Driving", "HD Map", "BEV", "Temporal Fusion", "Transformer", "Streaming"]
year: 2024
references:
  - "maptr-structured-modeling-online-vectorized-hd-map-construction"
  - "bevformer"
---

## 💡 한 줄 요약
StreamMapNet은 폴리라인 전체 점을 참조점으로 활용하는 **Multi-Point Attention**과 메모리 오버헤드 없이 장기 이력을 융합하는 **Streaming Temporal Fusion (Query Propagation + ConvGRU BEV Warping)**을 결합하여 넓은 인식 범위(100m $\times$ 50m)에서도 14.2 FPS의 실시간 맵 구축 SOTA를 달성했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Tianyuan Yuan, Yicheng Liu, Yue Wang, Yilun Wang, Hang Zhao (Tsinghua Univ., USC)
- **발행년도**: 2024 (arXiv:2308.12570, WACV 2024)
- **주요 기여점**:
  1. **Multi-Point Attention**: 중심점 1개만 보던 기존 Deformable Attention의 한계를 극복하고, 폴리라인 $N_p$개 점 전체를 Reference Point로 지정하여 비국소(Non-Local) 넓은 도로 선형 특징 수집.
  2. **Streaming Temporal Fusion**: 상위 $k$개 쿼리의 위치/의미를 전파하는 Query Propagation과 밀집 BEV 특징 맵을 자차 이동으로 정렬한 뒤 융합하는 ConvGRU 레지스터로 장기 기억 보존.
  3. **지리적 위치 누출 없는 공정한 Split 구축**: 기존 nuScenes(85% 겹침) 및 Argoverse2(54% 겹침) 주행 검증 영역 위치 누출(Data Leakage)을 규명하고 0% 겹침 새 Geographic Split 제시.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **단일 프레임 HD Map 모델 (VectorMapNet, MapTR)**: 프레임 독립 예측으로 인해 덤프트럭이나 대형 차량에 가려진(Occluded) 도로 선형을 복원하지 못함.
2. **Stacking 시간 융합 (BEVDet4D)**: 과거 프레임을 단순 Concatenate하여 프레임 수가 증가함에 따라 메모리/지연시간이 $\mathcal{O}(T)$로 증가.
3. **StreamMapNet**: 프레임 수와 무관한 $\mathcal{O}(1)$ 상수 비용으로 스트리밍 연속 HD 맵 구축.

---

## 📑 목차
- Chapter 1: Multi-Point Attention (비국소 폴리라인 어텐션)
- Chapter 2: Streaming Query Propagation & 자차 좌표계 보정
- Chapter 3: BEV Spatial Warping & ConvGRU 시간 융합
- Chapter 4: 다중 태스크 학습 손실 및 Polyline Match Loss
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: Multi-Point Attention 수식

### 1. 요약
폴리라인을 구성하는 $N_p$개 점 좌표 $P_i^j$ 각각을 Deformable Reference Point로 삼고 주변 $N_{\text{off}}$개 오프셋 피처를 집계하여 길고 곡률이 큰 도로 경계를 한 번에 포착합니다.

### 2. 수식 및 파이썬 코드 설명

$$Q_i = \sum_{j=1}^{N_p} \sum_{k=1}^{N_{\text{off}}} W_i^{(j-1) \cdot N_{\text{off}}+k} \cdot \text{DeformAttn}\left( Q_{i-1}, \ P_i^j + O_i^{(j-1) \cdot N_{\text{off}}+k}, \ \mathcal{F}_{\text{BEV}} \right)$$

- **$P_i^j$**: $i$번째 레이어에서 예측된 폴리라인의 $j$번째 점 좌표
- **$O_i$**: 점 $j$ 주변의 학습된 서브 오프셋 벡터

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiPointDeformableAttention(nn.Module):
    """
    StreamMapNet Multi-Point Attention: 폴리라인 N_p개 점 모두를 Reference Point로 사용
    """
    def __init__(self, embed_dim: int = 256, num_points: int = 20, num_offsets: int = 4):
        super().__init__()
        self.num_points = num_points
        self.num_offsets = num_offsets
        
        # N_p * N_off 오프셋 및 가중치 생성기
        self.offset_proj = nn.Linear(embed_dim, num_points * num_offsets * 2)
        self.weight_proj = nn.Linear(embed_dim, num_points * num_offsets)

    def forward(self, query: torch.Tensor, polyline_pts: torch.Tensor, bev_feat: torch.Tensor) -> torch.Tensor:
        """
        query: (B, N_queries, C)
        polyline_pts: (B, N_queries, N_p, 2) -> 폴리라인 예측 점 좌표
        bev_feat: (B, C, H, W)
        """
        B, N_q, C = query.shape
        offsets = self.offset_proj(query).view(B, N_q, self.num_points, self.num_offsets, 2)
        weights = torch.softmax(
            self.weight_proj(query).view(B, N_q, self.num_points, self.num_offsets), dim=-1
        )
        
        # N_p개 점 주변 오프셋 위치에서 BEV 피처를 실제로 샘플링 (정규화 좌표 [-1, 1] 가정)
        sample_locations = (polyline_pts.unsqueeze(3) + offsets).clamp(-1, 1)  # (B, N_q, N_p, N_off, 2)
        grid = sample_locations.view(B, N_q * self.num_points, self.num_offsets, 2)
        sampled = F.grid_sample(bev_feat, grid, align_corners=False)  # (B, C, N_q*N_p, N_off)
        sampled = sampled.view(B, C, N_q, self.num_points, self.num_offsets).permute(0, 2, 3, 4, 1)

        # 포인트/오프셋 차원에 대해 어텐션 가중합 후 쿼리 채널(C)에 맞춰 잔차 결합
        attn_out = (sampled * weights.unsqueeze(-1)).sum(dim=(2, 3)) / self.num_points  # (B, N_q, C)
        updated_query = query + attn_out
        return updated_query

# --- 사용 예시 ---
q_in = torch.randn(2, 50, 256)
pts_in = torch.rand(2, 50, 20, 2)
b_f = torch.randn(2, 256, 100, 100)
mp_attn = MultiPointDeformableAttention()
print("Multi-Point Attention 출력 Query Shape:", mp_attn(q_in, pts_in, b_f).shape)
```

---

## 🛠️ Chapter 2: Streaming Query Propagation & 자차 좌표계 보정

### 1. 요약
이전 프레임의 상위 $k$개 쿼리 $Q_{t-1}$와 폴리라인 위치 $\mathbf{P}_{t-1}$를 자차 이동 행렬 $\mathbf{T}$로 변환하여 현재 프레임의 쿼리로 연속 전달(Propagate)합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{P}_t = \mathbf{T}_{t-1 \to t} \cdot \mathbf{P}_{t-1}$$

$$Q_t = \text{MLP}\Big( \text{Concat}\big( Q_{t-1}, \ \text{Flatten}(\mathbf{T}) \big) \Big) + Q_{t-1}$$

```python
import torch

def propagate_queries_with_ego_transform(
    prev_queries: torch.Tensor, # (B, K_top, C) 이전 상위 K개 쿼리
    prev_polylines: torch.Tensor,# (B, K_top, N_p, 2) 이전 폴리라인 2D 좌표
    T_prev2curr: torch.Tensor   # (B, 4, 4) Ego-Motion 변환 행렬
) -> tuple:
    """
    StreamMapNet Query Propagation: 이전 쿼리 및 폴리라인 점을 현재 좌표계로 전파
    """
    B, K_top, N_p, _ = prev_polylines.shape
    
    # 1. 2D 폴리라인 점을 동차 3D 좌표로 확장 (x, y, 0, 1)
    pts_3d_homo = torch.stack([
        prev_polylines[..., 0], prev_polylines[..., 1],
        torch.zeros_like(prev_polylines[..., 0]), torch.ones_like(prev_polylines[..., 0])
    ], dim=-1) # (B, K_top, N_p, 4)
    
    # 2. 좌표 변환 P_curr = P_prev @ T_trans^T
    pts_transformed = torch.matmul(pts_3d_homo, T_prev2curr.transpose(-2, -1))
    curr_polylines = pts_transformed[..., :2] # (B, K_top, N_p, 2)
    
    return prev_queries, curr_polylines

# --- 사용 예시 ---
q_prev = torch.randn(1, 10, 256)
poly_prev = torch.rand(1, 10, 20, 2) * 50.0
T_trans = torch.eye(4)
T_trans[0, 3] = 2.0 # 2m 이동
_, poly_curr = propagate_queries_with_ego_transform(q_prev, poly_prev, T_trans.unsqueeze(0))
print("전파 및 좌표 정렬된 현재 폴리라인 Shape:", poly_curr.shape)
```

---

## 🛠️ Chapter 3: BEV Spatial Warping & ConvGRU 시간 융합

### 1. 요약
이전 프레임의 밀집 BEV 특징 맵 $\mathcal{F}_{\text{BEV}}^{t-1}$을 자차 이동 행렬 $\mathbf{T}$로 공간 워핑(Warping)한 후, ConvGRU를 통과시켜 현재 BEV 특징 맵 $\mathcal{F}_{\text{BEV}}^t$에 적응적 융합합니다.

### 2. 수식 및 파이썬 코드 설명

$$\tilde{\mathcal{F}}_{\text{BEV}}^{t-1} = \text{Warp}\left( \mathcal{F}_{\text{BEV}}^{t-1}, \ \mathbf{T} \right)$$

$$\mathcal{F}_{\text{BEV}}^t = \text{LayerNorm}\left( \text{ConvGRU}\big( \tilde{\mathcal{F}}_{\text{BEV}}^{t-1}, \ \mathcal{F}_{\text{BEV}}^t \big) \right)$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRUBEVFusion(nn.Module):
    """
    StreamMapNet 밀집 BEV 특징 맵 ConvGRU 시간 융합 모듈
    """
    def __init__(self, channels: int = 256):
        super().__init__()
        self.update_gate = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.reset_gate = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.out_gate = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)

    def forward(self, warped_prev_bev: torch.Tensor, curr_bev: torch.Tensor) -> torch.Tensor:
        concat_in = torch.cat([warped_prev_bev, curr_bev], dim=1)
        
        z = torch.sigmoid(self.update_gate(concat_in))
        r = torch.sigmoid(self.reset_gate(concat_in))
        
        h_candidate = torch.tanh(self.out_gate(torch.cat([r * warped_prev_bev, curr_bev], dim=1)))
        fused_bev = (1.0 - z) * warped_prev_bev + z * h_candidate
        return fused_bev

# --- 사용 예시 ---
prev_w = torch.randn(1, 256, 50, 50)
curr_b = torch.randn(1, 256, 50, 50)
gru_fusion = ConvGRUBEVFusion()
print("ConvGRU 융합 BEV Feature Shape:", gru_fusion(prev_w, curr_b).shape)
```

---

## 🛠️ Chapter 4: 다중 태스크 학습 손실 함수

### 1. 요약
폴리라인 좌표 Smooth L1 손실 $\mathcal{L}_{\text{line}}$, 분류 Focal Loss $\mathcal{L}_{\text{Focal}}$, 그리고 쿼리 전파 보조 손실 $\mathcal{L}_{\text{trans}}$를 가중 합산합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{L}_{\text{train}} = 50 \cdot \mathcal{L}_{\text{line}} + 5 \cdot \mathcal{L}_{\text{Focal}} + 5 \cdot \mathcal{L}_{\text{trans}}$$

```python
import torch
import torch.nn.functional as F

def compute_streammapnet_total_loss(
    pred_polylines: torch.Tensor, # (B, N_q, N_p, 2)
    gt_polylines: torch.Tensor,   # (B, N_q, N_p, 2)
    cls_logits: torch.Tensor,
    gt_cls_labels: torch.Tensor
) -> torch.Tensor:
    """
    StreamMapNet 다중 태스크 손실 계산 (Line Regression + Focal Classification)
    """
    loss_line = F.smooth_l1_loss(pred_polylines, gt_polylines, reduction='mean')
    loss_cls = F.cross_entropy(cls_logits.view(-1, cls_logits.shape[-1]), gt_cls_labels.view(-1))
    
    total_loss = 50.0 * loss_line + 5.0 * loss_cls
    return total_loss

# --- 사용 예시 ---
p_poly = torch.randn(2, 50, 20, 2)
g_poly = torch.randn(2, 50, 20, 2)
p_c = torch.randn(2, 50, 4)
g_c = torch.randint(0, 4, (2, 50))
print("StreamMapNet 학습 손실:", compute_streammapnet_total_loss(p_poly, g_poly, p_c, g_c).item())
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. Argoverse2 100m $\times$ 50m 지리적 0% 겹침 Split 리더보드

| 알고리즘 (Method) | 입력 모달리티 | 인식 범위 (Range) | mAP ↑ | 추론 속도 (FPS) ↑ |
|---|---|---|---|---|
| **VectorMapNet** | Camera | 100m $\times$ 50m | 25.7 | 5.5 FPS (느림) |
| **MapTR** | Camera | 100m $\times$ 50m | 40.2 | **18.0 FPS** |
| **StreamMapNet (Ours)** | **Camera** | **100m $\times$ 50m (광범위)** | **51.2 (+11.0%)** | **14.2 FPS (실시간 SOTA)** |

- **결과**: Multi-Point Attention 및 Streaming 시간 융합으로 넓은 100m 탐지 영역에서 MapTR을 **+11.0 mAP 차이로 완전히 압도**.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
StreamMapNet은 온라인 벡터 HD 맵 구축 분야에 Multi-Point Attention과 1프레임 지연 스트리밍 시간 융합을 도입하여 100m 대범위 고정밀 맵 생성을 실시간화한 탁월한 프레임워크입니다.

### 2. 한계점 및 아쉬운 점
- 벤치마크 재정의라는 기여가 모델 자체의 기여를 가릴 만큼 크며, 새 split에서의 절대 성능은 여전히 낮은 편(100m에서 mAP 51.2)이라 실제 배포까지는 추가 개선이 필요하다.
- LiDAR 없이 카메라만으로 먼 거리의 정밀도를 확보하는 데는 한계가 있다.

---

## 참고 자료
- [StreamMapNet 공식 GitHub 저장소](https://github.com/yuantianyuan01/StreamMapNet)
- [WACV 2024 논문 (arXiv:2308.12570)](https://arxiv.org/abs/2308.12570)

*관련 논문: [VectorMapNet](/posts/papers/vectormapnet-end-to-end-vectorized-hd-map-learning/), [MapTR](/posts/papers/maptr-structured-modeling-online-vectorized-hd-map-construction/), [BEVFormer](/posts/papers/bevformer/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
