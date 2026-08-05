---
title: "DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries"
date: 2026-04-18T09:00:00+09:00
draft: false
tags: ["3D Object Detection", "Multi-Camera", "Transformer", "Autonomous Driving"]
categories: ["Papers", "Autonomous Driving"]
year: 2021
references:
  - "detr-end-to-end-object-detection-with-transformers"
  - "nuscenes-multimodal-dataset-autonomous-driving"
---

## 💡 한 줄 요약
DETR3D는 명시적 깊이(Depth) 추정 단계를 배제하고, 3D Object Query의 가설 지점(Reference Point)을 카메라 투영 행렬을 이용해 2D 멀티뷰 특징 맵에 역투영(Back-Projection) 및 샘플링함으로써 NMS 후처리 없이 End-to-End 멀티뷰 3D 객체 탐지를 실현했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Yue Wang, Vitor Guizilini, Tianyuan Zhang, Yilun Wang, Hang Zhao, Justin Solomon (MIT, TRI, CMU, Li Auto, Tsinghua Univ.)
- **발행년도**: 2021 (arXiv:2110.06922, CoRL 2021)
- **주요 기여점**:
  1. **Top-Down 3D-to-2D Query 역투영 아키텍처**: 조악한 픽셀별 깊이 추정(Bottom-up) 오차 누적을 피하기 위해 3D 공간 상의 객체 쿼리를 2D 멀티뷰 카메라 특징 맵에 직관적으로 역투영.
  2. **End-to-End NMS-Free 3D 탐지**: Hungarian Bipartite Matching 기반의 Set-to-Set 오차 최적화로 중복 박스 제거 후처리(NMS) 완전 폐지.
  3. **카메라 중첩 영역 (Camera Overlap Area) 정보 극대화**: 6개 멀티뷰 카메라 뷰에 3D 쿼리가 동시에 사영될 경우 특징들을 마스킹 집계(Masked Aggregation)하여 경계선 왜곡 최소화.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **Pseudo-LiDAR (이미지 $\to$ Depth $\to$ Point Cloud)**: 2D 이미지에서 추정한 깊이가 조금만 틀려도 3D 공간 포인트 위치가 폭망함.
2. **Monocular 3D Detection (FCOS3D)**: 카메라 각 뷰를 독립 탐지한 후 후처리 NMS로 병합하여 카메라 간 중첩 영역 시너지 미흡.
3. **DETR3D**: 2D DETR의 Set Prediction 아이디어를 3D 기하학 좌표 투영계와 직접 결합하여 최초의 NMS-Free 멀티뷰 3D 탐지기 완성.

---

## 📑 목차
- Chapter 1: 3D Object Query 및 Reference Point 투영 수식
- Chapter 2: Bilinear 특징 샘플링 및 마스킹 집계 (Masked Aggregation)
- Chapter 3: Iterative Refinement 헤드 및 박스 예측
- Chapter 4: Hungarian Bipartite Matching 손실 함수
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: 3D Object Query 및 Reference Point 투영 수식

### 1. 요약
$N$개의 학습 가능한 3D Object Query $\mathbf{q}_{\ell i} \in \mathbb{R}^C$로부터 3D 묵시적 가설 위치 $\mathbf{c}_{\ell i} = (x, y, z)$를 회귀한 후, 6개 카메라의 투영 행렬 $\mathbf{P}_m = \mathbf{K}_m \mathbf{E}_m$을 곱해 2D 픽셀 좌표 $\mathbf{c}_{\ell m i}$로 사영합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{c}_{\ell i} = \Phi^{\text{ref}}(\mathbf{q}_{\ell i}) \in \mathbb{R}^3$$

$$\mathbf{c}_{\ell m i} = \mathbf{P}_m (\mathbf{c}_{\ell i} \oplus 1) = \mathbf{K}_m \left( \mathbf{R}_m \mathbf{c}_{\ell i} + \mathbf{t}_m \right)$$

- **$\mathbf{c}_{\ell m i} = [u_{m,i} \cdot z_{m,i}, \ v_{m,i} \cdot z_{m,i}, \ z_{m,i}]^T$**: $m$번째 카메라 2D 사영 동차 좌표
- **$\mathbf{K}_m, \mathbf{E}_m$**: $m$번째 카메라의 내재 및 외재 변환 행렬

```python
import torch

def project_3d_queries_to_multiview_images(
    query_ref_pts: torch.Tensor, # (B, N_queries, 3) 3D 세계 좌표계 Reference Points
    P_matrices: torch.Tensor,    # (B, N_cams, 3, 4) 6개 카메라 투영 행렬
    img_shape: tuple             # (H, W)
) -> tuple:
    """
    3D Query 점들을 6개 멀티뷰 2D 카메라 평면으로 역투영 및 유효성 마스킹
    """
    B, N_q, _ = query_ref_pts.shape
    _, N_cams, _, _ = P_matrices.shape
    H, W = img_shape
    
    # 1. Homogeneous 좌표 변환 (x, y, z, 1)
    pts_3d_homo = torch.cat([query_ref_pts, torch.ones((B, N_q, 1), device=query_ref_pts.device)], dim=-1) # (B, N_q, 4)
    
    # Broadcast 사영 (B, N_cams, N_q, 3)
    pts_3d_expanded = pts_3d_homo.unsqueeze(1).repeat(1, N_cams, 1, 1)
    P_expanded = P_matrices.unsqueeze(2).repeat(1, 1, N_q, 1, 1)
    
    pts_2d_homo = torch.matmul(P_expanded, pts_3d_expanded.unsqueeze(-1)).squeeze(-1) # (B, N_cams, N_q, 3)
    
    depths = pts_2d_homo[..., 2]
    u = pts_2d_homo[..., 0] / (depths + 1e-6)
    v = pts_2d_homo[..., 1] / (depths + 1e-6)
    
    # 2. 픽셀 유효 마스크 (depth > 0 및 이미지 경계 내 존재 여부)
    mask = (depths > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    
    # 정규화 좌표 [-1, 1] (Grid Sample 용)
    norm_u = (u / W) * 2.0 - 1.0
    norm_v = (v / H) * 2.0 - 1.0
    coords_2d = torch.stack([norm_u, norm_v], dim=-1)
    
    return coords_2d, mask # (B, N_cams, N_q, 2), (B, N_cams, N_q)

# --- 사용 예시 ---
q_pts = torch.randn(2, 900, 3)
P_cams = torch.eye(4)[:3, :].view(1, 1, 3, 4).repeat(2, 6, 1, 1)
uv_coords, valid_mask = project_3d_queries_to_multiview_images(q_pts, P_cams, (480, 640))
print("2D 사영 정규화 좌표 Shape:", uv_coords.shape, "Valid Mask True 개수:", valid_mask.sum().item())
```

---

## 🛠️ Chapter 2: Bilinear 특징 샘플링 및 마스킹 집계

### 1. 요약
역투영된 2D 좌표 위치에서 Bilinear Interpolation으로 각 카메라 특징 $f_{k, m}$을 샘플링한 후, 유효 마스크 $\sigma_{k, m, i}$로 평균내어 쿼리 특징 $\mathbf{f}_i$를 갱신합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{f}_{\ell i} = \frac{1}{\sum_{k} \sum_{m} \sigma_{\ell k m i} + \epsilon} \sum_{k=1}^K \sum_{m=1}^{N_{\text{cams}}} \sigma_{\ell k m i} \cdot \text{BilinearSample}\Big( \mathcal{F}_{k, m}, \ \mathbf{c}_{\ell m i} \Big)$$

$$\mathbf{q}_{(\ell+1) i} = \mathbf{q}_{\ell i} + \mathbf{f}_{\ell i}$$

```python
import torch
import torch.nn.functional as F

def sample_and_aggregate_multiview_features(
    img_feats: torch.Tensor, # (B, N_cams, C, H, W) 6개 카메라 2D 특징 맵
    coords_2d: torch.Tensor, # (B, N_cams, N_q, 2) 2D 정규화 사영 좌표
    mask: torch.Tensor       # (B, N_cams, N_q) 유효성 마스크
) -> torch.Tensor:
    """
    Bilinear Interpolation으로 6개 카메라 특징 샘플링 후 Masked Averaging 집계
    """
    B, N_cams, C, H, W = img_feats.shape
    N_q = coords_2d.shape[2]
    
    # 1. Grid Sample (B * N_cams, C, H, W) & (B * N_cams, N_q, 1, 2)
    feats_flat = img_feats.view(B * N_cams, C, H, W)
    grid_flat = coords_2d.view(B * N_cams, N_q, 1, 2)
    
    sampled_flat = F.grid_sample(feats_flat, grid_flat, mode='bilinear', align_corners=True) # (B*N_cams, C, N_q, 1)
    sampled = sampled_flat.view(B, N_cams, C, N_q).permute(0, 1, 3, 2) # (B, N_cams, N_q, C)
    
    # 2. Masked Averaging
    mask_expanded = mask.unsqueeze(-1).float() # (B, N_cams, N_q, 1)
    masked_feats = sampled * mask_expanded
    
    sum_feats = masked_feats.sum(dim=1) # (B, N_q, C)
    count = mask_expanded.sum(dim=1).clamp(min=1e-5) # (B, N_q, 1)
    
    aggregated_feat = sum_feats / count
    return aggregated_feat

# --- 사용 예시 ---
f_img = torch.randn(2, 6, 256, 32, 32)
uv_in = torch.rand(2, 6, 900, 2) * 2.0 - 1.0
m_in = torch.randint(0, 2, (2, 6, 900)).bool()
print("집계된 Query Feature Shape:", sample_and_aggregate_multiview_features(f_img, uv_in, m_in).shape)
```

---

## 🛠️ Chapter 3: Hungarian Matching 기반 Set Loss 수식

### 1. 요약
NMS 후처리를 제거하기 위해 $N$개 예측과 $M$개 GT 사이의 Hungarian Bipartite Matching 오차 행렬을 계산하여 1:1 매칭 학습을 수행합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{L}_{\text{box}}(\mathbf{b}, \hat{\mathbf{b}}) = \lambda_{\text{L1}} \sum_{j=1}^9 \left| b_j - \hat{b}_j \right|$$

$$\mathcal{L}_{\text{sup}} = \sum_{j=1}^{N} \left[ -\log \hat{p}_{\sigma^*(j)}(c_j) + \mathbf{1}_{\{c_j \neq \varnothing\}} \mathcal{L}_{\text{box}}(\mathbf{b}_j, \hat{\mathbf{b}}_{\sigma^*(j)}) \right]$$

```python
import torch
import torch.nn.functional as F

def compute_detr3d_box_loss(
    pred_boxes: torch.Tensor, # (N_matched, 9) -> (x, y, z, w, l, h, sin, cos, vx)
    gt_boxes: torch.Tensor    # (N_matched, 9)
) -> torch.Tensor:
    """
    Hungarian 매칭된 3D 바운딩 박스 간의 L1 Loss
    """
    loss_l1 = F.l1_loss(pred_boxes, gt_boxes, reduction='mean')
    return loss_l1

# --- 사용 예시 ---
p_box = torch.randn(10, 9)
g_box = torch.randn(10, 9)
print("DETR3D 3D Box L1 Loss:", compute_detr3d_box_loss(p_box, g_box).item())
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. nuScenes 테스트셋 3D 탐지 성과

| 방법 (Method) | NMS 필요 여부 | NDS ↑ | mAP ↑ | mATE (위치 오차) ↓ |
|---|---|---|---|---|
| **CenterNet-MultiView** | 필요 (NMS) | 0.328 | 0.252 | 0.812 |
| **FCOS3D** | 필요 (NMS) | 0.428 | 0.358 | 0.690 |
| **Pseudo-LiDAR** | 필요 (NMS) | 0.160 | 0.105 | 1.210 |
| **DETR3D (Ours)** | **불필요 (NMS-Free)** | **0.479** | **0.412** | **0.647** |

- **결과**: Top-down Query 역투영 접근 및 NMS-Free 설계 덕분에 **nuScenes 리더보드 1위 (0.479 NDS)** 및 카메라 중첩 영역 인지 대폭 향상.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
DETR3D는 깊이 추정 오차 누적과 NMS 연산 병목이라는 두 가지 고질적 난제를 3D-to-2D Query 역투영 기하학 매핑으로 해소하며 현대 BEV/Query 3D 탐지 연구의 중요한 이정표를 세웠습니다.

### 2. 한계점 및 아쉬운 점
- 여전히 translation error(mATE)가 높은 편 — depth 정보 없이 위치를 정확히 추정하는 것은 근본적인 한계로 남아 있다.
- sparse query 방식이라 dense한 장면 이해(예: occupancy)에는 그대로 적용하기 어렵다.

---

## 참고 자료
- [DETR3D 공식 GitHub 저장소](https://github.com/WangYueFt/detr3d)
- [CoRL 2021 논문 (arXiv:2110.06922)](https://arxiv.org/abs/2110.06922)

*관련 논문: [DETR](/posts/papers/detr-end-to-end-object-detection-with-transformers/), [Attention Is All You Need](/posts/papers/attention-is-all-you-need/), [BEVFormer](/posts/papers/bevformer/), [Lift Splat Shoot](/posts/papers/lift-splat-shoot/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
