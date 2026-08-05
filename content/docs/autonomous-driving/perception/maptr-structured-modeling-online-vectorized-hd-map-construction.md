---
title: "MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction"
date: 2026-04-20T12:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "HD Map"]
tags: ["HD Map", "Autonomous Driving", "Transformer", "BEV", "Vectorized Map"]
year: 2022
references:
  - "detr-end-to-end-object-detection-with-transformers"
  - "vectormapnet-end-to-end-vectorized-hd-map-learning"
  - "nuscenes-multimodal-dataset-autonomous-driving"
---

## 💡 한 줄 요약
맵 요소(차선, 횡단보도, 도로 경계)를 기하학적 등가 순열 집합(Permutation-Equivalent Point Set)으로 체계화하고, 인스턴스-포인트 계층적 쿼리(Hierarchical Query) 기반의 병렬 DETR 디코더로 예측하여 실시간(25.1 FPS)으로 벡터화 HD 맵을 생성한다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Bencheng Liao, Shaoyu Chen, Xinggang Wang, Tianheng Cheng, Qian Zhang, Wenyu Liu, Chang Huang (HUST, Horizon Robotics)
- **발행년도**: 2022 (arXiv:2208.14437, ICLR 2023)
- **주요 기여점**:
  1. **Permutation-Equivalent Modeling**: 차선(Polyline) 및 횡단보도(Polygon)의 순서 모호성(Ambiguity)을 제거하기 위해 점 집합 $V$와 등가 순열 그룹 $\Gamma$를 도입.
  2. **계층적 2단계 이분 매칭 (Hierarchical Bipartite Matching)**: 인스턴스 매칭 후 등가 순열 그룹 $\Gamma$ 내에서 맨해튼 거리가 가장 최소인 최적 점 순서 $\hat{\gamma}$를 동적 선택.
  3. **Point2Point 및 Cosine Edge Direction Loss**: 점 위치 정밀도뿐 아니라 연결선(Edge)의 방향 코사인 유사도까지 강제 최적화.
  4. **초고속 실시간 벡터화 매핑**: 병렬 디코딩 아키텍처로 VectorMapNet 대비 8배 가속된 25.1 FPS와 nuScenes 50.3 mAP 달성.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **Rasterized BEV Map (HDMapNet)**: BEV 세그멘테이션 래스터 이미지 출력 후 후처리로 인스턴스를 추출하나 200ms 이상의 지연 발생.
2. **Autoregressive Vectorized Map (VectorMapNet)**: 최초의 End-to-End 벡터화 방법이나 픽셀/점 하나씩 순차 예측하여 추론 시간이 매우 느림.
3. **MapTR**: 등가 순열 집합 표현과 병렬 Transformer 기법을 최초 도입하여 25 FPS 실시간 맵 구축 가능.

---

## 📑 목차
- Chapter 1: 등가 순열 모델링 (Permutation-Equivalent Modeling)
- Chapter 2: 계층적 쿼리 (Hierarchical Query) 아키텍처
- Chapter 3: 2단계 계층적 매칭 알고리즘
- Chapter 4: Point2Point & Cosine Edge Direction Loss 손실 함수
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: 등가 순열 모델링 (Permutation-Equivalent Modeling)

### 1. 요약
맵 요소 $V = \{v_j\}_{j=0}^{N_v - 1}$에 대해, 개방형 차선(Polyline)은 2가지 순서 $\Gamma_{\text{polyline}}$, 폐쇄형 횡단보도(Polygon)는 $2 N_v$가지 순서 $\Gamma_{\text{polygon}}$을 동등한 정답으로 정의합니다.

### 2. 수식 및 파이썬 코드 설명

$$\Gamma_{\text{polyline}} = \{\gamma^0, \gamma^1\}, \quad \gamma^0(j) = j, \quad \gamma^1(j) = N_v - 1 - j$$

$$\Gamma_{\text{polygon}} = \{\gamma^k\}_{k=0}^{2 N_v - 1}, \quad \gamma^{2m}(j) = (j + m) \bmod N_v, \quad \gamma^{2m+1}(j) = (N_v - 1 - j + m) \bmod N_v$$

```python
import torch

def generate_permutation_equivalent_sets(
    gt_points: torch.Tensor, # (Nv, 2) 맵 요소 점 좌표
    is_polygon: bool = False
) -> torch.Tensor:
    """
    MapTR: GT 점 집합에 대한 등가 순열 그룹 Gamma 생성
    """
    Nv, _ = gt_points.shape
    equivalent_sets = []
    
    if not is_polygon:
        # Polyline: 정방향(0 -> Nv-1) 및 역방향(Nv-1 -> 0)
        equivalent_sets.append(gt_points)
        equivalent_sets.append(torch.flip(gt_points, dims=[0]))
    else:
        # Polygon: 2 * Nv 가지 시작점 및 정/역방향 순회
        for m in range(Nv):
            rolled = torch.roll(gt_points, shifts=-m, dims=0)
            equivalent_sets.append(rolled)
            equivalent_sets.append(torch.flip(rolled, dims=[0]))
            
    return torch.stack(equivalent_sets, dim=0) # (|Gamma|, Nv, 2)

# --- 사용 예시 ---
pts_poly = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
gamma_sets = generate_permutation_equivalent_sets(pts_poly, is_polygon=True)
print("Polygon 등가 순열 수 (|Gamma|):", gamma_sets.shape[0])
```

---

## 🛠️ Chapter 2: 계층적 쿼리 (Hierarchical Query) 아키텍처

### 1. 요약
인스턴스 쿼리 $q_i^{\text{ins}}$와 포인트 쿼리 $q_j^{\text{pt}}$를 합산하여 맵 요소 $i$의 $j$번째 점 좌표를 예측하는 $q_{ij}^{\text{hie}}$를 구성합니다.

### 2. 수식 및 파이썬 코드 설명

$$q_{ij}^{\text{hie}} = q_i^{\text{ins}} + q_j^{\text{pt}} \in \mathbb{R}^C \quad (i = 1 \dots N_{\text{ins}}, \ j = 1 \dots N_v)$$

```python
import torch
import torch.nn as nn

class MapTRHierarchicalQuery(nn.Module):
    """
    MapTR 계층적 쿼리 임베딩 모듈 (Instance Query + Point Query)
    """
    def __init__(self, num_instances: int = 50, num_points: int = 20, embed_dim: int = 256):
        super().__init__()
        self.instance_embed = nn.Embedding(num_instances, embed_dim)
        self.point_embed = nn.Embedding(num_points, embed_dim)

    def forward(self) -> torch.Tensor:
        # q_ins: (N_ins, 1, C) / q_pt: (1, N_pt, C)
        q_ins = self.instance_embed.weight.unsqueeze(1)
        q_pt = self.point_embed.weight.unsqueeze(0)
        
        # Broadcasting Sum: (N_ins, N_pt, C)
        q_hie = q_ins + q_pt
        return q_hie

# --- 사용 예시 ---
h_query = MapTRHierarchicalQuery()
print("생성된 계층적 쿼리 Tensor Shape:", h_query().shape)
```

---

## 🛠️ Chapter 3: Point2Point & Cosine Edge Direction Loss

### 1. 요약
예측된 점과 최적 순열 점 간의 맨해튼 거리 오차 $\mathcal{L}_{\text{p2p}}$와 인접한 연결선(Edge) 벡터 간의 코사인 유사도 오차 $\mathcal{L}_{\text{dir}}$를 함께 최소화합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{L}_{\text{p2p}} = \sum_{j=0}^{N_v - 1} \left\| \hat{v}_j - v_{\hat{\gamma}(j)} \right\|_1$$

$$\mathcal{L}_{\text{dir}} = -\sum_{j=0}^{N_v - 1} \frac{\hat{e}_j \cdot e_{\hat{\gamma}(j)}}{\left\|\hat{e}_j\right\|_2 \left\|e_{\hat{\gamma}(j)}\right\|_2}, \quad \hat{e}_j = \hat{v}_{(j+1)\bmod N_v} - \hat{v}_j$$

```python
import torch
import torch.nn.functional as F

def compute_maptr_point_and_edge_loss(
    pred_pts: torch.Tensor, # (Nv, 2) 예측 점 좌표
    gt_pts: torch.Tensor,   # (Nv, 2) 최적 정렬된 GT 점 좌표
    is_polygon: bool = False
) -> torch.Tensor:
    """
    MapTR: Point-to-Point L1 Loss + Cosine Edge Direction Loss
    """
    # 1. Point2Point L1 Loss
    loss_p2p = F.l1_loss(pred_pts, gt_pts, reduction='sum')
    
    # 2. Edge Direction Vector 생성 (j+1 - j)
    if is_polygon:
        pred_edges = torch.roll(pred_pts, shifts=-1, dims=0) - pred_pts
        gt_edges = torch.roll(gt_pts, shifts=-1, dims=0) - gt_pts
    else:
        pred_edges = pred_pts[1:] - pred_pts[:-1]
        gt_edges = gt_pts[1:] - gt_pts[:-1]
        
    # 3. Cosine Similarity 계산
    cos_sim = F.cosine_similarity(pred_edges, gt_edges, dim=-1)
    loss_dir = (1.0 - cos_sim).sum()
    
    return loss_p2p + 0.005 * loss_dir

# --- 사용 예시 ---
p_res = torch.randn(20, 2)
g_res = torch.randn(20, 2)
print("MapTR 총 맵 손실 (P2P + Edge Dir):", compute_maptr_point_and_edge_loss(p_res, g_res).item())
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. nuScenes 온라인 HD 맵 구축 벤치마크 비교

| 알고리즘 (Method) | 입력 모달리티 | 백본 (Backbone) | mAP ↑ | 추론 속도 (FPS) ↑ |
|---|---|---|---|---|
| **VectorMapNet** | Camera | ResNet50 | 40.9 | 2.9 FPS (느림) |
| **MapTR-nano** | Camera | ResNet18 | 45.9 | **25.1 FPS (실시간)** |
| **MapTR-tiny** | Camera | ResNet50 | **50.3** | **11.2 FPS** |
| **MapTR-tiny (110ep)** | **Camera** | **ResNet50** | **58.7 (+17.8%)** | **11.2 FPS (LiDAR 초과)** |

- **결과**: 등가 순열 모델링과 병렬 디코딩 기법을 통해 **25.1 FPS 실시간 맵 구축**과 **카메라 전용 SOTA(58.7 mAP)** 달성.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
MapTR은 순열-등가 모델링을 통해 벡터화 지도 학습의 표현 모호성을 철폐하고, 실시간(25 FPS)으로 고품질 HD 맵을 온라인 생성하는 패러다임을 확립했습니다.

### 2. 한계점 및 아쉬운 점
- permutation-equivalent 모델링은 맵 요소의 점 개수 $N_v$가 고정되어야 하는 제약이 있어, 매우 복잡하거나 불규칙한 형상은 표현력이 제한될 수 있다.
- 실험이 nuScenes 도심 환경에 집중되어 있어, 고속도로나 복잡한 교차로 등 다양한 도로 토폴로지에서의 일반화 성능은 추가 검증이 필요하다.

---

## 참고 자료
- [MapTR 공식 GitHub 저장소](https://github.com/hustvl/MapTR)
- [ICLR 2023 논문 (arXiv:2208.14437)](https://arxiv.org/abs/2208.14437)

*관련 논문: [VectorMapNet](/posts/papers/vectormapnet-end-to-end-vectorized-hd-map-learning/), [DETR](/posts/papers/detr-end-to-end-object-detection-with-transformers/), [BEVFormer](/posts/papers/bevformer/), [StreamMapNet](/posts/papers/streammapnet-streaming-mapping-network-vectorized-online-hd-map-construction/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
