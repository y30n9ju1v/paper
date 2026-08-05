---
title: "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space"
date: 2026-04-29T00:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["Point Cloud", "3D Classification", "3D Segmentation", "Deep Learning", "LiDAR", "Hierarchical Learning", "PointNet"]
year: 2017
references:
  - "pointnet-deep-learning-on-point-sets-for-3d-classification-and-segmentation"
---

## 💡 한 줄 요약
PointNet의 "로컬 주변 기하 구조 미반영" 한계를 극복하기 위해 **Farthest Point Sampling (FPS)**, **Ball Query 공간 탐색**, **Mini-PointNet 인코딩**을 수직 결합한 계층적 Set Abstraction 구조와 밀도 적응형 피처 추출(MSG/MRG)을 제안했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas (Stanford University)
- **발행년도**: 2017 (arXiv:1706.02413, NIPS 2017)
- **주요 기여점**:
  1. **Set Abstraction (SA) 계층 구조**: Sampling(FPS) $\to$ Grouping(Ball Query) $\to$ Mini-PointNet 과정을 거쳐 2D CNN의 수용 영역(Receptive Field) 확장 과정을 3D 비정형 포인트 공간에 재현.
  2. **Multi-Scale Grouping (MSG) & Multi-Resolution Grouping (MRG)**: 반경 $r_1, r_2, \dots$ 다중 반경 그룹화 및 Random Input Dropout으로 거리/밀도 불균일(Density Heterogeneity) 문제 해결.
  3. **Feature Propagation (FP) 역거리 보간**: 3D Semantic Segmentation을 위해 kNN 역거리 가중치(Inverse Distance Weighting) 보간과 Skip Connection을 융합한 인버티드 U-Net 설계.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **PointNet**: 전역 포인트 집합 전체에 한 차례 Max-Pooling만 적용하므로 세밀한 로컬 이웃의 곡률, 엣지, 파트 연결 정보를 파악하지 못함.
2. **PointNet++**: 메트릭 공간 상의 반경 $r$ 구형 영역(Ball) 단위로 국소 점들을 중첩 묶어 계층적 로컬 특징 학습 완성.

---

## 📑 목차
- Chapter 1: Farthest Point Sampling (FPS) & Ball Query 수식
- Chapter 2: Set Abstraction (SA) 모듈 파이프라인
- Chapter 3: Multi-Scale Grouping (MSG) 밀도 적응 모듈
- Chapter 4: Feature Propagation (FP) 역거리 가중치 보간
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: Farthest Point Sampling (FPS) & Ball Query 수식

### 1. 요약
전체 점 집합 $P$에서 점들 간의 최소 유클리드 거리가 최대가 되는 정점 중심점 $c_i$를 선택(FPS)하고, 반경 $r$ 내 $K$개 점들을 그룹화(Ball Query)합니다.

### 2. 수식 및 파이썬 코드 설명

$$d(p_i, S) = \min_{s \in S} \| p_i - s \|_2, \quad p_{\text{next}} = \arg\max_{p_i \notin S} d(p_i, S)$$

$$\text{BallQuery}(c, r, K) = \Big\{ x_i \;\Big|\; \| x_i - c \|_2 \leq r \Big\}_{i=1}^K$$

```python
import torch

def farthest_point_sampling(pts: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    PointNet++ Farthest Point Sampling (FPS)
    pts: (B, N, 3)
    return: (B, npoint) 선택된 렌더링 중심점 인덱스
    """
    B, N, C = pts.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=pts.device)
    distance = torch.ones(B, N, device=pts.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=pts.device)
    batch_indices = torch.arange(B, dtype=torch.long, device=pts.device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = pts[batch_indices, farthest, :].unsqueeze(1) # (B, 1, 3)
        dist = torch.sum((pts - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        
    return centroids

# --- 사용 예시 ---
pts_input = torch.randn(2, 1024, 3)
fps_indices = farthest_point_sampling(pts_input, npoint=128)
print("FPS 선별 128개 중심점 인덱스 Shape:", fps_indices.shape)
```

---

## 🛠️ Chapter 2: Set Abstraction (SA) 모듈 파이프라인

### 1. 요약
중심점 $c$에 묶인 이웃 점 $x_i$의 좌표를 중심 상대 좌표 $x_i - c$로 정규화하고 Mini-PointNet을 통해 고차원 로컬 특징으로 압축합니다.

### 2. 수식 및 파이썬 코드 설명

$$x_i^{\text{relative}} = x_i - c$$

$$\mathbf{f}_{\text{SA}} = \text{MaxPool}_{k=1 \dots K} \Big( \text{MLP}\big( [x_k - c \;;\; \mathbf{f}_k] \big) \Big)$$

```python
import torch
import torch.nn as nn
# farthest_point_sampling은 Chapter 1에서 정의한 함수를 그대로 사용

class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction (SA) 모듈: Sampling -> Grouping -> Mini-PointNet
    """
    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: list):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        
        layers = []
        last_channel = in_channel + 3 # 상대 좌표 (+3)
        for out_channel in mlp:
            layers.extend([
                nn.Conv2d(last_channel, out_channel, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            ])
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor = None) -> tuple:
        """
        xyz: (B, N, 3), points: (B, N, C_feat)
        """
        B, N, _ = xyz.shape
        # 1. Sampling: FPS로 중심점 npoint개 선택
        fps_idx = farthest_point_sampling(xyz, self.npoint)  # (B, npoint)
        batch_indices = torch.arange(B, device=xyz.device).view(B, 1).repeat(1, self.npoint)
        new_xyz = xyz[batch_indices, fps_idx, :]  # (B, npoint, 3)

        # 2. Grouping: 각 중심점 반경(radius) 내 nsample개 이웃을 Ball Query로 탐색
        sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)  # (B, npoint, N)
        group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, self.npoint, 1)
        group_idx[sqrdists > self.radius ** 2] = N  # 반경 밖 포인트는 N(더미)으로 마스킹
        group_idx = group_idx.sort(dim=-1)[0][:, :, :self.nsample]  # 가까운 nsample개만 채택
        group_first = group_idx[:, :, 0:1].repeat(1, 1, self.nsample)
        mask = group_idx == N
        group_idx[mask] = group_first[mask]  # 반경 내 포인트가 부족하면 첫 이웃으로 패딩

        batch_indices_g = batch_indices.view(B, self.npoint, 1).repeat(1, 1, self.nsample)
        grouped_xyz = xyz[batch_indices_g, group_idx, :] - new_xyz.unsqueeze(2)  # 중심점 기준 상대좌표

        if points is not None:
            grouped_points = points[batch_indices_g, group_idx, :]
            grouped_feat = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_feat = grouped_xyz
            
        grouped_feat = grouped_feat.permute(0, 3, 1, 2) # (B, C_in+3, npoint, nsample)
        feat = self.mlp(grouped_feat)
        new_points, _ = torch.max(feat, dim=-1) # Max-Pooling over nsample
        return new_xyz, new_points.permute(0, 2, 1)

# --- 사용 예시 ---
sa_module = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=32, in_channel=0, mlp=[64, 128])
xyz_in = torch.randn(2, 1024, 3)
n_xyz, n_pts = sa_module(xyz_in)
print("SA 모듈 출력 좌표 Shape:", n_xyz.shape, "출력 특징 Shape:", n_pts.shape)
```

---

## 🛠️ Chapter 3: Multi-Scale Grouping (MSG) 밀도 적응 모듈

### 1. 요약
포인트 클라우드의 거리별 밀도 불균일성을 처리하기 위해 반경 $r_1 < r_2 < r_3$ 세 구형 영역을 동시 그룹화한 후 Concat합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{f}_{\text{MSG}} = \text{Concat}\Big( \text{SA}_{r_1}(X), \; \text{SA}_{r_2}(X), \; \dots, \; \text{SA}_{r_M}(X) \Big)$$

```python
import torch
import torch.nn as nn

class MultiScaleGroupingModule(nn.Module):
    """
    PointNet++ Multi-Scale Grouping (MSG) 모듈
    """
    def __init__(self, sa_list: list):
        super().__init__()
        self.sa_modules = nn.ModuleList(sa_list)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor = None) -> tuple:
        new_xyz = None
        outputs = []
        for sa in self.sa_modules:
            new_xyz, feat = sa(xyz, points)
            outputs.append(feat)
            
        # 다중 스케일 특징 Concat
        fused_msg_features = torch.cat(outputs, dim=-1)
        return new_xyz, fused_msg_features

# --- 사용 예시 ---
sa1 = PointNetSetAbstraction(npoint=128, radius=0.1, nsample=16, in_channel=0, mlp=[32, 64])
sa2 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=32, in_channel=0, mlp=[64, 128])
msg_module = MultiScaleGroupingModule([sa1, sa2])
xyz_in = torch.randn(2, 1024, 3)
_, msg_out = msg_module(xyz_in)
print("MSG 다중 스케일 융합 특징 Shape:", msg_out.shape)
```

---

## 🛠️ Chapter 4: Feature Propagation (FP) 역거리 가중치 보간

### 1. 요약
축소된 점 $N'$에서 원본 $N$개 점으로 특징을 복원하기 위해 $k=3$ 최근접 이웃 점들의 역거리 가중치(Inverse Distance Weighting) $w_i$로 보간(Interpolation)합니다.

### 2. 수식 및 파이썬 코드 설명

$$f^{(j)}(x) = \frac{\sum_{i=1}^k w_i(x) f_i^{(j)}}{\sum_{i=1}^k w_i(x)}, \quad w_i(x) = \frac{1}{d(x, x_i)^p}$$

```python
import torch

def inverse_distance_weighted_interpolation(
    sparse_xyz: torch.Tensor, # (B, N_sparse, 3)
    sparse_feat: torch.Tensor,# (B, N_sparse, C)
    dense_xyz: torch.Tensor,  # (B, N_dense, 3)
    k: int = 3,
    p: int = 2
) -> torch.Tensor:
    """
    PointNet++ Feature Propagation (FP): 역거리 가중치 (p=2) 기반 kNN 보간
    """
    B, N_dense, _ = dense_xyz.shape
    dists = torch.cdist(dense_xyz, sparse_xyz) # (B, N_dense, N_sparse)
    dists, idxs = torch.topk(dists, k=k, dim=-1, largest=False) # 상위 k개 최소 거리
    
    weights = 1.0 / (dists ** p + 1e-8)
    norm_weights = weights / torch.sum(weights, dim=-1, keepdim=True) # (B, N_dense, k)
    
    # 3개 이웃 특징 가중합
    interpolated_feat = torch.zeros(B, N_dense, sparse_feat.shape[-1], device=sparse_xyz.device)
    for b in range(B):
        for i in range(k):
            interpolated_feat[b] += norm_weights[b, :, i:i+1] * sparse_feat[b, idxs[b, :, i], :]
            
    return interpolated_feat

# --- 사용 예시 ---
s_xyz = torch.randn(2, 128, 3)
s_feat = torch.randn(2, 128, 64)
d_xyz = torch.randn(2, 1024, 3)
interp_res = inverse_distance_weighted_interpolation(s_xyz, s_feat, d_xyz)
print("FP 역거리 보간으로 복원된 특징 Shape:", interp_res.shape)
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. ModelNet40 및 ScanNet 벤치마크 성능 비교

| 알고리즘 (Method) | 대표 입력 표현 | 분류 정확도 (ModelNet40) ↑ | 세그멘테이션 mIoU (ScanNet) ↑ |
|---|---|---|---|
| **PointNet (Vanilla)** | Point Cloud | 89.2% | 73.0% |
| **PointNet++ (SSG)** | Point Cloud | 91.5% | 81.2% |
| **PointNet++ (MSG+DP)** | **Point Cloud (Multi-Scale)** | **91.9% (+2.7%)** | **84.5% (+11.5%)** |

- **결과**: PointNet 대비 로컬 계층 인코딩 덕분에 **ScanNet 3D Segmentation mIoU +11.5%p 폭발적 대폭 상승**.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
PointNet++는 포인트 클라우드 상에서 계층적 3D 수용 영역(Receptive Field)을 정립한 3D 비전 분야의 표준 인코더 구조입니다.

### 2. 자율주행 관련 시사점
- **LiDAR 센서 모델링**: 실제 LiDAR는 거리에 따라 포인트 밀도가 달라진다. PointNet++의 MSG/MRG는 이 불균일성을 명시적으로 처리 → 시뮬레이터에서 생성한 균일 밀도 포인트 클라우드와 실제 데이터 사이의 도메인 갭을 줄이는 아이디어로 활용 가능하다.
- **CenterPoint, PointPillars와의 관계**: 두 논문 모두 LiDAR 포인트를 BEV 격자로 변환하여 CNN을 적용한다. PointNet++는 "격자 변환 없이" 직접 포인트를 처리하는 대안으로, 세밀한 기하 정보 보존에 유리하다.
- **세그멘테이션 응용**: Feature Propagation의 보간 + skip connection 아이디어는 이후 3D 의미론적 세그멘테이션(ScanNet 등) 모델의 표준 구조로 정착했다.

### 3. 한계점 및 아쉬운 점
- MSG는 계산 비용이 높음. 저자들도 inference 속도 향상을 향후 과제로 명시 → 이후 PointPillars, CenterPoint가 속도 우선 설계로 실용화.
- FPS의 순차적 최원점 탐색은 병렬화가 어려워 대규모 실시간 포인트 클라우드 처리에는 여전히 부담이 됨.
- Set Abstraction 레벨 수, 반경 등 하이퍼파라미터에 성능이 민감하다는 점은 실무 적용 시 추가 튜닝 부담으로 남음.

---

## 참고 자료
- [PointNet++ 공식 코드 저장소](https://github.com/charlesq34/pointnet2)
- [NIPS 2017 논문 (arXiv:1706.02413)](https://arxiv.org/abs/1706.02413)

*관련 논문: [PointNet](/posts/papers/pointnet-deep-learning-on-point-sets-for-3d-classification-and-segmentation/), [PointPillars](/posts/papers/pointpillars-fast-encoders-object-detection-point-clouds/), [CenterPoint](/posts/papers/centerpoint-center-based-3d-object-detection-and-tracking/)*
