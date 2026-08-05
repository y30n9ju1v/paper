---
title: "BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection"
date: 2026-04-17T00:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving"]
tags: ["3D Object Detection", "BEV", "Depth Estimation", "Autonomous Driving"]
year: 2022
references:
  - "lift-splat-shoot"
  - "bevformer"
  - "nuscenes-multimodal-dataset-autonomous-driving"
---

## 💡 한 줄 요약
BEVDepth는 LiDAR 포인트 기반 명시적 깊이 지도학습(Explicit Depth Supervision), 카메라 파라미터 기반 카메라 인식 깊이 추정(Camera-aware Depth Prediction), 그리고 고속 Voxel Pooling 기술을 결합하여 멀티뷰 카메라 3D 객체 탐지에서 신뢰할 수 있는 깊이를 획득하고 nuScenes 카메라 전용 SOTA(60.9% NDS)를 달성했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Yinhao Li, Zheng Ge, Guanyi Yu, Jinrong Yang, Zengran Wang, Yukang Shi, Jianjian Sun, Zeming Li (Chinese Academy of Sciences, MEGVII Technology, HUST)
- **발행년도**: 2022 (arXiv:2206.10092v2, AAAI 2023)
- **주요 기여점**:
  1. **명시적 깊이 지도학습 (Explicit Depth Supervision)**: 탐지 손실만으로 깊이를 간접 학습하는 기존 LSS 방식의 한계를 극복하기 위해 LiDAR 포인트를 Ground-Truth 깊이 이미지로 변환하여 DepthNet을 직접 지도학습.
  2. **카메라 인식 깊이 추정 (Camera-aware Depth Prediction)**: 카메라 내재(Intrinsic) 및 외재(Extrinsic) 파라미터를 MLP로 인코딩한 후 SE(Squeeze-and-Excitation) 모듈을 통해 2D 특징 맵을 동적으로 가중 조정하여 카메라 변경에 강건한 일반화 성능 확보.
  3. **Depth Refinement & Efficient Voxel Pooling**: Frustum 3D 특징 정제 모듈을 추가하고 CUDA 스레드 단위의 고속 렌더링/풀링 알고리즘으로 기존 Lift-Splat 풀링 속도를 80배 가량 향상.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **2D 기반 3D 탐지 (CenterNet3D 등)**: 2D 바운딩 박스에서 깊이/3D 크기를 직접 회귀하나 거리 추정 정확도가 떨어짐.
2. **LSS (Lift-Splat-Shoot)**: 2D 이미지 특징에 이산 깊이 분포(Depth Bin Distribution)를 곱해 3D Frustum으로 확장한 뒤 BEV 평면에 사영하는 표준 프레임워크.
3. **BEVDepth**: LSS 구조의 가장 결정적 병목인 "부정확한 깊이 추정"을 LiDAR 깊이 직접 지도학습 및 카메라 파라미터 임베딩으로 완벽히 정정.

### 기존 한계점
- **부정확한 암묵적 깊이 (Inaccurate Depth)**: 기존 LSS는 3D 탐지 손실만으로 깊이를 학습하므로 픽셀별 실제 깊이 오차(AbsRel)가 0.23에 달할 정도로 조악함.
- **카메라 설정에의 과적합 (Depth Module Over-fitting)**: 해상도나 카메라 포즈가 바뀌면 깊이 예측 모듈의 성능이 폭망함.
- **BEV 특징 왜곡 (Imprecise BEV Semantics)**: 깊이 분포 오차로 인해 2D 이미지가 잘못된 3D/BEV 공간에 사영(Splat)되어 객체 위치 픽셀이 뭉개짐.

---

## 📑 목차
- Chapter 1: LiDAR 포인트의 카메라 2D 투영 및 GT 깊이 생성
- Chapter 2: 카메라 인식 깊이 예측 (Camera-aware DepthNet)
- Chapter 3: 명시적 깊이 BCE 손실 함수
- Chapter 4: Depth Refinement 및 Frustum 특징 생성 (Lift 과정)
- Chapter 5: 고속 Voxel Pooling (Efficient Voxel Pooling)
- Chapter 6: 주요 실험 및 결과
- Chapter 7: 결론 및 시사점

---

## 🛠️ Chapter 1: LiDAR 포인트의 카메라 2D 투영 및 GT 깊이 생성

### 1. 요약
LiDAR의 3D 포인트 $P = (X, Y, Z)$를 각 카메라의 외재 파라미터 $(R_i, t_i)$와 내재 파라미터 $K_i$를 이용하여 $i$번째 카메라 픽셀 좌표 $(u, v)$ 및 거리 $d$로 투영한 뒤, 원-핫 이산 깊이 지도 $D_i^{gt}$를 생성합니다.

### 2. 수식 및 파이썬 코드 설명

$$P_i^{cam} = R_i P + t_i = \begin{bmatrix} X_{cam} \\ Y_{cam} \\ Z_{cam} \end{bmatrix}$$

$$\begin{bmatrix} u \cdot Z_{cam} \\ v \cdot Z_{cam} \\ Z_{cam} \end{bmatrix} = K_i P_i^{cam} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_{cam} \\ Y_{cam} \\ Z_{cam} \end{bmatrix}$$

- **$K_i$**: 카메라 초점거리 $(f_x, f_y)$ 및 주점 $(c_x, c_y)$을 담은 $3 \times 3$ 내재 행렬
- **$R_i, t_i$**: LiDAR 좌표계에서 카메라 좌표계로의 3D 회전 행렬 및 이동 벡터
- **$d = Z_{cam}$**: 해당 픽셀 맺힘 지점에서의 실제 카메라 기준 수직 깊이값

```python
import torch

def project_lidar_to_camera_gt_depth(
    points_3d: torch.Tensor, # (N, 3) LiDAR 3D 포인트 클라우드
    R_w2c: torch.Tensor,     # (3, 3) 회전 행렬
    t_w2c: torch.Tensor,     # (3,) 이동 벡터
    K_intrinsics: torch.Tensor, # (3, 3) 카메라 내재 행렬
    img_shape: tuple,        # (H, W) 이미지 크기
    depth_bins: torch.Tensor # (D,) 이산 깊이 구간 경계 (예: 2m ~ 58m)
) -> torch.Tensor:
    """
    LiDAR 3D 포인트를 2D 카메라 평면으로 투영하여 이산 깊이 GT 맵 D_gt 생성
    """
    H, W = img_shape
    D = len(depth_bins) - 1
    
    # 1. LiDAR -> 카메라 좌표계 변환: P_cam = R @ P + t
    pts_cam = (R_w2c @ points_3d.T).T + t_w2c # (N, 3)
    
    # z > 0 (카메라 전방에 있는 포인트만 선별)
    valid_mask = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid_mask]
    
    # 2. 2D 원근 투영: p_img = K @ P_cam
    pts_2d_homo = (K_intrinsics @ pts_cam.T).T # (N, 3)
    depths = pts_2d_homo[:, 2]                 # 실제 깊이 Z_cam
    u = (pts_2d_homo[:, 0] / depths).long()     # 픽셀 u (열)
    v = (pts_2d_homo[:, 1] / depths).long()     # 픽셀 v (행)
    
    # 3. 이미지 영역 내부 필터링
    img_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, depths = u[img_mask], v[img_mask], depths[img_mask]
    
    # 4. 이산 깊이 Bin 버킷 할당 (One-hot 인코딩)
    depth_indices = torch.bucketize(depths, depth_bins) - 1
    valid_bin_mask = (depth_indices >= 0) & (depth_indices < D)
    
    u, v, depth_indices = u[valid_bin_mask], v[valid_bin_mask], depth_indices[valid_bin_mask]
    
    gt_depth_map = torch.zeros((H, W, D), dtype=torch.float32)
    gt_depth_map[v, u, depth_indices] = 1.0  # One-hot 라벨링
    
    return gt_depth_map

# --- 사용 예시 ---
pts_dummy = torch.randn(500, 3) + torch.tensor([0.0, 0.0, 10.0])
R_dummy = torch.eye(3)
t_dummy = torch.zeros(3)
K_dummy = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
bins_dummy = torch.linspace(2.0, 58.0, 11)
gt_map = project_lidar_to_camera_gt_depth(pts_dummy, R_dummy, t_dummy, K_dummy, (480, 640), bins_dummy)
print("생성된 GT Depth Map Shape:", gt_map.shape)
```

---

## 🛠️ Chapter 2: 카메라 인식 깊이 예측 (Camera-aware DepthNet)

### 1. 요약
각 카메라의 내재/외재 파라미터를 MLP로 인코딩한 벡터 $E_i$를 생성하고, 이를 Squeeze-and-Excitation(SE) 구조를 이용해 2D 이미지 특징 맵 $F_i^{2D}$의 채널 가중치로 곱해줌으로써 카메라 구도 변경에 영향받지 않는 가우시안 깊이 신경망을 구현합니다.

### 2. 수식 및 파이썬 코드 설명

$$E_i = \text{MLP}\Big(\text{Concat}\big( \text{Flatten}(R_i), \text{Flatten}(t_i), \text{Flatten}(K_i) \big)\Big)$$

$$F_i^{se} = F_i^{2D} \odot \sigma\Big( \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 E_i) \Big)$$

$$D_i^{pred} = \text{Softmax}\Big( \text{Conv2D}(F_i^{se}) \Big) \in \mathbb{R}^{D_{bins} \times H \times W}$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraAwareDepthNet(nn.Module):
    """
    카메라 파라미터(Extrinsics & Intrinsics) 임베딩 기반 Camera-aware SE DepthNet
    """
    def __init__(self, in_channels: int = 256, num_depth_bins: int = 11, param_dim: int = 3*3 + 3 + 3*3):
        super().__init__()
        # 1. 카메라 파라미터 임베딩 MLP
        self.param_mlp = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Linear(64, in_channels)
        )
        # 2. SE (Squeeze-and-Excitation) 채널 가중치 생성기
        self.se_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
        # 3. 이산 깊이 확률 헤드
        self.depth_head = nn.Conv2d(in_channels, num_depth_bins, kernel_size=1)

    def forward(
        self,
        img_feat: torch.Tensor, # (B, C, H, W) 2D 이미지 특징
        R: torch.Tensor,        # (B, 3, 3) 회전
        t: torch.Tensor,        # (B, 3) 이동
        K: torch.Tensor         # (B, 3, 3) 내재
    ) -> torch.Tensor:
        B, C, H, W = img_feat.shape
        
        # 1. 카메라 파라미터 직렬화 및 임베딩
        cam_params = torch.cat([R.view(B, -1), t.view(B, -1), K.view(B, -1)], dim=-1)
        param_emb = self.param_mlp(cam_params) # (B, C)
        
        # 2. SE 채널 가중치 곱 (Spatial Broadcasting)
        se_weights = self.se_fc(param_emb).view(B, C, 1, 1)
        modulated_feat = img_feat * se_weights
        
        # 3. Softmax 확률 분포 깊이 출력 (B, D_bins, H, W)
        depth_logits = self.depth_head(modulated_feat)
        depth_pred = F.softmax(depth_logits, dim=1)
        
        return depth_pred

# --- 사용 예시 ---
feat_in = torch.randn(2, 256, 32, 32)
R_in, t_in, K_in = torch.eye(3).repeat(2, 1, 1), torch.zeros(2, 3), torch.eye(3).repeat(2, 1, 1)
depth_net = CameraAwareDepthNet()
print("예측된 이산 깊이 확률 분포 Shape:", depth_net(feat_in, R_in, t_in, K_in).shape)
```

---

## 🛠️ Chapter 3: 명시적 깊이 지도 손실 함수 (BCE Depth Loss)

### 1. 요약
예측된 이산 깊이 확률 분포 $D^{pred}$와 LiDAR 기반 GT 원-핫 깊이 지도 $D^{gt}$ 사이의 Binary Cross Entropy 손실을 계산해 명시적으로 지도학습합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{L}_{depth} = -\frac{1}{H \cdot W} \sum_{u=1}^H \sum_{v=1}^W \sum_{d=1}^D \left[ D^{gt}(u,v,d) \log D^{pred}(u,v,d) + (1-D^{gt}) \log(1-D^{pred}) \right]$$

```python
import torch
import torch.nn.functional as F

def compute_explicit_depth_loss(
    depth_pred: torch.Tensor, # (B, D_bins, H, W) 예측 깊이 확률
    depth_gt: torch.Tensor    # (B, H, W, D_bins) One-hot GT 깊이
) -> torch.Tensor:
    """
    BEVDepth의 명시적 깊이 지도 손실 (BCE Loss)
    """
    # GT를 (B, D_bins, H, W) 형태로 permute
    gt_permuted = depth_gt.permute(0, 3, 1, 2)
    
    # 0인 영역(LiDAR 미투영 픽셀) 무시 마스크 생성
    mask = (gt_permuted.sum(dim=1, keepdim=True) > 0).float()
    
    # Binary Cross Entropy 계산
    bce_loss = F.binary_cross_entropy(depth_pred, gt_permuted, reduction='none')
    
    masked_loss = (bce_loss * mask).sum() / (mask.sum() * depth_pred.shape[1] + 1e-6)
    return masked_loss

# --- 사용 예시 ---
d_p = F.softmax(torch.randn(1, 10, 32, 32), dim=1)
d_g = torch.zeros(1, 32, 32, 10)
d_g[0, 10, 10, 3] = 1.0 # 샘플 GT 지정
print("명시적 깊이 BCE Loss:", compute_explicit_depth_loss(d_p, d_g).item())
```

---

## 🛠️ Chapter 4: Depth Refinement 및 Frustum 특징 생성 (Lift 과정)

### 1. 요약
2D 이미지 특징 $F^{2D} \in \mathbb{R}^{C_{ctx} \times H \times W}$와 깊이 분포 $D^{pred} \in \mathbb{R}^{D \times H \times W}$의 외적(Outer Product)을 통해 4D Frustum 특징 $V \in \mathbb{R}^{D \times C_{ctx} \times H \times W}$을 생성하는 LSS의 Lift 연산을 수행합니다.

### 2. 수식 및 파이썬 코드 설명

$$V(d, c, u, v) = D^{pred}(d, u, v) \cdot F^{2D}(c, u, v)$$

```python
import torch

def lift_image_features_to_frustum(
    depth_prob: torch.Tensor, # (B, D, H, W) 깊이 확률 분포
    context_feat: torch.Tensor # (B, C, H, W) 2D 컨텍스트 특징
) -> torch.Tensor:
    """
    LSS Lift 단계: 2D 이미지 특징을 깊이 확률 분포로 3D Frustum 특징으로 확산
    V = Depth (B, D, 1, H, W) * Context (B, 1, C, H, W) -> (B, D, C, H, W)
    """
    B, D, H, W = depth_prob.shape
    C = context_feat.shape[1]
    
    frustum_feat = depth_prob.unsqueeze(2) * context_feat.unsqueeze(1)
    return frustum_feat # (B, D, C, H, W)

# --- 사용 예시 ---
prob = F.softmax(torch.randn(1, 10, 16, 16), dim=1)
ctx = torch.randn(1, 64, 16, 16)
print("생성된 3D Frustum 특징 Shape:", lift_image_features_to_frustum(prob, ctx).shape)
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. nuScenes 테스트셋 리더보드 성능 비교

| 방법 (Method) | 입력 모달리티 | mAP ↑ | NDS ↑ | mATE (위치오차) ↓ | mAVE (속도오차) ↓ |
|---|---|---|---|---|---|
| **BEVDet** | Camera | 0.422 | 0.463 | 0.629 | 0.852 |
| **BEVFormer** | Camera | 0.481 | 0.569 | 0.593 | 0.380 |
| **PETRv2** | Camera | 0.490 | 0.582 | 0.561 | 0.342 |
| **BEVDepth (ResNet101)** | Camera | **0.503** | **0.600** | **0.445** | **0.261** |
| **BEVDepth† (ConvNeXt)** | Camera | **0.520** | **0.609** | **0.421** | **0.245** |

- **결과**: 명시적 깊이 지도학습 및 카메라 파라미터 연동 덕분에 위치 오차(mATE **0.421**)가 대폭 감소하며 **60.9% NDS로 카메라 전용 최고 성능(1위)** 달성.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
BEVDepth는 LSS 계열 3D 객체 탐지 모델의 고질적 약점이었던 "부정확한 암묵적 깊이"를 LiDAR 명시적 지도학습과 카메라 인식 SE 모듈로 완벽히 해결하여 BEV 멀티뷰 인식의 새로운 기준을 제시했습니다.

### 2. 실무적 시사점
- **LiDAR-Camera 자원 융합**: 배포 시에는 카메라만 사용하더라도, 학습 단계에서 LiDAR를 GT 깊이 생성기로 100% 활용하는 효율적 시스템 설계 기틀 마련.

### 3. 한계점 및 아쉬운 점
- 학습 시 LiDAR ground-truth 깊이가 필수적이라 순수 카메라 전용 데이터셋에는 직접 적용하기 어렵다.
- 카메라만으로는 원거리·저조도 환경에서 깊이 불확실성이 여전히 남아있어, LiDAR 기반 방법과의 근본적 격차를 완전히 좁히지는 못했다.

---

## 참고 자료
- [BEVDepth 공식 GitHub 저장소](https://github.com/Megvii-BaseDetection/BEVDepth)
- [AAAI 2023 논문 (arXiv:2206.10092)](https://arxiv.org/abs/2206.10092)

*관련 논문: [Lift Splat Shoot](/posts/papers/lift-splat-shoot/), [BEVFormer](/posts/papers/bevformer/), [BEVFusion](/posts/papers/bevfusion-multi-task-multi-sensor-fusion/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
