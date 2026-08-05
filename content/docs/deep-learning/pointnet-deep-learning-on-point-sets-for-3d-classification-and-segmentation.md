---
title: "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
date: 2026-04-19T22:56:00+09:00
draft: false
categories: ["Papers", "Computer Vision"]
tags: ["Point Cloud", "3D Classification", "3D Segmentation", "Deep Learning"]
year: 2017
references: []
---

## 💡 한 줄 요약
3D 포인트 클라우드를 복셀(Voxel)이나 2D 렌더링 이미지로 전환하는 손실 전처리 없이, **공유 MLP(Shared MLP)**와 **MaxPooling 대칭 함수(Symmetric Function)**의 조합으로 순열 불변성(Permutation Invariance)을 보장하며 원시 3D 점 집합을 직접 학습하는 최초의 딥러닝 아키텍처이다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas (Stanford University)
- **발행년도**: 2017 (CVPR 2017)
- **주요 기여점**:
  1. **순열 불변성 대칭 집계 (Symmetric Aggregator)**: 입력 점 순서 $N!$가지 조합에 관계없이 고정 길이 특징을 추출하는 Max-Pooling 대칭 연산 도입.
  2. **T-Net (Spatial Transformation Network)**: 3D 기하 변환 및 64D 피처 변환 행렬 $A$를 스스로 예측해 정규 자세(Canonical Pose) 정렬.
  3. **직교성 정규화 손실 (Orthogonality Regularization)**: 피처 변환 행렬의 정보 손실을 막는 $\| I - AA^T \|_F^2$ 정규화 손실 추가.
  4. **Universal Approximation 이론적 입증**: 뼈대 점 집합(Critical Point Set $\mathcal{C}_S$)을 통해 어떠한 연속 집합 함수도 근사 가능함을 증명.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **Volumetric 3D CNN (VoxNet)**: 포인트 클라우드를 $32 \times 32 \times 32$ 3D 복셀 격자로 변환하여 메모리 폭발 및 해상도 양자화 오차(Quantization Artifacts) 발생.
2. **Multi-View CNN (MVCNN)**: 3D 메시를 여러 각도의 2D 이미지로 렌더링한 후 2D CNN을 투입하므로 3D 공간 상 분할(Segmentation) 확장 한계.
3. **PointNet**: 원시 포인트 집합 $(N \times 3)$을 그대로 입력받아 $\mathcal{O}(N)$ 연산으로 분류 및 분할 동시 달성.

---

## 📑 목차
- Chapter 1: 대칭 함수 (Symmetric Function) & Max-Pooling 수식
- Chapter 2: T-Net (Joint Alignment Network) 수식
- Chapter 3: Feature Transform 직교성 정규화 손실 (Orthogonality Regularization)
- Chapter 4: Local-Global Feature Concatenation 분할 신경망
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: 대칭 함수 (Symmetric Function) & Max-Pooling 수식

### 1. 요약
순서 없는 집합 $S = \{x_1, \ldots, x_n\}$의 어떤 순열 조합에도 불변인 함수 $f(S)$를 공유 MLP $h$와 대칭 집계 함수 $g = \text{MaxPool}$의 합성으로 유도합니다.

### 2. 수식 및 파이썬 코드 설명

$$f(\{x_1, \ldots, x_n\}) \approx g\left( \text{MaxPool}_{i=1 \dots n} \left( h(x_i) \right) \right)$$

- **$x_i \in \mathbb{R}^3$**: $i$번째 3D 점 좌표 $(x, y, z)$
- **$h: \mathbb{R}^3 \to \mathbb{R}^K$**: 공유 MLP (1D Convolution)
- **$\text{MaxPool}$**: $n$개 점에 대해 각 채널 축별 최대값을 선택하는 대칭 연산

```python
import torch
import torch.nn as nn

class PointNetSymmetricAggregator(nn.Module):
    """
    PointNet 핵심: Shared MLP (1D Conv) + Max-Pooling 대칭 집계기
    """
    def __init__(self, in_dim: int = 3, out_dim: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, out_dim, 1),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C=3, N_pts)
        """
        feats = self.mlp(x) # (B, 1024, N_pts)
        # Max-Pooling over N_pts -> 순열 불변 전역 특징 벡터
        global_feature, _ = torch.max(feats, dim=-1) # (B, 1024)
        return global_feature

# --- 사용 예시 ---
pts = torch.randn(2, 3, 1024) # Batch=2, 3D 좌표, 1024개 점
aggregator = PointNetSymmetricAggregator()
print("PointNet 전역 특징 벡터 Shape:", aggregator(pts).shape)
```

---

## 🛠️ Chapter 2: T-Net (Joint Alignment Network) 수식

### 1. 요약
입력 좌표나 고차원 특징 공간에서 데이터의 회전/이동 불변성을 확보하기 위해 $K \times K$ 정규 자세(Canonical Pose) 아핀 변환 행렬 $A$를 예측해 곱해줍니다.

### 2. 수식 및 파이썬 코드 설명

$$A = \text{T-Net}(X) \in \mathbb{R}^{K \times K}, \quad X_{\text{aligned}} = X \cdot A$$

```python
import torch
import torch.nn as nn

class TNet(nn.Module):
    """
    PointNet T-Net: K x K 정규 자세 공간 정렬 변환 행렬 예측기
    """
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, k * k)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        feat = self.mlp(x)
        global_feat, _ = torch.max(feat, dim=-1)
        matrix = self.fc(global_feat).view(B, self.k, self.k)
        
        # Identity 행렬 더해 초기 안정성 확보
        identity = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        return matrix + identity

# --- 사용 예시 ---
tnet_3d = TNet(k=3)
pts_raw = torch.randn(2, 3, 1024)
trans_mat = tnet_3d(pts_raw)
print("예측된 3x3 변환 행렬 Shape:", trans_mat.shape)
```

---

## 🛠️ Chapter 3: Feature Transform 직교성 정규화 손실 수식

### 1. 요약
64차원 이상의 고차원 특징 변환 행렬 $A$가 정보를 왜곡하거나 소실시키지 않고 순수 강체 회전에 가깝도록 $AA^T = I$ 직교성을 유도합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{L}_{\text{reg}} = \| I - A A^T \|_F^2$$

- **$A \in \mathbb{R}^{K \times K}$**: T-Net이 출력한 $K \times K$ 변환 행렬
- **$\| \cdot \|_F$**: Frobenius Norm (모든 행렬 원소 제곱합)

```python
import torch

def compute_orthogonality_regularization_loss(A: torch.Tensor) -> torch.Tensor:
    """
    PointNet Feature Transform Matrix Orthogonality Loss: || I - A @ A^T ||_F^2
    """
    B, K, _ = A.shape
    identity = torch.eye(K, device=A.device).unsqueeze(0).repeat(B, 1, 1)
    A_A_T = torch.matmul(A, A.transpose(1, 2))
    
    loss_reg = torch.mean(torch.norm(identity - A_A_T, dim=(1, 2)) ** 2)
    return loss_reg

# --- 사용 예시 ---
matrix_64d = torch.randn(2, 64, 64)
print("PointNet Feature Transform 정규화 손실:", compute_orthogonality_regularization_loss(matrix_64d).item())
```

---

## 🛠️ Chapter 4: Local-Global Feature Concatenation 분할 신경망

### 1. 요약
포인트별 파트/장면 분할(Part/Semantic Segmentation)을 위해 64차원 국소 피처 $h^{\text{loc}}(x_i)$와 Max-Pooling 1024차원 전역 피처 $g(S)$를 각 점마다 결합(Concatenate)하여 1088차원 피처로 개별 분류합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathbf{f}_{\text{seg\_i}} = \text{Concat}\big( h^{\text{loc}}(x_i), \ g(S) \big) \in \mathbb{R}^{64 + 1024}$$

```python
import torch
import torch.nn as nn

class PointNetSegmentationHead(nn.Module):
    """
    PointNet Segmentation: Local (64d) + Global (1024d) Feature 결합 분류기
    """
    def __init__(self, num_classes: int = 13):
        super().__init__()
        self.conv_seg = nn.Sequential(
            nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        """
        local_feat: (B, 64, N_pts)
        global_feat: (B, 1024)
        """
        B, _, N_pts = local_feat.shape
        g_expand = global_feat.unsqueeze(-1).repeat(1, 1, N_pts) # (B, 1024, N_pts)
        
        fused = torch.cat([local_feat, g_expand], dim=1) # (B, 1088, N_pts)
        seg_logits = self.conv_seg(fused) # (B, Num_classes, N_pts)
        return seg_logits

# --- 사용 예시 ---
f_loc = torch.randn(2, 64, 1024)
f_glob = torch.randn(2, 1024)
seg_head = PointNetSegmentationHead(num_classes=13)
print("PointNet 점별 분할 Logits Shape:", seg_head(f_loc, f_glob).shape)
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. ModelNet40 3D 물체 분류 벤치마크 비교

| 알고리즘 (Method) | 입력 데이터 폼 | 3D 표면 표현 방식 | 분류 정확도 (Accuracy) ↑ | 처리 속도 (FLOPs) ↓ |
|---|---|---|---|---|
| **VoxNet** | 3D Voxel | $32 \times 32 \times 32$ Grid | 85.9% | - |
| **Subvolume** | 3D Voxel | Sub-Voxel Grid | **89.2%** | 3,633M (파라미터 16.6M, 고비용) |
| **MVCNN** | Multi-View Image | 2D 렌더링 이미지 | **90.1%** | 62,057M (파라미터 60.0M, 최고비용) |
| **PointNet (Ours)** | **Raw Point Cloud** | **Unordered Point Set** | **89.2% (3D 입력 중 최고)** | **440M (파라미터 3.5M, 저비용 초고속)** |

- **결과**: 복셀 데이터 폭발 없이 **89.2% 분류 정확도**로 3D 입력 방법 중 최고 성능(멀티뷰 렌더링 방식인 MVCNN의 90.1%에는 근소하게 못 미침) 및 초당 100만 포인트 실시간 추론 속도 확보.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
PointNet은 순서가 없는 3D 점 집합을 직접 딥러닝 공간으로 인코딩하는 백본의 효시로서, PointPillars, CenterPoint, BEVFusion 등 자율주행 perception 스택의 기초를 닦은 필독 논문입니다.

### 2. 자율주행 관련 시사점
VoxelNet은 각 복셀 내 점들에 PointNet 인코더를 적용하고, PointPillars는 수직 기둥(pillar)을 PointNet으로 인코딩한다. BEVFusion, CenterPoint 등 현대 LiDAR 탐지기의 포인트 인코딩 설계 철학이 이 논문에서 출발했다. 합성 데이터(Blensor 시뮬레이터)로 생성된 부분 스캔에서도 성능 저하 5.3%로 견고하여 도메인 갭 연구에도 시사점을 제공한다.

### 3. 한계점 및 아쉬운 점
- 지역 이웃 구조를 명시적으로 모델링하지 않음 → PointNet++에서 FPS + Ball Query로 해결.
- 대규모 장면 처리 시 점 수 증가에 따른 연산 부담.
- T-Net의 정규화 손실(0.001 가중치)이 실제로 강체 변환 불변성을 얼마나 견고하게 보장하는지에 대한 정량적 분석이 다소 제한적이라는 점은 아쉬움.

---

## 참고 자료
- [PointNet 공식 TensorFlow / PyTorch 구현](https://github.com/charlesq34/pointnet)
- [CVPR 2017 논문 (arXiv:1612.00593)](https://arxiv.org/abs/1612.00593)

*관련 논문: [PointPillars](/posts/papers/pointpillars-fast-encoders-object-detection-point-clouds/), [CenterPoint](/posts/papers/centerpoint-center-based-3d-object-detection-and-tracking/), [BEVFusion](/posts/papers/bevfusion-multi-task-multi-sensor-fusion/)*
