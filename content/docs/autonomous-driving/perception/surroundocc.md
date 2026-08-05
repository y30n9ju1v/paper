---
title: "SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving"
date: 2026-04-19T10:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "3D Occupancy"]
tags: ["3D Occupancy", "Autonomous Driving", "Multi-Camera", "Scene Understanding"]
year: 2023
references:
  - "bevformer"
  - "monoscene-monocular-3d-semantic-scene-completion"
  - "nuscenes-multimodal-dataset-autonomous-driving"
---

## 💡 한 줄 요약
BEV 평면의 높이 축 정보 손실을 막기 위해 3D 공간 쿼리를 2D 멀티카메라 특징 맵에 직접 어텐션하는 **2D-3D Spatial Deformable Cross-Attention**을 설계하고, Poisson Surface Reconstruction 기반 밀집(Dense) GT 자동 생성 파이프라인으로 nuScenes 3D Occupancy 예측 SOTA(20.30 mIoU)를 달성했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, Jiwen Lu (Tsinghua Univ., Tianjin Univ., PhiGent Robotics)
- **발행년도**: 2023 (arXiv:2303.09551, ICCV 2023)
- **주요 기여점**:
  1. **2D-3D Volumetric Spatial Attention**: BEV 평면 대신 3D 볼륨 쿼리 $Q \in \mathbb{R}^{C \times H \times W \times Z}$를 직접 생성하여 높이 정보를 온전히 유지.
  2. **Multi-Scale 3D U-Net & Decayed Loss**: 저해상도부터 고해상도까지 복셀 특징을 점진적으로 정제하는 3D Deconv 피라미드 스택.
  3. **Poisson Surface Reconstruction 기반 Dense GT 파이프라인**: 멀티프레임 동적/정적 LiDAR 포인트를 스티칭 후 메쉬 표면을 생성하여 가려진 영역까지 완전 밀집(Dense) Occupancy GT 구축.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **BEV-based Perception (BEVFormer)**: 3D 정보를 2D BEV 평면으로 사영하여 높이 축(Z-axis) 세부 기하가 손상됨.
2. **Sparse LiDAR Supervision (TPVFormer)**: 희소한 LiDAR 레이블로 학습하여 예측 결과도 듬성듬성한 빈 공간 형태로 생성됨.
3. **SurroundOcc**: 3D 공간 복셀과 밀집 Poisson GT를 이용해 완전한 밀집(Dense) 3D Occupancy 복원.

---

## 📑 목차
- Chapter 1: 2D-3D Volumetric Spatial Attention 수식
- Chapter 2: Multi-Scale 3D U-Net 피라미드 업샘플링
- Chapter 3: Decayed Multi-Scale Occupancy Loss
- Chapter 4: 3D Occupancy IoU & mIoU 평가 산식
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: 2D-3D Volumetric Spatial Attention 수식

### 1. 요약
3D 공간 상의 복셀 쿼리 $q \in Q_{x, y, z}$를 6개 카메라 $m$의 2D 평면으로 사영한 지점 $p_{m}$ 근방에서 Deformable Attention으로 특징을 집계합니다.

### 2. 수식 및 파이썬 코드 설명

$$\text{DeformAttn}(q, p, x) = \sum_{i=1}^{N_{\text{head}}} \mathbf{W}_i \sum_{j=1}^{N_{\text{key}}} \mathcal{A}_{ij} \cdot \mathbf{W}'_i x(p + \Delta p_{ij})$$

- **$q$**: 3D 볼륨 쿼리 벡터 ($x, y, z$)
- **$p$**: 3D 기준점을 2D 이미지 평면으로 사영한 정규화 좌표

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VolumetricSpatialCrossAttention3D(nn.Module):
    """
    SurroundOcc: 3D 볼륨 쿼리와 2D 멀티뷰 이미지 간 Deformable Cross-Attention
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, num_keys: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.num_keys = num_keys
        self.offset_head = nn.Linear(embed_dim, num_heads * num_keys * 2)
        self.weight_head = nn.Linear(embed_dim, num_heads * num_keys)

    def forward(self, query_3d: torch.Tensor, img_feats_2d: torch.Tensor) -> torch.Tensor:
        """
        query_3d: (B, Dx, Dy, Dz, C)
        img_feats_2d: (B, N_cams, C, H, W)
        """
        B, Dx, Dy, Dz, C = query_3d.shape
        N_cams, _, H, W = img_feats_2d.shape[1:]
        q_flat = query_3d.view(B, Dx * Dy * Dz, C)
        num_q = Dx * Dy * Dz

        # 3D 복셀 쿼리로부터 카메라별 참조점 오프셋과 어텐션 가중치 예측
        offsets = self.offset_head(q_flat).view(B, num_q, self.num_heads, self.num_keys, 2)
        weights = torch.softmax(
            self.weight_head(q_flat).view(B, num_q, self.num_heads, self.num_keys), dim=-1
        )

        # 카메라 내적 파라미터로 복셀을 각 카메라 이미지 평면에 투영했다고 가정한
        # 정규화 좌표(실제 구현에서는 3D->2D projection 결과)에 오프셋을 더해 참조점 생성
        base_ref = torch.zeros(B, num_q, 1, 1, 2, device=q_flat.device)
        sample_grid = (base_ref + offsets).clamp(-1, 1)  # (B, num_q, heads, keys, 2)

        # 카메라 1개(대표)에 대해 grid_sample로 이미지 특징을 실제로 샘플링
        img_feat_cam0 = img_feats_2d[:, 0]  # (B, C, H, W)
        grid = sample_grid.view(B, num_q * self.num_heads, self.num_keys, 2)
        sampled = F.grid_sample(img_feat_cam0, grid, align_corners=False)  # (B, C, num_q*heads, keys)
        sampled = sampled.view(B, C, num_q, self.num_heads, self.num_keys).permute(0, 2, 3, 4, 1)

        # 헤드/키 차원에 대해 어텐션 가중합
        attn_out = (sampled * weights.unsqueeze(-1)).sum(dim=(2, 3))  # (B, num_q, C)

        updated_volume = query_3d + attn_out.view(B, Dx, Dy, Dz, C)
        return updated_volume

# --- 사용 예시 ---
q_vol = torch.randn(1, 16, 16, 8, 128)
f_2d = torch.randn(1, 6, 128, 32, 32)
vol_attn = VolumetricSpatialCrossAttention3D()
print("3D Volumetric Attention 출력 Shape:", vol_attn(q_vol, f_2d).shape)
```

---

## 🛠️ Chapter 2: Multi-Scale 3D U-Net 피라미드 정제

### 1. 요약
저해상도 3D 피처 $Y_{j-1}$를 3D Deconvolution 연산으로 2배 업샘플하여 상위 스케일 특징 $F_j$와 합산 결합합니다.

### 2. 수식 및 파이썬 코드 설명

$$Y_j = F_j + \text{Deconv3D}(Y_{j-1})$$

```python
import torch
import torch.nn as nn

class MultiScale3DUnetDecoder(nn.Module):
    """
    SurroundOcc: 저해상도 3D 볼륨 피처를 고해상도로 점진 업샘플 정제
    """
    def __init__(self, in_channels: int = 128, out_channels: int = 128):
        super().__init__()
        self.deconv3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv3d = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, low_res_feat: torch.Tensor, high_res_feat: torch.Tensor) -> torch.Tensor:
        """
        low_res_feat: (B, C, Dx/2, Dy/2, Dz/2)
        high_res_feat: (B, C, Dx, Dy, Dz)
        """
        upsampled = self.deconv3d(low_res_feat)
        fused = self.conv3d(upsampled + high_res_feat)
        return fused

# --- 사용 예시 ---
low_f = torch.randn(1, 128, 8, 8, 4)
high_f = torch.randn(1, 128, 16, 16, 8)
unet_3d = MultiScale3DUnetDecoder()
print("업샘플 정제된 3D Volume Feature Shape:", unet_3d(low_f, high_f).shape)
```

---

## 🛠️ Chapter 3: Decayed Multi-Scale Occupancy Loss

### 1. 요약
피라미드 스케일 $j$ ($0$: 최상위 고해상도 $\dots$ $M-1$: 최하위 저해상도)에 대해 감쇄 가중치 $\alpha_j = \frac{1}{2^j}$를 곱해 총 손실을 계산합니다.

### 2. 수식 및 파이썬 코드 설명

$$\alpha_j = \frac{1}{2^j}$$

$$\mathcal{L}_{total} = \sum_{j=0}^{M-1} \frac{1}{2^j} \mathcal{L}_{occ}(Y_j, \text{GT}_j)$$

```python
import torch
import torch.nn.functional as F

def compute_decayed_multiscale_occupancy_loss(
    multiscale_preds: list, # [Level_0 (High), Level_1, Level_2 (Low)]
    multiscale_gts: list
) -> torch.Tensor:
    """
    SurroundOcc: 감쇄 가중치 alpha_j = 1 / (2^j) 기반 Multi-Scale Occupancy Loss
    """
    total_loss = 0.0
    for j, (pred, gt) in enumerate(zip(multiscale_preds, multiscale_gts)):
        weight = 1.0 / (2.0 ** j)
        loss_j = F.cross_entropy(pred, gt)
        total_loss += weight * loss_j
        
    return total_loss

# --- 사용 예시 ---
preds = [torch.randn(100, 18), torch.randn(100, 18), torch.randn(100, 18)]
gts = [torch.randint(0, 18, (100,)), torch.randint(0, 18, (100,)), torch.randint(0, 18, (100,))]
print("SurroundOcc Decayed Multi-Scale Loss:", compute_decayed_multiscale_occupancy_loss(preds, gts).item())
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. nuScenes 3D Occupancy 예측 성능 리더보드

| 알고리즘 (Method) | 입력 모달리티 | SC (Scene Completion) IoU ↑ | SSC mIoU ↑ |
|---|---|---|---|
| **MonoScene** | RGB (Monocular) | 23.96 | 7.31 |
| **BEVFormer** | RGB (Multi-Camera) | 30.50 | 16.75 |
| **TPVFormer** | RGB (Multi-Camera) | 30.86 | 17.10 |
| **SurroundOcc (Ours)** | **RGB (Multi-Camera)** | **31.49 (+0.63%)** | **20.30 (+3.20%)** |

- **결과**: 3D 볼륨 쿼리와 Poisson 밀집 GT 학습 덕분에 TPVFormer 대비 **SSC mIoU +3.2%p 대폭 향상**.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
SurroundOcc는 2D-3D Volumetric Spatial Attention과 Poisson Surface Reconstruction 밀집 GT 라벨링을 최초 구현하여 카메라 전용 3D Occupancy 예측 성능의 한계를 격상시켰습니다.

### 2. 한계점 및 아쉬운 점
- 단일 프레임 입력만 처리하므로 시간적 연속성(occupancy flow)이 없어 motion prediction에 직접 활용하기 어렵다.
- BEVFormer 대비 latency/memory가 다소 증가하여 실시간성과 정확도 사이의 트레이드오프가 존재한다.
- GT 생성 파이프라인이 멀티프레임 LiDAR 데이터에 의존하므로 LiDAR가 없는 순수 카메라 데이터셋에는 그대로 적용하기 어렵다.

---

## 참고 자료
- [SurroundOcc 공식 GitHub 저장소](https://github.com/weiyithu/SurroundOcc)
- [ICCV 2023 논문 (arXiv:2303.09551)](https://arxiv.org/abs/2303.09551)

*관련 논문: [MonoScene](/posts/papers/monoscene-monocular-3d-semantic-scene-completion/), [TPVFormer](/posts/papers/tpvformer-tri-perspective-view-3d-semantic-occupancy/), [Occ3D](/posts/papers/occ3d-large-scale-3d-occupancy-prediction-benchmark/), [BEVFormer](/posts/papers/bevformer/), [GaussianWorld](/posts/papers/gaussianworld-gaussian-world-model-for-streaming-3d-occupancy-prediction/)*
