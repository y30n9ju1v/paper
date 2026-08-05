---
title: "Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving"
date: 2026-04-17T09:10:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "3D Occupancy", "Benchmark & Dataset"]
tags: ["Autonomous Driving", "3D Occupancy", "LiDAR", "Dataset", "Benchmark"]
year: 2023
references:
  - "bevformer"
  - "nuscenes-multimodal-dataset-autonomous-driving"
  - "waymo-open-dataset"
---

## 💡 한 줄 요약
자율주행 서라운드뷰 3D Occupancy 예측을 위한 대규모 고해상도 벤치마크 데이터셋 **Occ3D (nuScenes & Waymo)**를 반자동 3단계 생성 파이프라인으로 구축하고, 증분 토큰 선택(Incremental Token Selection) 기반의 **Coarse-to-Fine 3D Occupancy Network (CTF-Occ)**를 제안하여 28.53 mIoU SOTA를 달성했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, Hang Zhao (Tsinghua Univ., USC, Shanghai AI Lab)
- **발행년도**: 2023 (NeurIPS 2023 Datasets & Benchmarks Track, arXiv:2304.14365)
- **주요 기여점**:
  1. **최초의 대규모 서라운드뷰 3D Occupancy 벤치마크 (Occ3D)**: nuScenes(40,000 프레임, 18개 클래스) 및 Waymo(200,000 프레임, 15개 클래스) 기반 고해상도 3D 복셀 GT 데이터 구축.
  2. **반자동 3단계 3D Occupancy GT 생성 파이프라인**: (1) 동적/정적 분리 멀티프레임 복셀 밀도화(Voxel Densification), (2) Ray Casting 폐색 추론(Occlusion Reasoning), (3) 2D 시맨틱 마스크 기반 복셀 정제(Image-guided Refinement).
  3. **3D-2D 일관성 품질 검증 지표**: 3D 복셀을 2D 이미지 평면으로 역투영하여 사람 주석 2D 시맨틱 세그멘테이션과 비교하는 자동 검증 모듈 제안 (최종 58.50 mIoU 달성).
  4. **CTF-Occ (Coarse-to-Fine Occupancy Network)**: 불확실한 복셀 토큰 Top-K개만 선별하는 증분 토큰 선택과 3D Deformable Cross-Attention으로 28.53 mIoU 달성.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **3D Object Detection (Bounding Box)**: 고정 카테고리(차량, 보행자) 이외의 일반 야외 장애물(General Objects)이나 불규칙 기하 형태(크레인, 쓰레기통 등)를 전혀 표현하지 못함.
2. **Semantic Scene Completion (NYUv2, SemanticKITTI)**: 실내 전용이거나 전방 서라운드뷰 360도 전체를 다루지 못함.
3. **Occ3D**: 서라운드뷰 Multi-Camera 자율주행 표준 3D Occupancy Benchmark 수립.

---

## 📑 목차
- Chapter 1: 반자동 3D Occupancy GT 생성 알고리즘
- Chapter 2: Ray Casting 폐색 추론 및 3D-2D 일관성 지표
- Chapter 3: CTF-Occ 네트워크 (Coarse-to-Fine & Incremental Token Selection)
- Chapter 4: OHEM (Online Hard Example Mining) Occupancy Loss
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: 반자동 3D Occupancy GT 생성 파이프라인

### 1. 요약
희소한 LiDAR 스캔을 동적 객체(Dynamic Object)와 정적 장면(Static Scene)으로 분리하여 멀티프레임 집계 후 VDBFusion 메쉬 재구성을 거쳐 복셀 밀도화(Densification)를 수행합니다.

---

## 🛠️ Chapter 2: Ray Casting 폐색 추론 및 3D-2D 일관성 수식

### 1. 요약
카메라/LiDAR 원점 $\mathbf{o}$에서 복셀 중심 $v$ 방향으로 광선 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$을 투사해 가시성 마스크(Visibility Mask)를 산출하고, 2D 시맨틱 주석과의 3D-2D 일관성 IoU를 측정합니다.

### 2. 수식 및 파이썬 코드 설명

$$\text{Vis}(v) = \begin{cases} \text{Occupied} & \text{if } \exists t, \ \mathbf{r}(t) \in v \text{ and reflects LiDAR} \\ \text{Free} & \text{if } \forall t, \ \mathbf{r}(t) \in v \text{ passes through} \\ \text{Unobserved} & \text{otherwise} \end{cases}$$

$$\text{IoU}_{3D-2D} = \frac{TP}{TP + FP + FN} = \frac{\sum_i \mathbb{I}\Big( \text{RayCast3D}(i) = c \ \land \ \text{GT}_{2D}(i) = c \Big)}{\sum_i \mathbb{I}\Big( \text{RayCast3D}(i) = c \ \lor \ \text{GT}_{2D}(i) = c \Big)}$$

```python
import torch

def compute_3d_2d_consistency_iou(
    raycasted_3d_classes: torch.Tensor, # (N_pixels,) 3D 복셀에서 2D 픽셀로 Ray-cast된 시맨틱 클래스
    gt_2d_classes: torch.Tensor,        # (N_pixels,) 사람이 주석한 2D GT 시맨틱 클래스
    num_classes: int = 18
) -> float:
    """
    Occ3D: 데이터셋 품질 검증을 위한 3D-2D Consistency IoU 계산
    """
    valid_mask = (gt_2d_classes >= 0) & (gt_2d_classes < num_classes)
    pred = raycasted_3d_classes[valid_mask]
    gt = gt_2d_classes[valid_mask]
    
    ious = []
    for c in range(num_classes):
        tp = ((pred == c) & (gt == c)).sum().float()
        fp = ((pred == c) & (gt != c)).sum().float()
        fn = ((pred != c) & (gt == c)).sum().float()
        
        union = tp + fp + fn
        if union > 0:
            ious.append((tp / union).item())
            
    return sum(ious) / len(ious) if len(ious) > 0 else 0.0

# --- 사용 예시 ---
p_ray = torch.randint(0, 18, (10000,))
g_2d = torch.randint(0, 18, (10000,))
print("Occ3D 3D-2D Consistency mIoU:", compute_3d_2d_consistency_iou(p_ray, g_2d))
```

---

## 🛠️ Chapter 3: CTF-Occ 증분 토큰 선택 & Deformable Cross-Attention

### 1. 요약
3D 복셀 중 대다수인 빈 공간(Free Voxel) 연산을 배제하기 위해 이진 분류기(Binary Classifier)로 불확실한 복셀 토큰 Top-K개만 선별하여 2D 이미지 특징과 3D Deformable Cross-Attention을 수행합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{T}_{\text{topk}} = \text{TopK}_{v \in V_{\text{voxel}}}\Big( \sigma\big( \text{MLP}(F_v) \big) \Big)$$

$$\hat{F}_v = \text{DeformableCrossAttn}\Big( F_v, \ \pi(v), \ \mathcal{F}_{2D} \Big) \quad (v \in \mathcal{T}_{\text{topk}})$$

```python
import torch
import torch.nn as nn

class IncrementalTokenSelector(nn.Module):
    """
    CTF-Occ: 연산 가속을 위한 불확실성 기준 Top-K 복셀 토큰 선별기
    """
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.binary_classifier = nn.Linear(embed_dim, 1)

    def forward(self, voxel_feats: torch.Tensor, top_k: int = 5000) -> tuple:
        """
        voxel_feats: (B, N_voxels, C)
        """
        B, N_v, C = voxel_feats.shape
        logits = self.binary_classifier(voxel_feats).squeeze(-1) # (B, N_voxels)
        probs = torch.sigmoid(logits)
        
        # 엔트로피 기반 불확실성 산출: H(p) = -p log(p) - (1-p) log(1-p)
        uncertainty = -probs * torch.log(probs + 1e-6) - (1.0 - probs) * torch.log(1.0 - probs + 1e-6)
        
        # Top-K 불확실 복셀 선택
        _, topk_indices = torch.topk(uncertainty, k=top_k, dim=-1) # (B, top_k)
        
        selected_feats = torch.gather(voxel_feats, 1, topk_indices.unsqueeze(-1).repeat(1, 1, C))
        return selected_feats, topk_indices

# --- 사용 예시 ---
v_f = torch.randn(2, 50000, 128)
selector = IncrementalTokenSelector()
s_f, idx = selector(v_f, top_k=5000)
print("선택된 Top-K 복셀 특징 Shape:", s_f.shape)
```

---

## 🛠️ Chapter 4: OHEM Weighted Occupancy Loss

### 1. 요약
손실값이 높은 난이도 상위 복셀 샘플(Hard Examples)만 온라인 마이닝하여 클래스 불균형을 극복하는 OHEM Cross-Entropy Loss를 적용합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{L}_{occ} = -\frac{1}{N_{\text{hard}}} \sum_{i \in \text{HardSamples}} w_{c_i} \log P(y_i = c_i \mid X)$$

```python
import torch
import torch.nn.functional as F

def ohem_weighted_occupancy_loss(
    pred_logits: torch.Tensor, # (N_voxels, C_classes)
    gt_labels: torch.Tensor,   # (N_voxels,)
    hard_ratio: float = 0.25   # 상위 25% 난이도 복셀만 손실 역전파
) -> torch.Tensor:
    """
    CTF-Occ: Online Hard Example Mining (OHEM) Occupancy Loss
    """
    loss_all = F.cross_entropy(pred_logits, gt_labels, reduction='none') # (N_voxels,)
    
    num_hard = int(len(loss_all) * hard_ratio)
    topk_losses, _ = torch.topk(loss_all, k=num_hard)
    
    return topk_losses.mean()

# --- 사용 예시 ---
p_log = torch.randn(10000, 18)
g_lab = torch.randint(0, 18, (10000,))
print("OHEM Weighted Occupancy Loss:", ohem_weighted_occupancy_loss(p_log, g_lab).item())
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. Occ3D-nuScenes 벤치마크 3D Occupancy 예측 리더보드

| 알고리즘 (Method) | 입력 모달리티 | 복셀 표현 | mIoU ↑ | Traffic Cone IoU ↑ |
|---|---|---|---|---|
| **MonoScene** | Camera | Voxel | 7.31 | 1.20 |
| **BEVDet** | Camera | BEV Flatten | 21.50 | 5.21 |
| **BEVFormer** | Camera | BEV Grid | 26.88 | 12.10 |
| **OccFormer** | Camera | 3D Voxel | 27.10 | 14.50 |
| **CTF-Occ (Ours)** | **Camera** | **Coarse-to-Fine Voxel** | **28.53 (+1.65%)** | **22.33 (+10.23%)** |

- **결과**: 복셀 높이 축을 압축하지 않는 3D Coarse-to-Fine 구조와 OHEM 학습 덕분에 교통 콘(Traffic Cone) 등 미세 장애물 탐지 성능 **+10.2%p 폭발적 향상**.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
Occ3D는 자율주행 3D 인지의 패러다임을 바운딩 박스 탐지에서 서라운드뷰 3D Occupancy 예측으로 확장한 이정표 벤치마크이자, 고속 CTF-Occ 네트워크를 동시 제공하는 세미날 연구입니다.

### 2. 한계점 및 아쉬운 점
- **센서 캘리브레이션 오차**: LiDAR-카메라 간 정밀 캘리브레이션과 멀티-프레임 집계 정확도가 서로 강하게 의존한다.
- **동적 변형 객체**: 달리는 동물이나 팔을 흔드는 사람 등 강체(rigid body) 가정을 충족하지 못하는 객체는 모션 블러가 발생한다.
- **General Objects 한계**: nuScenes와 Waymo 데이터셋 모두 제한된 카테고리만 주석되어 있어, 쓰레기통이나 교통 콘 같은 객체는 GO 클래스로 뭉뚱그려지며 더 세밀한 인간 주석이 필요하다.

---

## 참고 자료
- [Occ3D 공식 GitHub 저장소](https://github.com/Tsinghua-MARS-Lab/Occ3D)
- [NeurIPS 2023 논문 (arXiv:2304.14365)](https://arxiv.org/abs/2304.14365)

*관련 논문: [MonoScene](/posts/papers/monoscene-monocular-3d-semantic-scene-completion/), [TPVFormer](/posts/papers/tpvformer-tri-perspective-view-3d-semantic-occupancy/), [SurroundOcc](/posts/papers/surroundocc/), [BEVFormer](/posts/papers/bevformer/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/), [Waymo Open Dataset](/posts/papers/waymo-open-dataset/)*
