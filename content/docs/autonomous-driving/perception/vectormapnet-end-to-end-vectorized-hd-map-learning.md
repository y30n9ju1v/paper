---
title: "VectorMapNet: End-to-end Vectorized HD Map Learning"
date: 2026-04-20T20:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "HD Map"]
tags: ["Autonomous Driving", "HD Map", "BEV", "Transformer", "DETR", "Polyline"]
year: 2022
references:
  - "detr-end-to-end-object-detection-with-transformers"
  - "nuscenes-multimodal-dataset-autonomous-driving"
---

## 💡 한 줄 요약
VectorMapNet은 맵 요소를 래스터 픽셀 세그멘테이션 대신 희소 폴리라인(Polyline) 집합으로 정의하고, DETR 탐지기(Element Detector)와 자기회귀 정점 시퀀스 생성기(Autoregressive Polyline Generator)의 2단계 구조로 센서 정보로부터 직접 벡터화된 HD 맵을 End-to-End 예측하여 nuScenes 래스터 SOTA 대비 +22.7 mAP를 달성했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Yicheng Liu, Tianyuan Yuan, Yue Wang, Yilun Wang, Hang Zhao (Shanghai Qi Zhi Inst., Tsinghua Univ., MIT, Li Auto)
- **발행년도**: 2022 (arXiv:2206.08920, ICML 2023)
- **주요 기여점**:
  1. **최초의 학습 기반 3D 벡터 HD 맵 네트워크**: 별도의 휴리스틱 후처리 연산 없이 센서 입력(카메라/LiDAR)에서 인스턴스 정점 시퀀스를 직접 출력하는 파이프라인.
  2. **2단계 탐지-생성 아키텍처 (Map Detector + Polyline Generator)**: Element Detector가 맵 요소의 위치(키포인트 $\mathcal{A}_i$)와 클래스를 탐지하고, Polyline Generator가 이를 조건으로 점 좌표 토큰 시퀀스를 자기회귀 생성.
  3. **이산 범주형 분포(Categorical Distribution) 정점 인코딩**: 점 좌표를 양자화 토큰으로 변환해 비대칭 다중 모드(Multimodal) 도로 경계 분포 모델링.
  4. **다운스트림 플래닝 호환성**: 예측된 벡터 폴리라인을 모션 예측 모델에 직접 투입하여 minADE 오차 -0.083m 감축 검증.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **Rasterized HD Map (HDMapNet)**: 2D BEV 세그멘테이션 이미지를 얻은 후 다단계 휴리스틱 알약/선 후처리로 벡터화하므로 지연시간이 길고 인스턴스 일관성 파괴.
2. **VectorMapNet**: 최초로 맵 구성을 희소 폴리라인 집합 예측 문제로 정식화(Formalization)하여 딥러닝 기반 실시간 벡터 매핑 발전 초석 제안.

---

## 📑 목차
- Chapter 1: 멀티모달 카메라-LiDAR BEV 특징 융합
- Chapter 2: Map Element Detector (DETR 쿼리 & 키포인트 회귀)
- Chapter 3: Autoregressive Polyline Generator (자기회귀 정점 시퀀스 생성)
- Chapter 4: 다중 태스크 학습 손실 함수 (Bipartite Loss + NLL Loss)
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: 멀티모달 카메라-LiDAR BEV 특징 융합

### 1. 요약
카메라 IPM 특징 $\mathcal{F}_{\text{BEV}}^{\mathcal{I}}$와 LiDAR PointPillars 특징 $\mathcal{F}_{\text{BEV}}^{\mathcal{P}}$를 채널 축으로 결합한 후, 2D Conv 레이어를 통과시켜 공유 BEV 피처 맵 $\mathcal{F}_{\text{BEV}}$를 구성합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{F}_{\text{BEV}} = \text{Conv2D}\left( \Big[ \mathcal{F}_{\text{BEV}}^{\mathcal{I}} \;;\; \mathcal{F}_{\text{BEV}}^{\mathcal{P}} \Big] \right) \in \mathbb{R}^{H \times W \times (C_1 + C_2)}$$

```python
import torch
import torch.nn as nn

class MultiSensorBEVFeatureFusion(nn.Module):
    """
    VectorMapNet: 카메라 IPM BEV 피처와 LiDAR PointPillars 피처 채널 결합
    """
    def __init__(self, cam_channels: int = 64, lidar_channels: int = 64, out_channels: int = 256):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(cam_channels + lidar_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, cam_bev: torch.Tensor, lidar_bev: torch.Tensor) -> torch.Tensor:
        """
        cam_bev: (B, C1, H, W)
        lidar_bev: (B, C2, H, W)
        """
        fused_in = torch.cat([cam_bev, lidar_bev], dim=1)
        fused_bev = self.fusion_conv(fused_in)
        return fused_bev

# --- 사용 예시 ---
c_b = torch.randn(1, 64, 100, 100)
l_b = torch.randn(1, 64, 100, 100)
fusion = MultiSensorBEVFeatureFusion()
print("융합된 공유 BEV Feature Map Shape:", fusion(c_b, l_b).shape)
```

---

## 🛠️ Chapter 2: Map Element Detector (DETR 쿼리 & 키포인트 회귀)

### 1. 요약
Element Query $q_{i,j}^{\text{kp}}$를 Deformable DETR 디코더에 통과시켜 각 맵 요소 $i$의 키포인트 좌표 $\mathcal{A}_i = \{a_{i,j}\}_{j=1}^k$와 클래스 확률 $l_i$를 탐지합니다.

### 2. 수식 및 파이썬 코드 설명

$$a_{i,j} = \text{MLP}_{kp}(q_{i,j}^{\text{kp}}) \in \mathbb{R}^2, \quad l_i = \text{MLP}_{cls}\left( [q_{i,1}^{\text{kp}}, \ldots, q_{i,k}^{\text{kp}}] \right)$$

```python
import torch
import torch.nn as nn

class MapElementDetectorHead(nn.Module):
    """
    VectorMapNet Element Detector: 키포인트 위치(A_i) 및 클래스(l_i) 회귀
    """
    def __init__(self, embed_dim: int = 256, num_keypoints: int = 2, num_classes: int = 4):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.kp_head = nn.Linear(embed_dim, 2)
        self.cls_head = nn.Linear(embed_dim * num_keypoints, num_classes)

    def forward(self, query_kp: torch.Tensor) -> tuple:
        """
        query_kp: (B, N_elements, k_keypoints, C)
        """
        B, N_elem, k, C = query_kp.shape
        
        # 1. 키포인트 위치 (B, N_elem, k, 2)
        kp_coords = torch.sigmoid(self.kp_head(query_kp))
        
        # 2. 인스턴스 클래스 분류
        concat_query = query_kp.view(B, N_elem, k * C)
        cls_logits = self.cls_head(concat_query)
        
        return kp_coords, cls_logits

# --- 사용 예시 ---
q_kp = torch.randn(1, 50, 2, 256) # 50개 요소, 2개 Bounding Box 키포인트
det_head = MapElementDetectorHead()
kp_res, cls_res = det_head(q_kp)
print("예측 키포인트 좌표 Shape:", kp_res.shape, "클래스 Logits Shape:", cls_res.shape)
```

---

## 🛠️ Chapter 3: Autoregressive Polyline Generator

### 1. 요약
키포인트 좌표 $\mathcal{A}_i$와 클래스 $l_i$를 조건으로, 정점 좌표 토큰 $v_{i,n}^f$를 이산 확률 분포 $\hat{p}$로부터 자기회귀(Autoregressive) 생성합니다.

### 2. 수식 및 파이썬 코드 설명

$$p(\mathcal{V}_i^{\text{poly}} \mid \mathcal{A}_i, l_i, \mathcal{F}_{\text{BEV}}) = \prod_{n=1}^{2N_v} p(v_{i,n}^f \mid v_{i,<n}^f, \mathcal{A}_i, l_i, \mathcal{F}_{\text{BEV}})$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoregressivePolylineGenerator(nn.Module):
    """
    VectorMapNet Polyline Generator: 양자화 토큰 자기회귀 디코더
    """
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.val_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(100, embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8)
        self.out_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_tokens: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        """
        input_tokens: (B, N_seq) 이전 좌표 양자화 토큰
        cond_feat: (B, 1, C) Detector의 조건 임베딩
        """
        B, N_seq = input_tokens.shape
        pos = torch.arange(N_seq, device=input_tokens.device).unsqueeze(0)
        
        # Coordinate + Position Embedding
        x = self.val_embed(input_tokens) + self.pos_embed(pos)
        x = x.permute(1, 0, 2) # (N_seq, B, C)
        memory = cond_feat.permute(1, 0, 2)
        
        out = self.decoder_layer(x, memory)
        logits = self.out_head(out.permute(1, 0, 2)) # (B, N_seq, vocab_size)
        return logits

# --- 사용 예시 ---
tokens = torch.randint(0, 1000, (1, 20))
cond_f = torch.randn(1, 1, 256)
poly_gen = AutoregressivePolylineGenerator()
print("자기회귀 생성 Logits Shape:", poly_gen(tokens, cond_f).shape)
```

---

## 🛠️ Chapter 4: 다중 태스크 학습 손실 함수

### 1. 요약
Element Detector의 Hungarian Bipartite Matching 손실 $\mathcal{L}_{det}$와 Polyline Generator의 Negative Log-Likelihood 손실 $\mathcal{L}_{gen}$을 가중 합산합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{L}_{gen} = -\frac{1}{2N_v} \sum_{n=1}^{2N_v} \log \hat{p}(v_{i,n}^f \mid v_{i,<n}^f, \mathcal{A}_i, l_i, \mathcal{F}_{\text{BEV}})$$

$$\mathcal{L} = \mathcal{L}_{det} + \mathcal{L}_{gen}$$

```python
import torch
import torch.nn.functional as F

def compute_vectormapnet_loss(
    pred_token_logits: torch.Tensor, # (B, N_seq, Vocab_size)
    gt_tokens: torch.Tensor,         # (B, N_seq)
    loss_det: torch.Tensor
) -> torch.Tensor:
    """
    VectorMapNet 총 손실 = Detector Hungarian Loss + Generator Autoregressive NLL Loss
    """
    B, N_seq, V = pred_token_logits.shape
    loss_gen = F.cross_entropy(pred_token_logits.view(-1, V), gt_tokens.view(-1), reduction='mean')
    
    total_loss = loss_det + loss_gen
    return total_loss

# --- 사용 예시 ---
p_logits = torch.randn(2, 20, 1000)
g_toks = torch.randint(0, 1000, (2, 20))
l_d = torch.tensor(1.5)
print("VectorMapNet 총 학습 손실:", compute_vectormapnet_loss(p_logits, g_toks, l_d).item())
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. nuScenes 온라인 HD 맵 벤치마크 (Chamfer & Fréchet mAP)

| 알고리즘 (Method) | 입력 모달리티 | Pedestrian Crosswalk | Divider | Boundary | 전체 mAP ↑ |
|---|---|---|---|---|---|
| **STSU** | Camera | 7.0 | 11.6 | 16.5 | 11.7 |
| **HDMapNet** | Camera + LiDAR | 16.3 | 29.6 | 46.7 | 31.0 |
| **VectorMapNet (Ours)** | Camera + LiDAR | 37.6 | 50.5 | 47.5 | 45.2 |
| **VectorMapNet + Fine-Tune** | **Camera + LiDAR** | **48.2 (+31.9%)** | **60.1 (+30.5%)** | **53.0 (+6.3%)** | **53.7 (+22.7%)** |

- **결과**: HDMapNet 래스터 기법 대비 **mAP +22.7%p 대폭 폭발적 향상** 및 완전 벡터화 파이프라인 정립.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
VectorMapNet은 맵 구축 문제를 희소 폴리라인 집합 탐지 및 자기회귀 생성으로 재정의한 최초의 End-to-End 벡터화 HD 맵 학술 선구자입니다.

### 2. 한계점 및 아쉬운 점
- 단일 프레임 입력으로 시간적 일관성을 보장하지 못한다.
- 탐지기-생성기 간 2단계 구조의 불일치로 학습 스케줄이 복잡하다(teacher forcing 이후 fine-tuning 필요).
- 자기회귀 생성 방식은 폴리라인 길이에 비례해 추론 속도가 느려질 수 있어, StreamMapNet 등 이후 연구에서 단일 스테이지 구조로 대체된다.

---

## 참고 자료
- [VectorMapNet 공식 GitHub 저장소](https://github.com/Mrmoore98/VectorMapNet_code)
- [ICML 2023 논문 (arXiv:2206.08920)](https://arxiv.org/abs/2206.08920)

*관련 논문: [DETR](/posts/papers/detr-end-to-end-object-detection-with-transformers/), [MapTR](/posts/papers/maptr-structured-modeling-online-vectorized-hd-map-construction/), [StreamMapNet](/posts/papers/streammapnet-streaming-mapping-network-vectorized-online-hd-map-construction/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
