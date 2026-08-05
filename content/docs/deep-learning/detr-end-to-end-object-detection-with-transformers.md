---
title: "DETR: End-to-End Object Detection with Transformers"
date: 2026-04-20T14:00:00+09:00
draft: false
categories: ["Papers", "Computer Vision", "Deep Learning"]
tags: ["Object Detection", "Transformer", "Bipartite Matching", "DETR", "Facebook AI"]
year: 2020
references:
  - "resnet-deep-residual-learning-for-image-recognition"
  - "attention-is-all-you-need"
---

## 💡 한 줄 요약
객체 탐지를 앵커(Anchor)나 NMS(Non-Maximum Suppression) 같은 수작업 후처리 연산 없이 **이분 매칭(Bipartite Matching) 손실**과 **Transformer 인코더-디코더 쿼리(Object Query)** 시스템을 통해 직접 집합 예측(Direct Set Prediction) 문제로 완전히 재정의했다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko (Facebook AI Research)
- **발행년도**: 2020 (arXiv:2005.12872, ECCV 2020)
- **주요 기여점**:
  1. **NMS 및 Anchor 수작업 설계 완전 제거**: 헝가리안 알고리즘(Hungarian Algorithm) 기반 1:1 매칭 손실로 중복 예측을 원천 방지.
  2. **End-to-End Transformer 기반 탐지 파이프라인**: CNN 백본 + Transformer 인코더-디코더 + $N$개의 학습 가능한 Object Query로 구성된 직관적 구조.
  3. **GIoU 및 L1 회귀 손실 결합**: 박스 크기에 불변한 GIoU(Generalized IoU)와 L1 손실을 결합하여 대형 객체 탐지 성능 대폭 향상(Faster R-CNN 대비 AP$_L$ +7.8 AP).

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **Anchor/Proposal 기반 기존 탐지기 (Faster R-CNN, YOLO)**: 수천 개의 Candidate Bounding Box 앵커를 사전 정의하고 NMS 억제 연산으로 중복 박스를 제거하는 수작업 서러게이트(Surrogate) 방식.
2. **DETR**: 집합 간 매칭을 수학적으로 정립하여 이미지 입력에서 정답 바운딩 박스 집합을 직접 출력하는 최초의 End-to-End 쿼리 아키텍처.

---

## 📑 목차
- Chapter 1: 헝가리안 알고리즘 기반 이분 매칭 (Bipartite Matching) 수식
- Chapter 2: Hungarian Multi-Task 손실 함수 (분류 + 박스)
- Chapter 3: L1 및 GIoU (Generalized IoU) 박스 손실 수식
- Chapter 4: Object Query & Transformer Decoder 구조
- Chapter 5: 주요 실험 및 결과
- Chapter 6: 결론 및 시사점

---

## 🛠️ Chapter 1: 헝가리안 알고리즘 기반 이분 매칭 수식

### 1. 요약
$N$개의 예측 박스와 정답(GT) 박스 집합 사이의 매칭 비용 $\mathcal{L}_{\text{match}}$를 최소화하는 최적의 1:1 단사 함수 순열 $\hat{\sigma}$를 헝가리안 알고리즘으로 산출합니다.

### 2. 수식 및 파이썬 코드 설명

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^N \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})$$

$$\mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)}) = -\mathbb{1}_{\{c_i \neq \varnothing\}} \hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)})$$

```python
import torch
from scipy.optimize import linear_sum_assignment

def hungarian_bipartite_matching(
    pred_logits: torch.Tensor, # (N_queries, Num_classes)
    pred_boxes: torch.Tensor,  # (N_queries, 4) -> (x, y, w, h)
    gt_classes: torch.Tensor,  # (N_gt,)
    gt_boxes: torch.Tensor     # (N_gt, 4)
) -> tuple:
    """
    DETR: Scipy linear_sum_assignment를 활용한 Bipartite Matching
    """
    N_q = pred_logits.shape[0]
    N_gt = gt_classes.shape[0]
    
    # 1. 분류 비용: -p(c_i)
    prob = pred_logits.softmax(-1)
    cost_class = -prob[:, gt_classes] # (N_q, N_gt)
    
    # 2. 박스 L1 비용
    cost_bbox = torch.cdist(pred_boxes, gt_boxes, p=1) # (N_q, N_gt)
    
    # 3. 총 매칭 비용 행렬 (Cost Matrix)
    cost_matrix = 1.0 * cost_class + 5.0 * cost_bbox
    cost_matrix_np = cost_matrix.detach().cpu().numpy()
    
    # 4. 헝가리안 알고리즘 수행
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix_np)
    
    return torch.as_tensor(pred_indices, dtype=torch.long), torch.as_tensor(gt_indices, dtype=torch.long)

# --- 사용 예시 ---
p_logits = torch.randn(100, 91) # 100 queries, 91 classes
p_boxes = torch.rand(100, 4)
g_cls = torch.tensor([1, 5, 12]) # 3개의 GT 객체
g_bx = torch.rand(3, 4)
p_idx, g_idx = hungarian_bipartite_matching(p_logits, p_boxes, g_cls, g_bx)
print("매칭된 Query 인덱스:", p_idx, "GT 인덱스:", g_idx)
```

---

## 🛠️ Chapter 2: Hungarian Multi-Task 손실 함수 (분류 + 박스)

### 1. 요약
헝가리안 매칭 인덱스 $\hat{\sigma}$에 따라 매칭된 Query-GT 쌍에 대해서는 Cross Entropy 분류 및 박스 회귀 손실을, 매칭되지 않은 N-N_gt 개 Query에 대해서는 배경(No Object, $\varnothing$) 분류 손실만 부여합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{L}_{\text{Hungarian}}(y, \hat{y}) = \sum_{i=1}^N \left[ -\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)}) \right]$$

```python
import torch
import torch.nn.functional as F

def compute_detr_hungarian_loss(
    pred_logits: torch.Tensor, # (N_q, Num_classes)
    pred_boxes: torch.Tensor,  # (N_q, 4)
    gt_classes: torch.Tensor,  # (N_gt,)
    gt_boxes: torch.Tensor,    # (N_gt, 4)
    match_pred_idx: torch.Tensor,
    match_gt_idx: torch.Tensor,
    bg_class_id: int = 90
) -> torch.Tensor:
    """
    DETR Hungarian Loss (Cross-Entropy + Box Loss)
    """
    N_q = pred_logits.shape[0]
    
    # Target Class 라벨 초기화 (기본: 배경 class)
    target_classes = torch.full((N_q,), bg_class_id, dtype=torch.long, device=pred_logits.device)
    target_classes[match_pred_idx] = gt_classes[match_gt_idx]
    
    # 1. 분류 손실 (논문과 동일하게 배경 클래스 가중치를 0.1로 낮춰,
    #    쿼리 대부분이 배경인 클래스 불균형이 손실을 지배하지 않도록 함)
    num_classes = pred_logits.shape[-1]
    class_weight = torch.ones(num_classes, device=pred_logits.device)
    class_weight[bg_class_id] = 0.1
    loss_ce = F.cross_entropy(pred_logits, target_classes, weight=class_weight)
    
    # 2. 박스 회귀 손실 (매칭된 실제 객체에 대해서만)
    loss_box = F.l1_loss(pred_boxes[match_pred_idx], gt_boxes[match_gt_idx], reduction='sum') / max(1, len(match_gt_idx))
    
    return loss_ce + 5.0 * loss_box

# --- 사용 예시 ---
p_logits = torch.randn(100, 91)
p_boxes = torch.rand(100, 4)
g_cls = torch.tensor([1, 5, 12])
g_bx = torch.rand(3, 4)
p_i = torch.tensor([10, 45, 80])
g_i = torch.tensor([0, 1, 2])
print("DETR 총 헝가리안 손실:", compute_detr_hungarian_loss(p_logits, p_boxes, g_cls, g_bx, p_i, g_i).item())
```

---

## 🛠️ Chapter 3: L1 및 GIoU (Generalized IoU) 박스 손실 수식

### 1. 요약
L1 손실 $\| b_i - \hat{b} \|_1$과 박스 크기에 정규화되는 스케일 불변 GIoU 손실 $\mathcal{L}_{\text{giou}}$를 선형 결합합니다.

### 2. 수식 및 파이썬 코드 설명

$$\mathcal{L}_{\text{box}}(b_i, \hat{b}) = \lambda_{\text{iou}} \mathcal{L}_{\text{giou}}(b_i, \hat{b}) + \lambda_{\text{L1}} \| b_i - \hat{b} \|_1$$

$$\mathcal{L}_{\text{giou}}(A, B) = 1 - \left( \frac{|A \cap B|}{|A \cup B|} - \frac{|C \setminus (A \cup B)|}{|C|} \right)$$

```python
import torch

def compute_giou_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU (GIoU) Loss 계산 (center_x, center_y, w, h -> x1, y1, x2, y2 변환 적용)
    """
    # center_x, center_y, w, h -> x1, y1, x2, y2
    b1_x1, b1_x2 = pred_boxes[:, 0] - pred_boxes[:, 2]/2, pred_boxes[:, 0] + pred_boxes[:, 2]/2
    b1_y1, b1_y2 = pred_boxes[:, 1] - pred_boxes[:, 3]/2, pred_boxes[:, 1] + pred_boxes[:, 3]/2
    
    b2_x1, b2_x2 = gt_boxes[:, 0] - gt_boxes[:, 2]/2, gt_boxes[:, 0] + gt_boxes[:, 2]/2
    b2_y1, b2_y2 = gt_boxes[:, 1] - gt_boxes[:, 3]/2, gt_boxes[:, 1] + gt_boxes[:, 3]/2
    
    # 교집합(Intersection)
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    # 합집합(Union)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-7)
    
    # 최소 볼록 외경 상자(Conclosing Enclosing Box C)
    enc_x1 = torch.min(b1_x1, b2_x1)
    enc_y1 = torch.min(b1_y1, b2_y1)
    enc_x2 = torch.max(b1_x2, b2_x2)
    enc_y2 = torch.max(b1_y2, b2_y2)
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)
    
    giou = iou - (enc_area - union_area) / (enc_area + 1e-7)
    return (1.0 - giou).mean()

# --- 사용 예시 ---
p_box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
g_box = torch.tensor([[0.55, 0.55, 0.2, 0.2]])
print("GIoU Loss:", compute_giou_loss(p_box, g_box).item())
```

---

## 🛠️ Chapter 4: Object Query & Transformer Decoder 구조

### 1. 요약
$N$개의 학습 가능한 위치 벡터인 Object Query가 디코더 Cross-Attention을 통해 인코더 비전 피처와 상호작용하고, 3층 MLP 박스 헤드로 예측값을 산출합니다.

### 2. 수식 및 파이썬 코드 설명

$$\text{Decoder}(Q_{\text{query}}, \mathcal{F}_{\text{enc}}) \to \mathbf{H}_{\text{out}} \in \mathbb{R}^{N \times d_{\text{model}}}$$

$$\hat{b}_i = \text{MLP}_{\text{box}}(\mathbf{h}_i), \quad \hat{c}_i = \text{Linear}_{\text{class}}(\mathbf{h}_i)$$

```python
import torch
import torch.nn as nn

class DETRDecoderHead(nn.Module):
    """
    DETR Object Query Decoder 및 Prediction Heads
    """
    def __init__(self, num_queries: int = 100, d_model: int = 256, num_classes: int = 91):
        super().__init__()
        self.object_queries = nn.Parameter(torch.randn(num_queries, d_model))
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        
        self.class_head = nn.Linear(d_model, num_classes)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
            nn.Sigmoid() # 0 ~ 1 범위 정규화 좌표
        )

    def forward(self, enc_features: torch.Tensor) -> tuple:
        """
        enc_features: (HW, B, d_model)
        """
        B = enc_features.shape[1]
        queries = self.object_queries.unsqueeze(1).repeat(1, B, 1) # (N_q, B, d_model)
        
        hs = self.decoder(queries, enc_features) # (N_q, B, d_model)
        hs = hs.permute(1, 0, 2) # (B, N_q, d_model)
        
        pred_logits = self.class_head(hs)
        pred_boxes = self.bbox_head(hs)
        return pred_logits, pred_boxes

# --- 사용 예시 ---
detr_head = DETRDecoderHead()
f_enc = torch.randn(400, 2, 256) # 20x20 피처 맵
logits, boxes = detr_head(f_enc)
print("DETR Class Logits Shape:", logits.shape, "Box Predictions Shape:", boxes.shape)
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. COCO 2017 객체 탐지 리더보드 비교

| 알고리즘 (Method) | 백본 (Backbone) | 대형 객체 AP$_L$ ↑ | 전체 mAP ↑ | 추론 속도 (FPS) ↑ |
|---|---|---|---|---|
| **Faster R-CNN + FPN** | ResNet-50 | 53.4 | 42.0 | 26 FPS |
| **DETR (Ours)** | **ResNet-50** | **61.1 (+7.7%)** | **42.0** | **28 FPS (End-to-End)** |
| **DETR-DC5 (Ours)** | **ResNet-50** | **61.1** | **43.3 (+1.3%)** | 12 FPS |

- **결과**: NMS 없이 완전 End-to-End로 동작하면서 전역 Self-Attention 덕분에 대형 객체 탐지 성능(AP$_L$)에서 기존 SOTA를 **+7.7%p 격차로 완파**.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
DETR은 객체 탐지를 직접 집합 예측(Direct Set Prediction)으로 재정의하고 Bipartite Matching 손실을 도입하여, 후속 MapTR, DETR3D 등 모든 자율주행 쿼리 기반 탐지 모델의 원류를 정립한 시대를 연 거두입니다.

### 2. 한계점 및 아쉬운 점
- 소형 객체 성능이 Faster R-CNN 대비 열세 — Deformable DETR(2021)이 deformable attention으로 해결.
- 매우 긴 학습 스케줄 필요 (500 epoch) — Conditional DETR, DAB-DETR 등으로 수렴 속도 개선.
- 인코더의 높은 계산 비용 (HW 크기 self-attention) — Deformable DETR, Sparse DETR로 효율화.
- 헝가리안 매칭이 학습 초반 불안정할 수 있다는 점은 논문에서 충분히 다루지 않은 아쉬운 부분.

---

## 참고 자료
- [DETR 공식 GitHub 저장소](https://github.com/facebookresearch/detr)
- [ECCV 2020 논문 (arXiv:2005.12872)](https://arxiv.org/abs/2005.12872)

*관련 논문: [Attention Is All You Need](/posts/papers/attention-is-all-you-need/), [ResNet](/posts/papers/resnet-deep-residual-learning-for-image-recognition/), [DETR3D](/posts/papers/detr3d-3d-object-detection-multi-view-images/), [VectorMapNet](/posts/papers/vectormapnet-end-to-end-vectorized-hd-map-learning/), [MapTR](/posts/papers/maptr-structured-modeling-online-vectorized-hd-map-construction/)*
