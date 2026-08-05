---
title: "Attention Is All You Need"
date: 2026-04-20T13:00:00+09:00
draft: false
categories: ["Papers", "Transformer", "NLP"]
tags: ["Transformer", "Self-Attention", "Multi-Head Attention", "Positional Encoding", "Seq2Seq"]
year: 2017
references: []
---

## 💡 한 줄 요약
RNN과 CNN을 완전히 제거하고 오직 Attention 메커니즘만으로 인코더-디코더를 구성한 **Transformer**를 제안하여, 기계 번역에서 계산 복잡도 $\mathcal{O}(1)$ 경로 길이와 완전한 병렬성을 달성하고 현대 모든 대형 모델의 원류가 되었다.

---

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain, Google Research, Univ. of Toronto)
- **발행년도**: 2017 (arXiv:1706.03762, NeurIPS 2017)
- **주요 기여점**:
  1. **Self-Attention 중심 아키텍처**: 순차적 Recurrence(RNN) 및 신관 수용 영역의 Convolution(CNN)을 제거하고, 시퀀스 내 위치 간 관련성을 일시에 파악하는 Self-Attention 구조 정립.
  2. **Scaled Dot-Product & Multi-Head Attention**: $Q, K, V$ 행렬 내적을 $\sqrt{d_k}$로 스케일링하고, $h$개 헤드로 분할하여 다양한 관점의 기하학적·의미론적 관련성을 병렬 추출.
  3. **Sinusoidal Positional Encoding**: 삼각함수 파장을 이용해 순서 정보(Position)를 파라미터 추가 없이 임베딩에 주입하여 길이에 구애받지 않는 외삽(Extrapolation) 허용.
  4. **Warmup Learning Rate Schedule**: 모델 차원에 역제곱근으로 비례하는 동적 스케줄링으로 기울기 폭주/소실 방지.

---

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)

### 관련 연구 흐름
1. **RNN/LSTM 기반 Seq2Seq**: 시퀀스를 1개 토큰씩 순차 처리해야 하므로 GPU 병렬화가 불가능하고, 장거리 의존성(Long-range Dependency) 정보가 희석됨.
2. **CNN 기반 Seq2Seq (ConvS2S)**: 병렬 처리는 가능하나 멀리 떨어진 토큰 간 관계를 묶기 위해 $\mathcal{O}(\log_k n)$의 깊은 레이어 스택 필요.
3. **Transformer**: 임의의 두 토큰 간 최대 경로 길이를 $\mathcal{O}(1)$로 단축하고 완전한 병렬 처리를 완성.

---

## 📑 목차
- Chapter 1: Scaled Dot-Product Attention 수식 해설
- Chapter 2: Multi-Head Attention (MHA) 병렬 구조
- Chapter 3: Position-wise Feed-Forward Networks (FFN)
- Chapter 4: Sinusoidal Positional Encoding 위치 인코딩
- Chapter 5: Warmup Learning Rate Scheduler 수식
- Chapter 6: 주요 실험 및 결과
- Chapter 7: 결론 및 시사점

---

## 🛠️ Chapter 1: Scaled Dot-Product Attention 수식 해설

### 1. 요약
Query($Q$)와 Key($K$)의 내적값에 $\frac{1}{\sqrt{d_k}}$ 스케일을 곱해 Softmax 확률을 산출한 뒤 Value($V$)를 가중합합니다.

### 2. 수식 및 파이썬 코드 설명

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- **$Q \in \mathbb{R}^{N \times d_k}, K \in \mathbb{R}^{M \times d_k}, V \in \mathbb{R}^{M \times d_v}$**: Query, Key, Value 행렬
- **$\sqrt{d_k}$**: $d_k$ 차원이 커질 때 내적값이 커져 Softmax 기울기가 소실(Gradient Vanishing)되는 현상을 방지하는 스케일링 인자

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
    Q: torch.Tensor, # (B, N, d_k)
    K: torch.Tensor, # (B, M, d_k)
    V: torch.Tensor, # (B, M, d_v)
    mask: torch.Tensor = None # (B, N, M) 마스크 (옵션)
) -> tuple:
    """
    Scaled Dot-Product Attention 수식 구현
    Attention(Q, K, V) = Softmax( Q @ K^T / sqrt(d_k) ) @ V
    """
    d_k = Q.size(-1)
    
    # 1. Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # 2. Masking 처리 (미래 토큰 가리기 등)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # 3. Softmax 확률 산출
    attn_weights = F.softmax(scores, dim=-1) # (B, N, M)
    
    # 4. Value 가중합
    output = torch.matmul(attn_weights, V) # (B, N, d_v)
    return output, attn_weights

# --- 사용 예시 ---
q_in = torch.randn(2, 10, 64) # Batch=2, Seq_len=10, d_k=64
k_in = torch.randn(2, 10, 64)
v_in = torch.randn(2, 10, 64)
out_attn, weights = scaled_dot_product_attention(q_in, k_in, v_in)
print("Attention 출력 Shape:", out_attn.shape, "가중치 Shape:", weights.shape)
```

---

## 🛠️ Chapter 2: Multi-Head Attention (MHA) 병렬 구조

### 1. 요약
단일 Attention 연산 대신, $d_{\text{model}}$ 차원을 $h$개 헤드로 분할($d_k = d_{\text{model}} / h$)하여 병렬 계산한 후 결과를 사영(Concat & Project)합니다.

### 2. 수식 및 파이썬 코드 설명

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA) 모듈
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = Q.shape
        M = K.shape[1]
        
        # 1. 선형 사영 후 h개 헤드로 분할 (B, h, N, d_k)
        q_heads = self.W_q(Q).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k_heads = self.W_k(K).view(B, M, self.num_heads, self.d_k).transpose(1, 2)
        v_heads = self.W_v(V).view(B, M, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention 병렬 계산
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v_heads) # (B, h, N, d_k)
        
        # 3. Concat & Output Projection
        concat_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.W_o(concat_out)

# --- 사용 예시 ---
mha = MultiHeadAttention(d_model=512, num_heads=8)
x_in = torch.randn(2, 10, 512)
print("Multi-Head Attention 결과 Shape:", mha(x_in, x_in, x_in).shape)
```

---

## 🛠️ Chapter 3: Position-wise Feed-Forward Networks (FFN)

### 1. 요약
각 위치별로 독립적 적용되는 2층 Fully-Connected 네트워크로, 중간 차원 $d_{ff} = 2048$로 확충 후 다시 $d_{\text{model}} = 512$로 축소합니다.

### 2. 수식 및 파이썬 코드 설명

$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

```python
import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN)
    """
    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))

# --- 사용 예시 ---
ffn = PositionwiseFeedForward()
x_dummy = torch.randn(2, 10, 512)
print("FFN 출력 Shape:", ffn(x_dummy).shape)
```

---

## 🛠️ Chapter 4: Sinusoidal Positional Encoding 위치 인코딩

### 1. 요약
시퀀스의 토큰 위치 $pos$와 차원 $i$에 따라 주기(주파수)가 다른 사인/코사인 곡선을 조합하여 고유한 위치 임베딩을 구성합니다.

### 2. 수식 및 파이썬 코드 설명

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

```python
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding 위치 인코딩 텐서 생성
    """
    def __init__(self, d_model: int = 512, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, d_model)
        """
        return x + self.pe[:, :x.size(1)]

# --- 사용 예시 ---
pe_module = SinusoidalPositionalEncoding()
embeddings = torch.zeros(1, 20, 512)
print("Positional Encoding 주입 후 Shape:", pe_module(embeddings).shape)
```

---

## 🛠️ Chapter 5: Warmup Learning Rate Scheduler 수식

### 1. 요약
학습 초기 $warmup\_steps=4000$ 동안 학습률을 선형 증가시키고, 이후 step의 역제곱근에 비례해 감소시킵니다.

### 2. 수식 및 파이썬 코드 설명

$$lrate = d_{\text{model}}^{-0.5} \cdot \min\left( step\_num^{-0.5}, \ step\_num \cdot warmup\_steps^{-1.5} \right)$$

```python
import torch

def compute_transformer_lr(step_num: int, d_model: int = 512, warmup_steps: int = 4000) -> float:
    """
    Transformer 동적 Warmup Learning Rate 스케줄러
    """
    step_num = max(1, step_num)
    arg1 = step_num ** -0.5
    arg2 = step_num * (warmup_steps ** -1.5)
    return (d_model ** -0.5) * min(arg1, arg2)

# --- 사용 예시 ---
print("Step 1000 학습률:", compute_transformer_lr(1000))
print("Step 4000 (최대) 학습률:", compute_transformer_lr(4000))
print("Step 20000 학습률:", compute_transformer_lr(20000))
```

---

## 📊 주요 실험 및 결과 (Experiments & Results)

### 1. WMT 2014 기계 번역 벤치마크 비교

| 모델 (Method) | EN-DE (BLEU) ↑ | EN-FR (BLEU) ↑ | 학습 연산량 (FLOPs) ↓ |
|---|---|---|---|
| **ConvS2S** | 25.16 | 40.46 | $9.6 \times 10^{18}$ |
| **GNMT + RL** | 26.30 | 41.16 | $1.8 \times 10^{20}$ |
| **Transformer-base** | 27.30 | 38.10 | **$3.3 \times 10^{18}$ (초저비용)** |
| **Transformer-big** | **28.40 (+2.1)** | **41.80 (+0.64)** | **$2.3 \times 10^{19}$** |

- **결과**: 기존 RNN/CNN SOTA 대비 **BLEU 점수 +2.1 점 향상**과 동시에 **학습 연산 비용을 수십 분의 1로 절감**.

---

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
Transformer는 "Attention Is All You Need"라는 명제하에 순차적 계산 연결을 탈피하고 모든 AI 신경망 패러다임을 Self-Attention 기반으로 전환한 가장 파괴적인 세미날 논문입니다.

### 2. 자율주행 분야에서의 활용

| 후속 논문 | Transformer 활용 방식 |
|----------|----------------------|
| **BEVFormer** | Spatial Cross-Attention으로 카메라 feature → BEV 공간 변환 |
| **DETR3D** | 3D object query의 2D back-projection에 Cross-Attention 활용 |
| **UniAD** | 5개 태스크를 Query로 연결하는 통합 Transformer 파이프라인 |
| **LDM** | UNet 중간 레이어에 Cross-Attention 삽입해 텍스트·조건 통합 |
| **GAIA-1** | GPT 방식(Decoder-only Transformer)으로 AV World Model 학습 |
| **MagicDrive** | BEV 맵·3D 박스를 Cross-Attention으로 이미지 생성에 조건화 |
| **VAD** | 벡터화 장면 표현을 Transformer로 처리해 경량 계획 수행 |

BEV 인식부터 World Model까지 전 파이프라인이 Transformer 기반이므로, 이 논문의 Self-Attention / Cross-Attention / Positional Encoding 개념을 정확히 이해하는 것이 이후 논문 해석의 필수 전제 조건이다.

### 3. 한계점 및 아쉬운 점
- $O(n^2)$ 복잡도로 인해 매우 긴 시퀀스(고해상도 이미지, 긴 문서)에는 그대로 적용하기 어려움 — 이후 Longformer, Linformer, FlashAttention 등이 이를 해결.
- Sinusoidal Positional Encoding은 고정값이라 유연성이 떨어지며, 이후 연구들은 학습 가능한 위치 임베딩이나 RoPE 등으로 대체.
- 논문 자체는 번역 태스크에 집중되어 있어, 비전·멀티모달 등으로의 일반화 가능성은 후속 연구(ViT, DETR 등)에서 비로소 검증됨.

---

## 참고 자료
- [Google Research Transformer GitHub](https://github.com/tensorflow/tensor2tensor)
- [NeurIPS 2017 논문 (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762)

*관련 논문: [ResNet](/posts/papers/resnet-deep-residual-learning-for-image-recognition/), [ViT](/posts/papers/vit-an-image-is-worth-16x16-words/), [DETR](/posts/papers/detr-end-to-end-object-detection-with-transformers/), [BEVFormer](/posts/papers/bevformer/)*
