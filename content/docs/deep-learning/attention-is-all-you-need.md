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
RNN과 CNN을 완전히 제거하고 Attention 메커니즘만으로 구성된 Transformer를 제안하여, 기계 번역에서 더 적은 학습 비용으로 SOTA를 달성하고 이후 모든 Transformer 기반 모델의 원류가 되었다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain / Google Research)
- **발행년도**: 2017 (NeurIPS 2017, arXiv: 1706.03762)
- **주요 기여점**:
  1. Recurrence와 Convolution을 완전히 제거하고 Self-Attention만으로 시퀀스 변환을 수행하는 최초의 아키텍처 Transformer 제안
  2. Scaled Dot-Product Attention과 Multi-Head Attention을 결합해 임의의 두 위치 간 경로 길이를 O(1)로 단축
  3. Sinusoidal Positional Encoding으로 순서 정보를 학습 없이 주입, 학습보다 긴 시퀀스에도 외삽 가능
  4. WMT 2014 영어-독일어/영어-프랑스어 번역에서 이전 SOTA 대비 적은 학습 비용으로 더 높은 BLEU 달성

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: 2017년 당시 기계 번역의 SOTA는 LSTM/GRU 기반 Encoder-Decoder였다. 입력 시퀀스를 순서대로 읽어 hidden state를 만들고 디코더가 출력을 생성하는 방식이며, Attention은 이 구조의 장거리 의존성 희석 문제를 완화하기 위한 보조 장치로 이미 사용되고 있었다. 한편 ConvS2S 등 CNN 기반 seq2seq 모델도 병렬화를 위해 제안되었으나 멀리 떨어진 위치 간 관계 파악을 위해 레이어를 깊게 쌓아야 했다.
- **기존 한계점**:
  1. RNN의 순차 처리 — RNN/LSTM은 시간 순서대로 hidden state를 계산해야 하므로 병렬화가 불가능하고, 긴 시퀀스일수록 학습 속도가 느리며 앞 정보가 뒤로 갈수록 희석되는 장거리 의존성(long-range dependency) 문제가 발생한다.
  2. CNN의 제한된 수용 영역 — CNN은 병렬화는 가능하나 멀리 떨어진 위치 간 관계를 파악하려면 레이어를 깊게 쌓아야 한다(ConvS2S: O(log_k(n)) 경로 길이).
  3. Attention의 보조적 사용 — 기존 모델들은 RNN과 Attention을 함께 사용했고, Attention은 보조 메커니즘에 불과했다.
- **이 논문의 접근 방식**: Recurrence와 Convolution을 완전히 제거하고, **Self-Attention만으로** 입출력의 전역적 의존성을 모델링한다. 임의의 두 위치 간 경로 길이를 O(1)로 단축하고 완전한 병렬 학습을 실현한다.

## 📑 목차
- Chapter 1-2: Introduction & Background
- Chapter 3: Model Architecture — Encoder/Decoder, Attention, FFN, Positional Encoding
- Chapter 4: Why Self-Attention
- Chapter 5: Training
- Chapter 6: Results
- Chapter 7: Conclusion

## 🛠️ Chapter 1-2: Introduction & Background

**요약**

2017년 당시 기계 번역의 SOTA는 LSTM/GRU 기반 Encoder-Decoder였다. 이 모델들은 입력 시퀀스를 순서대로 읽어 hidden state를 만들고 디코더에서 출력을 생성한다. 문제는 본질적으로 순차적이라 병렬화가 어렵고, 긴 문장에서 초반 토큰의 정보가 점점 희석된다는 것이다.

Attention 메커니즘은 이 희석 문제를 완화하기 위해 이미 사용되고 있었지만, 항상 RNN과 함께 쓰였다. Transformer는 **"RNN 없이 Attention만으로도 충분하다"** 는 아이디어에서 출발한다.

**핵심 개념**

- **Sequence Transduction**: 입력 시퀀스를 출력 시퀀스로 변환하는 문제 (번역, 요약, 파싱 등)
- **Self-Attention (Intra-Attention)**: 같은 시퀀스 내의 서로 다른 위치 간 관계를 계산하는 Attention. "나"라는 단어가 문장 내 다른 어떤 단어와 관련이 깊은지 파악
- **장거리 의존성 문제**: 예: "The animal didn't cross the street because **it** was too tired" — "it"이 "animal"을 가리킨다는 것을 파악하려면 멀리 떨어진 토큰 간 직접 연결이 필요

## 🛠️ Chapter 3: Model Architecture

### 3.1 Encoder and Decoder Stacks

**요약**

Transformer는 **Encoder 스택**과 **Decoder 스택**으로 구성된다. 둘 다 $N=6$개의 동일한 레이어를 쌓는다.

- **Encoder 레이어**: ① Multi-Head Self-Attention → ② Position-wise Feed-Forward Network. 각 서브레이어 후 잔차 연결(Residual Connection)과 Layer Normalization 적용
- **Decoder 레이어**: ① Masked Multi-Head Self-Attention → ② Multi-Head Cross-Attention (Encoder 출력 참조) → ③ Position-wise FFN. 마스킹은 미래 위치를 보지 못하도록 해 자동회귀(auto-regressive) 특성 보장

**핵심 개념**

- **Residual Connection**: $\text{LayerNorm}(x + \text{Sublayer}(x))$ — 깊은 네트워크에서 gradient 소실 방지, ResNet에서 유래
- **Layer Normalization**: 각 레이어 출력을 정규화해 학습 안정화
- **Auto-Regressive**: 디코더가 출력을 한 토큰씩 생성할 때 이전에 생성한 토큰만 참조하도록 제한
- **$d_\text{model} = 512$**: 모든 서브레이어와 임베딩의 출력 차원

### 3.2 Scaled Dot-Product Attention

**요약**

Transformer의 핵심 연산이다. Query(Q), Key(K), Value(V) 세 행렬을 입력받아 가중합(weighted sum)을 출력한다. 단순히 말하면: "Query가 어떤 Key와 유사한지 계산하고, 그 유사도에 비례해 Value를 가져온다."

**수식 예제**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**수식 설명**

- **$Q \in \mathbb{R}^{n \times d_k}$** (Query): "나는 무엇을 찾고 싶은가?" — 각 위치의 질문 벡터
- **$K \in \mathbb{R}^{m \times d_k}$** (Key): "나는 어떤 정보를 가지고 있는가?" — 각 위치의 식별자 벡터
- **$V \in \mathbb{R}^{m \times d_v}$** (Value): "실제로 전달할 내용은 무엇인가?" — 각 위치의 실제 정보 벡터
- **$QK^T$**: Q와 K의 내적(dot product) — 두 벡터가 얼마나 비슷한지 수치화. 값이 클수록 관련성 높음
- **$\frac{1}{\sqrt{d_k}}$**: 스케일링 인수 — $d_k$가 커질수록 내적값이 커져 softmax gradient가 0에 가까워지는 문제를 방지. 분산을 1로 유지
  - 예: $d_k = 64$이면 $\sqrt{64} = 8$로 나눔
- **$\text{softmax}(\cdot)$**: 유사도를 확률(합=1)로 변환 → 어느 위치에 얼마나 집중할지 결정
- **$\cdot V$**: 확률 가중치로 Value를 가중합 → 관련성 높은 위치의 정보를 더 많이 가져옴
- **직관 예시**: "나는(Q) → 동물(K)과 유사 → 동물의 정보(V)를 가져와 'it = animal' 파악"

### 3.3 Multi-Head Attention

**요약**

단일 Attention 대신 $h$개의 Attention을 병렬로 수행하고 결과를 합친다. 각 head는 서로 다른 관점에서 관계를 학습한다. 예: head 1은 문법적 관계, head 2는 의미적 관계 등.

**수식 예제**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

$$\text{where} \quad \text{head}_i = \text{Attention}(Q W_i^Q,\ K W_i^K,\ V W_i^V)$$

**수식 설명**

- **$h = 8$**: 8개의 독립적인 Attention head를 병렬로 수행
- **$W_i^Q \in \mathbb{R}^{d_\text{model} \times d_k}$**: i번째 head의 Query 투영 행렬 (학습 파라미터)
- **$W_i^K \in \mathbb{R}^{d_\text{model} \times d_k}$**: i번째 head의 Key 투영 행렬
- **$W_i^V \in \mathbb{R}^{d_\text{model} \times d_v}$**: i번째 head의 Value 투영 행렬
- **$d_k = d_v = d_\text{model}/h = 64$**: 각 head의 차원. 전체 차원을 head 수로 나눠 총 계산량을 단일 head와 동일하게 유지
- **$\text{Concat}(\cdot)$**: 8개 head의 출력($\mathbb{R}^{n \times 64}$ 각각)을 이어 붙여 $\mathbb{R}^{n \times 512}$로 만듦
- **$W^O \in \mathbb{R}^{hd_v \times d_\text{model}}$**: 최종 선형 투영 — 합쳐진 출력을 원래 차원으로 변환
- **직관**: "8명의 전문가가 각자 다른 관점으로 같은 문장을 분석하고, 그 결과를 취합한다"

**3가지 Attention 사용 방식**

| 사용 위치 | Q 출처 | K, V 출처 | 역할 |
|-----------|--------|-----------|------|
| Encoder Self-Attention | 이전 Encoder 레이어 | 이전 Encoder 레이어 | 입력 시퀀스 내 위치 간 관계 파악 |
| Decoder Masked Self-Attention | 이전 Decoder 레이어 | 이전 Decoder 레이어 | 생성된 출력 내 관계 파악 (미래 마스킹) |
| Decoder Cross-Attention | 이전 Decoder 레이어 | Encoder 최종 출력 | 입력과 출력의 관계 파악 (번역의 핵심) |

### 3.4 Position-wise Feed-Forward Networks

**요약**

각 Attention 서브레이어 다음에 오는 완전 연결 네트워크. 각 위치마다 **독립적으로, 동일하게** 적용된다 (위치 간 파라미터 공유). Attention이 "어디를 볼지" 결정한다면, FFN은 "본 정보로 무엇을 계산할지" 담당한다.

**수식 예제**

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

**수식 설명**

- **$x \in \mathbb{R}^{d_\text{model}}$**: 각 위치의 입력 벡터 ($d_\text{model} = 512$)
- **$W_1 \in \mathbb{R}^{512 \times 2048}$, $b_1$**: 첫 번째 선형 변환 파라미터 — 512 → 2048로 확장
- **$\max(0, \cdot)$**: ReLU 활성화 함수 — 음수를 0으로 만들어 비선형성 추가
- **$W_2 \in \mathbb{R}^{2048 \times 512}$, $b_2$**: 두 번째 선형 변환 — 2048 → 512로 다시 축소
- **$d_{ff} = 2048$**: 내부 레이어의 차원, $d_\text{model}$의 4배
- **직관**: "Attention으로 관련 정보를 모은 뒤, FFN이 그 정보를 변환·처리하는 작업 공간 역할"

### 3.5 Positional Encoding

**요약**

Transformer는 RNN과 달리 순서 정보가 없다. 입력 토큰의 위치 정보를 주입하기 위해 **Positional Encoding**을 임베딩에 더한다. 사인·코사인 함수를 사용해 각 위치마다 고유한 패턴을 만든다.

**수식 예제**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)$$

**수식 설명**

- **$pos$**: 시퀀스 내 토큰의 위치 (0, 1, 2, ...)
- **$i$**: 임베딩 차원의 인덱스 (0, 1, ..., $d_\text{model}/2 - 1$)
- **$2i$**: 짝수 차원 → sin 함수 사용
- **$2i+1$**: 홀수 차원 → cos 함수 사용
- **$10000^{2i/d_\text{model}}$**: 차원마다 다른 주파수를 사용해 각 위치가 고유한 인코딩을 갖도록 함. $i$가 클수록 주파수가 낮아짐(파장이 길어짐)
- **장점**: 학습 없이 고정값, 훈련보다 긴 시퀀스에도 외삽(extrapolation) 가능
- **직관**: "음악의 음계처럼, 여러 주파수의 파동을 겹쳐 각 위치를 고유하게 표현한다. 위치 1과 2는 비슷한 패턴이지만 미세하게 다르고, 멀리 떨어진 위치일수록 패턴 차이가 커진다"

## 🛠️ Chapter 4: Why Self-Attention

**요약**

Self-Attention이 Recurrent, Convolutional 레이어보다 왜 우수한지 세 가지 기준으로 비교한다.

| 레이어 타입 | 레이어당 복잡도 | 순차 연산 수 | 최대 경로 길이 |
|------------|---------------|------------|-------------|
| **Self-Attention** | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

*$n$: 시퀀스 길이, $d$: 표현 차원, $k$: 커널 크기, $r$: 제한된 어텐션 범위*

**핵심 개념**

- **순차 연산 O(1)**: Self-Attention은 모든 위치를 한 번에 병렬 계산 → GPU 활용도 극대화
- **최대 경로 길이 O(1)**: 임의의 두 위치가 단 한 번의 Attention으로 직접 연결 → 장거리 의존성 학습 용이
- **트레이드오프**: Self-Attention은 $O(n^2)$ 복잡도로, 시퀀스가 길면(n이 크면) 비용이 큼. 이 문제는 이후 Longformer, FlashAttention 등으로 해결

## 🛠️ Chapter 5: Training

**요약**

WMT 2014 영어-독일어(4.5M 문장), 영어-프랑스어(36M 문장) 데이터셋으로 학습. 8개 NVIDIA P100 GPU 사용.

**핵심 개념**

- **Adam Optimizer**: $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$
- **Warmup Learning Rate Schedule**: 처음 4000 스텝 동안 선형 증가, 이후 step의 역제곱근에 비례해 감소

**수식 예제**

$$lrate = d_\text{model}^{-0.5} \cdot \min(step\_num^{-0.5},\ step\_num \cdot warmup\_steps^{-1.5})$$

**수식 설명**

- **$d_\text{model}^{-0.5}$**: 모델 차원이 클수록 학습률을 낮춤 (큰 모델의 불안정성 방지)
- **$step\_num^{-0.5}$**: 학습이 진행될수록 학습률 감소 (수렴 안정화)
- **$step\_num \cdot warmup\_steps^{-1.5}$**: 초반 warmup 동안 학습률 선형 증가
- **$\min(\cdot)$**: 두 값 중 작은 것 선택 — warmup 구간(작은 step)에서는 두 번째 항, 이후에는 첫 번째 항이 지배
- **$warmup\_steps = 4000$**: 약 4000 스텝까지 워밍업
- **직관**: "처음에는 천천히 학습 속도를 높이다가, 어느 시점부터 점점 줄여 안정적으로 수렴한다"

**정규화 기법**
- **Residual Dropout** ($P_{drop} = 0.1$): 각 서브레이어 출력에 Dropout 적용
- **Label Smoothing** ($\epsilon_{ls} = 0.1$): 정답 레이블을 100% 확신하지 않고 10% 불확실성 부여 → 과적합 방지, BLEU 향상

## 📊 주요 실험 및 결과 (Experiments & Results)

- **사용 데이터셋 / 벤치마크**: WMT 2014 영어-독일어(EN-DE), 영어-프랑스어(EN-FR) 기계 번역, 영어 구문 분석(constituency parsing)
- **주요 성과**:

| 모델 | EN-DE BLEU | EN-FR BLEU | Training Cost (FLOPs) |
|------|-----------|-----------|----------------------|
| ConvS2S | 25.16 | 40.46 | $9.6 \times 10^{18}$ |
| GNMT + RL (Ensemble) | 26.30 | 41.16 | $1.8 \times 10^{20}$ |
| **Transformer (base)** | **27.3** | **38.1** | **$3.3 \times 10^{18}$** |
| **Transformer (big)** | **28.4** | **41.8** | **$2.3 \times 10^{19}$** |

  - EN-DE: 이전 SOTA 대비 **+2 BLEU** 향상, 훈련 비용은 기존 모델의 수십 분의 1
  - EN-FR: 단일 모델 기준 새로운 SOTA (BLEU 41.0), 훈련 비용 1/4
  - **영어 구문 분석(constituency parsing)**: 언어 과제에서도 강한 일반화 성능 확인
  - **아키텍처 ablation (Table 3 핵심)**: Head 수 1개보다 8개가 0.9 BLEU 향상하지만 32개로 너무 많아도 성능 저하. $d_k$ 크기 축소 시 성능 저하 → attention 유사도 계산의 정밀도가 중요. Dropout이 과적합 방지에 핵심적.

## 💡 결론 및 시사점 (Conclusion & Insights)

Transformer는 "Attention is All You Need"라는 제목 그대로, **Attention 하나로 RNN과 CNN을 대체**할 수 있음을 증명했다. 장거리 의존성을 O(1) 경로로 처리하고, 완전한 병렬 학습을 가능하게 해 현대 딥러닝의 패러다임을 바꾸었다.

**현재 논문 목록과의 연결:**

| 후속 논문 | Transformer 활용 방식 |
|----------|----------------------|
| **BEVFormer** | Spatial Cross-Attention으로 카메라 feature → BEV 공간 변환 |
| **DETR3D** | 3D object query의 2D back-projection에 Cross-Attention 활용 |
| **UniAD** | 5개 태스크를 Query로 연결하는 통합 Transformer 파이프라인 |
| **LDM** | UNet 중간 레이어에 Cross-Attention 삽입해 텍스트·조건 통합 |
| **GAIA-1** | GPT 방식(Decoder-only Transformer)으로 AV World Model 학습 |
| **MagicDrive** | BEV 맵·3D 박스를 Cross-Attention으로 이미지 생성에 조건화 |
| **VAD** | 벡터화 장면 표현을 Transformer로 처리해 경량 계획 수행 |

- **자율주행 합성 데이터 관점**: BEV 인식부터 World Model까지 전 파이프라인이 Transformer 기반이므로, 이 논문의 Self-Attention / Cross-Attention / Positional Encoding 개념을 정확히 이해하는 것이 이후 논문 해석의 필수 전제 조건이다.
- **한계점 및 아쉬운 점**:
  - $O(n^2)$ 복잡도로 인해 매우 긴 시퀀스(고해상도 이미지, 긴 문서)에는 그대로 적용하기 어려움 — 이후 Longformer, Linformer, FlashAttention 등이 이를 해결
  - Sinusoidal Positional Encoding은 고정값이라 유연성이 떨어지며, 이후 연구들은 학습 가능한 위치 임베딩이나 RoPE 등으로 대체
  - 논문 자체는 번역 태스크에 집중되어 있어, 비전·멀티모달 등으로의 일반화 가능성은 후속 연구(ViT, DETR 등)에서 비로소 검증됨

---

*관련 논문: [ResNet](/posts/papers/resnet-deep-residual-learning-for-image-recognition/), [ViT](/posts/papers/vit-an-image-is-worth-16x16-words/), [DETR](/posts/papers/detr-end-to-end-object-detection-with-transformers/), [BEVFormer](/posts/papers/bevformer/)*
