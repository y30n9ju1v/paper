---
title: "딥러닝 기초 용어집"
date: 2026-07-26T16:00:00+09:00
draft: false
weight: 10
categories: ["Glossary"]
---

이 폴더의 논문들([Transformer](/posts/papers/attention-is-all-you-need/), [ResNet](/posts/papers/resnet-deep-residual-learning-for-image-recognition/), [ViT](/posts/papers/vit-an-image-is-worth-16x16-words/), [DETR](/posts/papers/detr-end-to-end-object-detection-with-transformers/), [PointNet](/posts/papers/pointnet-deep-learning-on-point-sets-for-3d-classification-and-segmentation/), [PointNet++](/posts/papers/pointnet-plus-plus-deep-hierarchical-feature-learning/))에서 반복적으로 등장하는 핵심 용어를 정리합니다.

## 기초 개념 (사전지식)

논문 본문에서는 당연히 알고 있다고 가정하고 넘어가는 배경지식입니다.

- **신경망(Neural Network) / MLP(Multi-Layer Perceptron)**: 입력을 받아 여러 층의 "가중치 곱하기 + 더하기 + 비선형 함수" 연산을 거쳐 출력을 내는 함수. 층을 여러 개 쌓으면(deep) 복잡한 패턴도 표현할 수 있어 "딥러닝"이라 부름. 이 블로그의 거의 모든 논문에서 "MLP"라고 하면 이 기본 구조를 가리킴
- **파라미터(Parameter) / 가중치(Weight)**: 신경망 내부의 숫자들로, 학습 과정에서 값이 조정되는 대상. "모델을 학습한다"는 것은 결국 이 숫자들을 데이터에 맞게 바꾸는 것
- **손실 함수(Loss Function)**: 모델의 예측이 정답과 얼마나 다른지를 하나의 숫자로 나타내는 함수. 이 값이 작을수록 모델이 잘 맞히고 있다는 뜻이며, 학습은 이 값을 최소화하는 과정
- **경사하강법(Gradient Descent)**: 손실 함수의 값을 줄이는 방향으로 파라미터를 조금씩 이동시키는 최적화 방법. "산에서 안개 속에 있을 때, 발밑 경사만 보고 한 걸음씩 낮은 쪽으로 내려가는 것"과 같은 직관
- **역전파(Backpropagation)**: 손실 함수의 값이 각 파라미터를 얼마나 조금씩 바꾸면 줄어드는지(기울기, gradient)를 출력층에서 입력층 방향으로 거꾸로 계산해나가는 알고리즘. 경사하강법이 실제로 어느 방향으로 갈지 계산해주는 절차
- **미분 가능(Differentiable)**: 어떤 계산 과정이 역전파로 기울기를 구할 수 있는 형태라는 뜻. "미분 가능한 렌더링"처럼 이 단어가 나오면 "그 계산도 역전파·학습의 대상이 될 수 있다"는 의미
- **학습(Training) vs 추론(Inference)**: 학습은 데이터를 보며 파라미터를 조정하는 단계, 추론은 학습이 끝난 모델에 새 입력을 넣어 결과만 뽑아내는 단계. 추론은 파라미터가 고정되어 있어 학습보다 훨씬 빠름
- **에폭(Epoch) / 이터레이션(Iteration) / 배치(Batch)**: 전체 학습 데이터를 한 번 다 보는 것이 1 에폭, 그 안에서 데이터를 작은 묶음(배치) 단위로 나눠 한 번 파라미터를 갱신하는 게 1 이터레이션
- **과적합(Overfitting)**: 모델이 학습 데이터의 세부 패턴(노이즈 포함)까지 외워버려서, 정작 처음 보는 데이터에는 성능이 떨어지는 현상 — "시험 문제를 이해한 게 아니라 답만 외운 것"에 비유됨
- **활성화 함수(Activation Function)**: ReLU 같이, 선형 계산 결과에 비선형성을 추가하는 함수. 이게 없으면 층을 아무리 쌓아도 결국 하나의 선형 함수와 같아져 복잡한 패턴을 표현하지 못함
- **임베딩(Embedding)**: 단어·이미지 조각·좌표 같은 이산적이거나 원시적인 입력을, 신경망이 다루기 좋은 고정 길이의 숫자 벡터로 바꾼 표현
- **소프트맥스(Softmax)**: 여러 숫자를 모두 0~1 사이 값으로, 합이 1이 되도록 변환하는 함수. "각 항목이 정답일 확률"처럼 해석할 수 있게 만들어줌
- **어텐션(Attention)의 직관**: "지금 내가 보고 있는 부분(query)이, 전체 정보(key) 중 어디에 더 집중해야 하는지 가중치를 계산하고, 그 가중치대로 실제 정보(value)를 가져오는" 메커니즘. Self-attention은 이 query/key/value가 모두 같은 시퀀스에서 나온 경우

## Transformer / Attention

- **Self-Attention (Intra-Attention)**: 같은 시퀀스 내 서로 다른 위치 간의 관계를 계산하는 attention. RNN 없이도 장거리 의존성을 직접 연결로 학습할 수 있게 함
- **장거리 의존성 문제**: 문장이 길어질수록 앞쪽 정보가 RNN을 거치며 희석되는 문제. Self-attention은 임의의 두 위치를 한 번의 연산으로 직접 연결해 이를 해결
- **Multi-Head Attention**: 여러 개의 attention을 병렬로 수행해 서로 다른 표현 부분공간의 정보를 동시에 포착
- **Positional Encoding (위치 인코딩)**: Transformer는 순서 정보가 없으므로, 사인/코사인 함수로 각 위치에 고유한 패턴을 부여해 순서 정보를 주입
- **Auto-Regressive (자기회귀)**: 디코더가 한 토큰씩 생성하며 이전에 생성한 토큰만 참조하도록 마스킹하는 방식
- **Residual Connection**: $\text{LayerNorm}(x + \text{Sublayer}(x))$ — 입력을 출력에 더해줘 깊은 네트워크에서도 그래디언트가 잘 흐르게 함 (ResNet에서 유래)
- **Layer Normalization**: 레이어 출력을 정규화해 학습을 안정화
- **연산 복잡도 트레이드오프**: Self-attention은 모든 위치를 한 번에 병렬 계산(경로 길이 O(1))하지만 시퀀스 길이의 제곱($O(n^2)$)에 비례해 비용이 커짐 — 이후 Deformable Attention, FlashAttention 등이 이를 완화

## Object Detection / Set Prediction (DETR 계열)

- **Direct Set Prediction**: 고정 개수 N개의 예측을 한 번에 병렬로 출력해, NMS(비최대 억제) 없이 중복을 제거하는 방식
- **Object Query**: 디코더에 입력되는 N개의 학습 가능한 위치 임베딩. 각 쿼리가 이미지의 서로 다른 영역/객체를 담당하도록 학습됨
- **Bipartite Matching / Hungarian Algorithm (헝가리안 알고리즘)**: 예측 집합과 정답(GT) 집합 사이의 최소 비용 1:1 매칭을 찾는 알고리즘. 이 매칭으로 순열에 무관한(permutation-invariant) 손실을 정의할 수 있어 병렬 예측이 가능해짐
- **Auxiliary Decoding Loss**: 디코더의 각 레이어마다 예측 헤드와 손실을 추가해 학습을 안정화하는 기법
- **Deformable Attention**: 전체 특징 맵이 아니라 각 쿼리마다 소수의 학습된 key point에만 attention을 수행해 연산량을 줄이는 방식 — BEVFormer, MapTR 등 인지 논문 다수가 이 위에 구축됨

## 이미지 백본 (ResNet / ViT)

- **Degradation Problem**: 네트워크가 깊어질수록 학습 오류가 오히려 커지는 현상 — 과적합이 아니라 최적화 실패가 원인
- **Residual Learning**: 레이어가 원하는 함수 $\mathcal{H}(x)$ 전체를 배우는 대신, 잔차 $\mathcal{F}(x) := \mathcal{H}(x) - x$만 학습하도록 재정식화. Identity shortcut을 통해 정보가 항상 전달되므로 깊은 네트워크도 안정적으로 학습됨
- **Bottleneck Block**: 1×1 → 3×3 → 1×1 컨볼루션 조합으로, 연산량을 유지하면서 더 깊게 쌓을 수 있게 하는 구조 (ResNet-50/101/152)
- **귀납적 편향 (Inductive Bias)**: 모델이 학습 전부터 내장하고 있는 가정. CNN은 "가까운 픽셀끼리 관련이 크다(locality)"와 "같은 패턴은 위치가 달라도 같다(translation equivariance)"를 내장하지만, ViT는 이를 제거하고 데이터로부터 직접 학습
- **스케일링 법칙 (Scaling Law)**: 데이터와 모델 크기가 커질수록 성능이 계속 향상되는 현상. ViT는 NLP의 스케일링 법칙이 비전에도 적용됨을 보임

## Point Cloud (PointNet 계열)

- **순열 불변성 (Permutation Invariance)**: N개 점의 집합은 입력 순서를 어떻게 바꿔도 같은 결과가 나와야 한다는 성질 — LiDAR 포인트 클라우드처럼 순서가 없는 데이터를 다룰 때 반드시 필요
- **대칭 함수 (Symmetric Function)**: 입력 순서에 무관한 함수(합, 최댓값 등). PointNet은 max pooling으로 순열 불변성을 보장
- **공유 MLP (Shared MLP)**: 모든 점에 동일한 가중치의 MLP를 독립적으로 적용해 점별 특징을 추출하는 방식
- **T-Net (Input/Feature Transform)**: 입력 점들을 정규 자세(canonical pose)로 정렬하기 위해 3×3(또는 64×64) 변환 행렬을 직접 예측하는 서브네트워크
- **FPS (Farthest Point Sampling)**: 이미 선택된 점들로부터 가장 먼 점을 반복적으로 골라, 전체 공간을 고르게 커버하는 샘플링 방법
- **Ball Query**: 반경 $r$ 이내의 점들을 이웃으로 정의하는 방식. kNN과 달리 실제 공간 스케일이 고정되어 특징의 공간적 일관성이 보장됨
- **계층적 특징 학습**: 저수준(점) → 중간(부분 구조) → 고수준(전체 물체) 표현을 단계적으로 쌓아가는 방식 — CNN의 계층적 특징 추출을 점 집합에 적용한 것
