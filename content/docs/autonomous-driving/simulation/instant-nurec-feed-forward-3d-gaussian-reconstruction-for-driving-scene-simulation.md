---
title: "Instant NuRec: Feed-Forward 3D Gaussian Reconstruction for Driving Scene Simulation"
date: 2026-07-26T14:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving"]
tags: ["3D Gaussian Splatting", "Autonomous Driving", "Simulation", "Feed-Forward", "Closed-Loop", "NuRec"]
year: 2026
references:
  - "3d-gaussian-splatting"
  - "3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting"
  - "omnire-omni-urban-scene-reconstruction"
  - "waymo-open-dataset"
---

## 💡 한 줄 요약
멀티뷰 주행 로그를 한 번의 forward pass만으로 정적/동적 3D 가우시안, 하늘 큐브맵, 카메라별 ISP 보정까지 포함한 완전히 시뮬레이션 가능한 3DGS 월드로 변환해, 기존 per-scene 최적화(NuRec, ~75분) 대비 1000배 이상 빠른 약 1.5초 만에 동등한 폐루프 시뮬레이션 품질을 달성한다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Jiahui Huang, Jiawei Ren, Michal Tyszkiewicz, Xin Kang, Seung Wook Kim, Shengyu Huang, Laura Leal-Taixé, Zan Gojcic, Sanja Fidler 외 (NVIDIA)
- **발행년도**: 2026 (arXiv:2607.14203)
- **주요 기여점**:
  1. 정적 가우시안, 동적(궤적 기반) 가우시안, 하늘 큐브맵, 카메라별 ISP 색보정을 한 번의 forward pass로 동시에 예측하는 **레이어드 3DGS 표현**과 이를 만드는 feed-forward 인코더-디코더 아키텍처 설계
  2. 픽셀 단위 예측과 가우시안 개수를 분리하는 **쿼리 포인트 기반 3DGS 디코더**(Dense/Selective 두 전략)로, 화질 손실을 최소화하면서 가우시안 수를 최대 3배까지 절감
  3. 사전학습 → 컨텍스트(깊이·법선·시맨틱·모션) 학습 → GS 학습의 **3단계 학습 커리큘럼**과, 긴 시퀀스를 겹치는 청크로 나눠 합치는 **frustum-ownership 병합**으로 학습 안정성과 긴 클립 처리를 동시에 확보
  4. NVIDIA AV 데이터 약 4만 개 클립으로 학습해, Waymo Open Dataset에서 기존 feed-forward 방법 대비 최대 2.01dB PSNR 개선을 달성하고, 140개 시나리오의 폐루프 정책 평가에서 기존 per-scene 최적화 NuRec과 **동일한 정책 순위**를 재현함으로써 대체 가능성을 검증

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: NeRF/3DGS가 사진 같은 새 시점 합성의 표준을 세운 뒤, Urban Radiance Fields·Neural Scene Graphs가 정적/동적 장면 분리를, OmniRe가 비강체(non-rigid) 액터까지 분리를 확장했고, 3DGRT/3DGUT가 왜곡 카메라 지원을 더했다. 이 계보는 모두 장면 하나마다 별도로 최적화(per-scene optimization)해야 했는데, 최근에는 Zero-1-to-3, STORM, DepthSplat, Depth-Anything-3, DGGT 같은 feed-forward 재구성 방법들이 등장해 최적화 없이 한 번의 네트워크 추론으로 3D를 복원하는 방향으로 발전해왔다. 이 논문은 이 feed-forward 흐름을 자율주행 장면 재구성에 본격적으로 적용한다.
- **기존 한계점**:
  1. NuRec을 포함한 기존 per-scene 최적화 방법은 장면 하나당 수 분~수 시간이 걸리고, LiDAR·다중 주행 포즈·시맨틱 마스크·큐보이드 트랙 같은 풍부한 보조 입력이 필요함
  2. 이런 비용 구조로는 최신 자율주행 차량 플랫폼이 하루에 수집하는 수백만 개의 주행 클립을 도저히 다 재구성할 수 없음
  3. GAIA-1, DriveDreamer 같은 생성형 월드 모델은 그럴듯한 관측을 빠르게 만들어내지만 2D 픽셀만 출력할 뿐, 명시적인 3D 상태(재구성)를 제공하지 않아 시뮬레이션 자산으로 바로 쓰기 어려움
- **이 논문의 접근 방식**: 멀티뷰 이미지와 포즈를 입력받아 정적/동적 가우시안, 하늘, ISP 보정을 한 번에 예측하는 공유 인코더-다중 디코더 구조를 설계하고, 4만 개 규모의 실제 주행 클립으로 학습해 재구성 비용을 "1회 forward pass"로 상각한다.

## 📑 목차
- Chapter 1: 문제 정의와 레이어드 3DGS 출력
- Chapter 2: 공유 인코더와 쿼리 기반 디코더
- Chapter 3: 3단계 학습 커리큘럼과 긴 시퀀스 처리

## 🛠️ Chapter 1: 문제 정의와 레이어드 3DGS 출력

**요약**
입력은 V대의 카메라가 T개의 시간 프레임(2~4Hz)에 걸쳐 찍은 RGB 이미지와, 각 이미지의 6-DoF 포즈·카메라 내부 파라미터입니다. 출력은 정적 가우시안, 동적 가우시안, 하늘 큐브맵, 카메라별 ISP 보정 4가지로 구성된 "레이어드" 3DGS 표현으로, 이 조합만으로 바로 시뮬레이션에 쓸 수 있는 장면이 완성됩니다.

**핵심 개념**
- **정적 레이어 $G^s$**: 위치 $\mu$, 스케일 $s$, 회전 쿼터니언 $q$, 불투명도 $\alpha$, 색상 $c$, 법선 $n$, 시맨틱 클래스(도로/비도로)를 갖는 $N_s$개의 가우시안
- **동적 레이어 $G^d$**: 3개의 시간 지점(knot)으로 정의된 구간별 선형 궤적을 갖는 $N_d$개의 가우시안 — 차량·보행자 같은 움직이는 물체를 표현
- **하늘 큐브맵과 ISP 보정**: 하늘은 별도의 큐브맵 이미지로, 카메라마다 다른 화이트밸런스·감마·비네팅 차이는 3×4 아핀 변환으로 흡수

**수식 예제**

$$\mu(t) = (1-\lambda)\mu_k + \lambda\mu_{k+1}, \quad \lambda = \frac{t-t_k}{t_{k+1}-t_k}$$

**수식 설명**
- **$\mu(t)$**: 시간 $t$에서의 가우시안 위치
- **$\mu_k, \mu_{k+1}$**: 앞뒤 두 시간 지점(knot)에서의 위치
- **$\lambda$**: 두 knot 사이에서 현재 시간이 차지하는 비율(0~1)
- **직관**: 3개의 지점(과거·현재·미래)만 예측해두고, 그 사이는 직선으로 보간해 움직이는 물체의 위치를 표현 — 복잡한 비강체 동작(보행자 관절 움직임 등)까지는 못 담지만 차량처럼 강체에 가까운 움직임은 충분히 표현

## 🛠️ Chapter 2: 공유 인코더와 쿼리 기반 디코더

**요약**
모든 카메라·시간의 이미지를 하나의 alternating-attention Vision Transformer(Depth-Anything-3 설계를 따름)로 인코딩해 공유 특징을 얻고, 이 특징으로부터 깊이·법선·시맨틱을 먼저 예측한 뒤, 그 깊이맵에서 뽑은 쿼리 포인트가 다시 인코더 특징을 cross-attention으로 참조해 최종 가우시안 속성(스케일·회전·불투명도)을 예측합니다.

**핵심 개념**
- **Intra/Cross-image attention**: 인코더 레이어가 한 이미지 내부의 attention과 여러 뷰 사이의 attention을 번갈아 수행해, 카메라 간 정보를 공유하면서도 각 뷰의 디테일을 보존
- **쿼리 포인트로 가우시안 수 분리**: 픽셀마다 하나씩 가우시안을 만드는 대신(Dense, 뷰당 약 35만 개), 스트라이드로 묶은 토큰 클러스터 하나가 여러 개의 가우시안을 예측하게 해(Selective, 뷰당 약 12만 개) 화질 손실을 최소화하며 개수를 약 3배 줄임
- **모션 디코더**: 같은 쿼리 포인트를 입력으로 받아, 타임스탬프로 조건화된 얕은 Transformer(AdaLN)로 미래/과거 시점의 변위 $(\Delta_-, \Delta_+)$를 예측해 3개의 궤적 knot을 생성

**수식 예제**

$$D_{gi} \le \min_j D_{gj} + \delta$$

**수식 설명**
- **$D_{gi}$**: 가우시안 $g$의 중심에서 청크 $i$의 가장 가까운 카메라까지의 유클리드 거리
- **의미**: 긴 클립을 겹치는 여러 청크로 나눠 처리한 뒤 합칠 때, 각 가우시안을 "자신을 가장 가까이서 관측한 청크"에만 남기는 규칙 — 이 frustum-ownership 병합이 없으면 청크 경계에서 중복·품질 저하가 크게 발생 (ablation에서 PSNR이 29.93→26.10으로 하락)

## 🛠️ Chapter 3: 3단계 학습 커리큘럼과 긴 시퀀스 처리

**요약**
전체 손실 $\mathcal{L} = \mathcal{L}_{context} + \mathcal{L}_{motion} + \mathcal{L}_{render}$을 한 번에 학습하지 않고, 사전학습 → 컨텍스트(깊이/법선/시맨틱/모션) 학습 → 렌더링(GS) 학습 순서로 단계를 나눕니다. 특히 3단계에서는 인코더를 얼려(freeze) 렌더링 손실의 불안정한 그래디언트가 이미 잘 학습된 기하 특징을 망가뜨리지 않도록 보호합니다.

**핵심 개념**
- **Stage 1 사전학습**: Depth-Anything-3 프로토콜을 따라 깊이·광선 예측으로 백본을 초기화
- **Stage 2 컨텍스트 학습**: LiDAR와 자동 라벨링(시맨틱 분할, 큐보이드 트래킹)으로 깊이·법선·시맨틱·모션을 감독 — 렌더링 손실은 아직 사용하지 않음
- **Stage 3 GS 학습**: 인코더는 고정하고 ISP·GS 디코더만 렌더링 손실(L1 + LPIPS + 하늘 투명도)로 학습 — "인코더를 얼리면 고분산 렌더링 그래디언트가 잘 조건화된 기하 특징을 오염시키는 것을 막는다"는 것이 저자들의 설명
- **학습 데이터**: NVIDIA AV 데이터 플랫폼의 약 4만 개 필터링된 주행 클립(카메라 최대 5대, 30Hz로 300~600프레임, LiDAR 포함), 8개 노드에서 약 6일 학습

## 📊 주요 실험 및 결과 (Experiments & Results)
- **사용 데이터셋 / 벤치마크**: Waymo Open Dataset(공개 벤치마크, DepthSplat/STORM/Depth-Anything-3/DGGT와 비교), NVIDIA 내부 데이터(기존 NuRec과 직접 비교), Pandar128 LiDAR 데이터(LiDAR 재구성 확장 평가)
- **주요 성과**:
  - Waymo Open Dataset(2초 윈도우, 4프레임 컨텍스트)에서 최고 성능의 feed-forward 경쟁자 DGGT(PSNR 26.25) 대비 Instant NuRec은 **PSNR 28.26**로 2dB 이상 개선, 특히 동적 영역 PSNR은 21.76→24.93으로 격차가 더 큼
  - NVIDIA 내부 데이터에서 per-scene 최적화 NuRec(~75분) 대비 **약 1.5초**(1000배 이상 빠름)로 비슷한 수준의 화질·탐지 지표 재현
  - 140개 시나리오 × 6회 반복(20초 롤아웃, 500ms마다 재계획)의 **폐루프 정책 평가**에서, VaVAM·Alpamayo 등 5개 정책 설정에 대해 Instant NuRec과 기존 NuRec의 **정책 순위가 완전히 동일** — "폐루프 정책 평가에서 per-scene 최적화의 신뢰할 만한 대체재"임을 실증
  - Ablation: 3단계 학습 대신 단일 단계로 학습하면 PSNR 29.93→27.65로 하락, LPIPS 손실 제거 시 26.81로 하락(블러 발생), frustum-ownership 병합 제거 시 26.10으로 가장 크게 하락

## 💡 결론 및 시사점 (Conclusion & Insights)
Key Contributions가 논문 저자의 주장이라면, 여기서는 다 읽고 난 뒤의 평가와 실무 적용 관점을 담습니다.
- 폐루프 정책 평가에서 기존 NuRec과 정책 순위가 완전히 동일했다는 결과가 이 논문에서 가장 실무적으로 중요한 대목이다 — 재구성 품질 지표(PSNR 등)가 다소 다르더라도, "회귀 테스트 용도로 바꿔 써도 결론이 바뀌지 않는다"는 것을 직접 검증했기 때문에, 클라우드 회귀 테스트 파이프라인에 feed-forward 재구성을 도입할 때의 핵심 리스크(품질 저하로 인한 잘못된 판정)를 상당 부분 해소해준다
- 1000배 이상의 속도 차이는 "하루 수백만 클립" 규모의 fleet 데이터를 전수 재구성해 회귀 테스트 후보로 쓸 수 있게 한다는 의미로, 지금까지 샘플링해서 일부만 재구성하던 워크플로우 자체를 바꿀 수 있는 잠재력이 있음
- **한계점 및 아쉬운 점**:
  - 가우시안 예산과 화질 사이에 여전히 트레이드오프가 있어, per-scene 최적화 수준의 디테일(얇은 구조물 등)을 완전히 따라잡으려면 더 많은 가우시안이 필요
  - 학습에 쓰인 카메라 리그 분포에서 크게 벗어난 셋업(낮게 장착된 카메라, 어안 전용 등)은 파인튜닝이 필요해 일반화 범위가 학습 데이터에 묶여 있음
  - 3-knot 구간별 선형 궤적으로는 보행자 관절 움직임처럼 1초 이내의 비강체 동작을 세밀하게 표현하지 못함
  - 현재는 겹치는 청크를 나눠 처리한 뒤 합치는 방식이라, 이전 재구성을 조건으로 쓰는 스트리밍 방식으로의 확장이 향후 과제로 남음

---

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [3DGUT](/posts/papers/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting/), [OmniRe](/posts/papers/omnire-omni-urban-scene-reconstruction/), [Waymo Open Dataset](/posts/papers/waymo-open-dataset/)*
