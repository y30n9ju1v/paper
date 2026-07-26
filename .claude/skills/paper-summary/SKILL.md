---
name: paper-summary
group: content
description: 논문 PDF를 챕터별로 요약하고 핵심 개념을 정리하여 Hugo 블로그 포스트로 생성합니다.
parameters:
  - name: pdf_path
    description: 처리할 PDF 파일의 경로
    required: true
---

# Paper Summary Skill

논문 PDF를 받아서 챕터별로 요약하고 핵심 개념을 정리한 마크다운 파일을 `content/docs/` 디렉토리에 생성합니다.

## Usage

```
/paper-summary <pdf-file-path>
```

## Example

```
/paper-summary ~/Downloads/research-paper.pdf
```

## What it does

- PDF 파일을 입력받습니다
- 논문의 제목, 저자, 발행년도를 추출합니다
- 논문의 핵심 요약(한 줄 요약)과 핵심 기여점(Key Contributions, 이 논문이 스스로 주장하는 성과)을 작성합니다
- 논문의 Related Works(및 Introduction의 관련 연구 언급)를 바탕으로 이 분야의 연구 흐름(어떤 접근에서 어떤 접근으로 이어져 왔는지)을 정리하고, 그 흐름 속에서 저자가 짚은 기존 한계점(gap, 여러 개면 모두 리스트로)과 이 논문의 접근 방식을 명확히 정리합니다
- 각 챕터/섹션별로 상세한 요약, 핵심 개념, 수식 및 수식 해설을 작성합니다
- 수식은 KaTeX 형식($$...$$)으로 작성하며, **각 수식마다 변수 설명과 직관적 의미를 초보자도 이해할 수 있도록** 추가합니다
- 주요 실험 결과(벤치마크, 성능 비교, 데이터셋)를 정리합니다
- 날짜는 KST(한국 표준시) 타임스탐프로 작성합니다 (예: `2026-04-10T08:30:00+09:00`)
  `year`는 논문의 실제 발행년도(arXiv 공개 또는 학회 발표 기준)를 정수로 작성합니다 (예: `year: 2017`)
  `references`는 이 논문이 직접 인용하거나 기반으로 하는 논문들의 slug 리스트입니다. slug는 `content/docs/` 아래 파일명(`.md` 제외)을 사용합니다. 블로그에 없는 논문은 제외합니다.
- 논문 주제에 맞는 카테고리 폴더를 판단하여 `content/docs/<category>/논문-제목.md` 파일을 Hugo 블로그 포스트 형식으로 생성합니다
   - `deep-learning/` — Transformer, CNN, RL 등 기반 딥러닝 논문
   - `novel-view-synthesis/` — NeRF, 3DGS 등 뷰 합성 논문
   - `generative-models/` — Diffusion, GAN, World Model 등 생성 모델 논문
   - `autonomous-driving/perception/` — BEV, Occupancy, HD Map, 3D Detection 등
   - `autonomous-driving/simulation/` — 센서 시뮬레이터, 클로즈드 루프 시뮬레이션
   - `autonomous-driving/planning/` — E2E 자율주행, Planning 논문
   - `autonomous-driving/dataset/` — 데이터셋 및 벤치마크 논문

## Output format

생성되는 마크다운 파일은 다음 구조를 따릅니다:

```markdown
---
title: "논문 제목"
date: 2026-04-10T08:30:00+09:00
draft: false
categories: ["Papers"]
year: 2017
references:
  - "attention-is-all-you-need"
  - "nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis"
---

## 💡 한 줄 요약
논문의 핵심 목표와 성과를 1~2문장으로 요약합니다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Author Name et al.
- **발행년도**: 2026
- **주요 기여점**:
  1. 기여 1
  2. 기여 2

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
이 논문이 속한 분야가 어떤 흐름으로 발전해왔는지, 그리고 그 흐름 속에서 이 논문이 어떤 한계를 극복하기 위해 작성되었는지 설명합니다.
- **연구 흐름**: (예: A 방식 → B 방식으로 발전) 이 논문 이전에 어떤 접근들이 있었고, 어떤 순서/계보로 이어져 왔는지
- **기존 한계점**:
  1. 한계 1
  2. 한계 2
- **이 논문의 접근 방식**: 설명

## 📑 목차
- Chapter 1: ...
- Chapter 2: ...

## 🛠️ Chapter 1: [제목]

**요약**
해당 챕터의 주요 내용을 초보자도 이해할 수 있도록 상세하게 설명합니다.

**핵심 개념**
- **개념1**: 상세한 설명
- **개념2**: 상세한 설명

**수식 예제**

$$C = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

**수식 설명**
이 수식은 여러 객체를 투명도를 고려하여 합치는 방법을 나타냅니다:
- **$C$**: 최종 출력 색상 (우리가 화면에 보는 색)
- **$c_i$**: i번째 객체의 색상
- **$\alpha_i$**: i번째 객체의 투명도 (0 = 완전 투명, 1 = 완전 불투명)
- **$\sum$**: 모든 객체를 하나씩 더한다는 의미
- **$\prod_{j=1}^{i-1}(1 - \alpha_j)$**: 앞의 모든 객체들을 지나온 빛의 투과율
  - 예: 1번째 객체가 반투명(투명도 0.5)이면, 뒤의 객체는 50%만 보임

## 📊 주요 실험 및 결과 (Experiments & Results)
- **사용 데이터셋 / 벤치마크**: 데이터셋 이름
- **주요 성과**: 기존 SOTA 대비 성능 향상치 및 정량적 결과 정리

## 💡 결론 및 시사점 (Conclusion & Insights)
Key Contributions가 논문 저자의 주장이라면, 여기서는 다 읽고 난 뒤의 평가와 실무 적용 관점을 담습니다.
- 논문의 결론과 실무적/연구적 시사점을 정리합니다.
- **한계점 및 아쉬운 점**: 논문에서 스스로 밝히지 않았거나, 읽으면서 느낀 아쉬운 점
```
