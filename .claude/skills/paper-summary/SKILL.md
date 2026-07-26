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
   - `deep-learning/` — Transformer, CNN 등 인지/재구성 모델이 의존하는 기반 딥러닝 논문
   - `novel-view-synthesis/` — NeRF, 3DGS 등 범용 뷰 합성/재구성 기술 논문
   - `autonomous-driving/perception/` — BEV, Occupancy, HD Map, 3D Detection 등
   - `autonomous-driving/simulation/` — 신경 재구성 기반 센서 시뮬레이터, 클로즈드 루프 시뮬레이션, 회귀 테스트 도구
   - `autonomous-driving/dataset/` — 데이터셋 및 벤치마크 논문

   새 카테고리가 필요하면(예: 기존 폴더 어디에도 맞지 않는 논문) 먼저 사용자에게 물어보고 폴더를 만듭니다. 파일명은 항상 소문자 kebab-case 슬러그를 사용합니다(예: `bevformer.md`, `carla-an-open-urban-driving-simulator.md`) — 대문자나 카멜/파스칼 케이스는 쓰지 않습니다.
- 논문 포스트 작성/수정이 끝나면 `content/docs/glossary/`의 해당 카테고리 용어집도 함께 업데이트합니다 (아래 "Glossary Maintenance" 참고).

## Glossary Maintenance

논문을 새로 추가하거나 기존 논문을 업데이트할 때마다, 그 논문의 **핵심 개념** 섹션에 나온 용어 중 아직 용어집에 없는 것을 해당 카테고리의 용어집 파일에 추가합니다.

- 카테고리 → 용어집 파일 매핑:
  - `deep-learning/` → `content/docs/glossary/deep-learning.md`
  - `novel-view-synthesis/` → `content/docs/glossary/novel-view-synthesis.md`
  - `autonomous-driving/perception/` → `content/docs/glossary/perception.md`
  - `autonomous-driving/simulation/` → `content/docs/glossary/simulation.md`
  - `autonomous-driving/dataset/` 등 아직 전용 용어집이 없는 카테고리는, 논문이 3개 이상 쌓이고 용어집이 필요해 보이면 사용자에게 새 용어집 파일을 만들지 물어봅니다.
- 추가 절차:
  1. 새 논문의 **핵심 개념** 항목들을 훑어보고, 해당 용어집 파일에 이미 있는 용어(같은 개념의 다른 표기 포함)는 건너뜁니다.
  2. 새로운 용어는 기존 용어집의 주제별 섹션(`## BEV(Bird's-Eye-View) 공통 개념` 등) 중 가장 잘 맞는 곳에 추가합니다. 맞는 섹션이 없으면 새 `##` 섹션을 만듭니다.
  3. 정의는 그 논문 파일에 이미 쓴 문장을 재사용하되, 다른 논문에서도 통용되는 일반적인 설명이 되도록 다듬습니다 (특정 논문의 실험 수치보다는 개념 자체의 정의에 집중).
  4. 용어집 최상단의 "## 기초 개념 (사전지식)" 섹션은 그 분야를 처음 접하는 사람을 위한 배경지식용이므로, 논문에 나온 전문 용어가 아니라 "이걸 몰라도 논문을 못 읽을 정도"인 진짜 사전지식(예: 카메라 포즈, 손실 함수, 바운딩 박스 같은 개념)일 때만 추가합니다.
  5. 용어집 안의 각 논문 링크 목록(파일 상단 소개 문장)에 새 논문 링크도 추가합니다.
  6. 다른 용어집 파일이나 논문 포스트를 향한 링크(`/posts/papers/<slug>/`)를 추가했다면, 그 slug의 파일이 실제로 존재하는지 확인합니다.

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
