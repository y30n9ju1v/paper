---
title: "Cam2Sim: Neural Scenario Reconstruction for Closed-Loop Autonomous Driving Simulation"
date: 2026-07-26T15:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving"]
tags: ["3D Gaussian Splatting", "Autonomous Driving", "Simulation", "CARLA", "Closed-Loop", "Sim-to-Real"]
year: 2026
references:
  - "CARLA-An-Open-Urban-Driving-Simulator"
  - "3d-gaussian-splatting"
---

## 💡 한 줄 요약
실제 주행 녹화 데이터를 도로 기하·궤적·주차 차량까지 포함한 CARLA 실행 가능 시나리오로 자동 재구성하고, 그 위에 3D Gaussian Splatting으로 렌더링을 입혀 시뮬레이터의 겉모습을 실제 카메라 영상에 가깝게 만듦으로써, 폐루프 ADS 테스트에서 시뮬레이션 전용 베이스라인보다 실제 주행과 훨씬 유사한 행동을 재현하는 도구 Cam2Sim을 제안한다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Davide Jannussi (Politecnico di Torino), Stefano Carlo Lambertenghi, Constantin Carste (TUM), Andrea Stocco (TUM, fortiss)
- **발행년도**: 2026 (arXiv:2607.04770, ASE '26 Tools and Datasets track)
- **주요 기여점**:
  1. ROS bag 형태의 실제 주행 녹화를 카메라 기반 3D 탐지(FCOS3D)와 LiDAR 기반 탐지(PointPillars)로 주차 차량까지 자동 추출하고, OpenStreetMap을 OpenDRIVE 맵으로 변환해 **CARLA 실행 가능 시나리오**를 자동 생성하는 end-to-end 파이프라인 구현
  2. COLMAP으로 포즈·희소 포인트를 추정하고 Nerfstudio(splatfacto)로 학습한 **3D Gaussian Splatting을 CARLA 시나리오에 결합**해, 시뮬레이터의 텍스처·조명 차이로 인한 인지 입력 왜곡을 줄임
  3. 궤적 리플레이 모드와 ADS 폐루프 실행 모드를 모두 지원하도록 설계해, 카메라 기반 end-to-end 모델(DAVE-2)이 실제 카메라 스트림 대신 CARLA/GS 렌더링을 그대로 받아 운전하게 하는 **행동 충실도(behavior fidelity) 검증 체계**를 구축
  4. 뮌헨 주택가 450m 실주행 시나리오에서, 시뮬레이션 전용 베이스라인은 3/3 실패(완주율 20% 내외)한 반면 GS로 증강한 Cam2Sim은 실제 주행과 동일하게 **3/3 모두 완주**함을 실증

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: 자율주행 테스트는 실도로 주행(비싸고 위험) → 녹화 데이터셋 기반 오프라인 평가(폐루프 상호작용을 못 봄) → 게임 엔진 기반 시뮬레이터(CARLA 등, 폐루프는 가능하지만 sim-to-real 겉모습 격차) 순으로 발전해왔다. 이후 GAN·확산 모델·NeRF·3DGS 같은 신경 렌더링이 시각적으로 사실적인 장면 생성을 가능하게 했지만, DriveRecon 등 상당수는 "뷰포인트를 렌더링"하는 데 그치고 실제로 실행 가능한 시뮬레이션으로 이어지지는 않았다. Gambi et al.(2019) 같은 선행 연구는 경찰 보고서로부터 실행 가능한 시나리오를 재구성했지만 인지 수준의 sim-to-real 격차는 다루지 않았다.
- **기존 한계점**:
  1. 실도로 테스트는 비용이 크고 위험하며 재현이 어려움
  2. 녹화된 데이터셋 기반 평가는 ADS와 환경 사이의 폐루프 상호작용(ADS의 결정이 환경에 영향을 주는 것)을 담아내지 못함
  3. 시뮬레이션 전용 접근은 폐루프는 가능하지만, 텍스처·조명·장면 외관의 차이가 ADS가 인지하는 입력 분포를 바꿔 실제 주행과 다른 행동을 유발할 수 있음(sim-to-real gap)
- **이 논문의 접근 방식**: 실제 녹화에서 도로 기하·궤적·주차 차량을 재구성해 CARLA에서 그대로 재생 가능한 시나리오를 자동으로 만들고, 그 위에 Gaussian Splatting 렌더링을 얹어 ADS가 보는 카메라 입력을 실제 영상에 가깝게 만든다 — "배포된 ADS가 렌더링된 카메라 스트림을 직접 소비하므로, 폐루프 실행이 성공한다는 것 자체가 생성된 관측의 실용적 검증"이라는 것이 핵심 논리다.

## 📑 목차
- Chapter 1: 데이터 추출과 시나리오 재구성
- Chapter 2: Gaussian Splatting 준비와 결합
- Chapter 3: 리플레이·폐루프 실행과 행동 충실도 평가

## 🛠️ Chapter 1: 데이터 추출과 시나리오 재구성

**요약**
파이프라인의 첫 단계는 ROS bag 녹화를 구조화된 데이터셋으로 변환하는 것입니다. 최소 입력은 카메라 이미지와 포즈이고, LiDAR·오도메트리·조향각 등은 선택적으로 함께 추출됩니다. 이후 주차 차량을 자동으로 검출하고, OpenStreetMap 데이터를 CARLA가 쓰는 OpenDRIVE 맵 포맷으로 변환해 시나리오의 뼈대를 만듭니다.

**핵심 개념**
- **센서 스트림 동기화**: 서로 다른 주기로 기록된 센서 데이터를 타임스탬프 기준 보간으로 ego 포즈에 맞춰 정렬
- **이중 경로 주차 차량 검출**: 카메라 경로는 FCOS3D로 단안 3D 탐지 후 월드 좌표로 변환해 반복 검출을 클러스터링하고 신뢰도 가중 평균으로 최종 위치를 계산하며, LiDAR 경로는 PointPillars 검출기(+수동 보정 UI)를 사용
- **OpenStreetMap → OpenDRIVE 변환**: Overpass API로 지리 좌표 지도를 가져온 뒤 CARLA가 이해하는 도로망 포맷으로 변환

## 🛠️ Chapter 2: Gaussian Splatting 준비와 결합

**요약**
CARLA 시나리오만으로는 여전히 게임 엔진 특유의 인공적인 겉모습이 남습니다. Cam2Sim은 선택된 프레임에서 자차(ego) 후드를 제거하고 하늘 마스크(SegFormer, Cityscapes로 학습)를 생성한 뒤, COLMAP으로 카메라 포즈와 희소 3D 포인트를 추정하고 Nerfstudio(splatfacto/splatfacto-big)로 3DGS를 학습해, 이 렌더링 결과를 CARLA 좌표계에 정합시켜 최종 시각 출력으로 사용합니다.

**핵심 개념**
- **하늘 마스크로 정적 장면에 집중**: 하늘 영역을 제외하고 학습시켜, 모델이 도로·건물 같은 정적 장면 콘텐츠에 집중하도록 유도
- **좌표계 정합**: Nerfstudio가 복원한 카메라 포즈와 데이터셋의 주석(annotation)을 매칭해 정합 파라미터를 추정 — 이렇게 해야 CARLA의 ego 차량 움직임과 GS 렌더링이 같은 좌표계에서 일치함
- **"실행이 곧 검증"이라는 설계 철학**: 배포된 ADS가 결국 렌더링된 카메라 스트림을 그대로 입력으로 받으므로, 폐루프 주행이 성공적으로 끝나는지 자체가 생성된 관측의 실용적 품질 검증이 됨

## 🛠️ Chapter 3: 리플레이·폐루프 실행과 행동 충실도 평가

**요약**
Cam2Sim은 두 가지 실행 모드를 지원합니다. 리플레이 모드는 기록된 궤적을 그대로 따라가며 CARLA 렌더링과 GS 렌더링을 나란히 비교할 수 있게 하고, 폐루프 모드는 실제로 ADS(현재는 카메라 기반 end-to-end 모델 DAVE-2)가 CARLA 원본 이미지 또는 GS로 증강된 이미지를 받아 직접 운전하게 합니다. 뮌헨 주택가 450m 구간(양방향 통행, 제한속도 30km/h, 주차 차량으로 차선이 일부 좁아진 구간)에서 이 둘을 실제 주행과 비교했습니다.

**핵심 개념**
- **재구성 기하 품질**: 도로/자동차/배경 3개 클래스에 대한 IoU로 측정 (평균 IoU 0.774, 도로 0.847, 배경 0.889, 자동차 0.577 — 자동차 클래스의 편차가 가장 큼)
- **행동 충실도 지표**: 완주율/실패율, 궤적 유사도(Fréchet distance), 코리더 이탈 비율·초과 거리, 조향 떨림(steering jitter)

**실험 결과 (Table 1 요약)**

| 지표 | 실제 주행 | Cam2Sim | 시뮬레이션 전용 |
|---|---|---|---|
| 실패율 | 0/3 | 0/3 | 3/3 |
| 완주율(%) | 100–100–100 | 100–100–100 | 21–23–20 |
| Fréchet Distance(m) | 0.00 | 2.22 | – |
| 코리더 이탈 비율(%) | 0.0 | 67.3 | – |
| 평균 조향 떨림(rad/s) | 1.167 | 0.867 | – |

- **핵심 발견**: 시뮬레이션 전용 베이스라인은 세 번 모두 시나리오를 완주하지 못한 반면(오프로드 이탈 2회, 충돌 1회), GS로 증강한 Cam2Sim은 세 번 모두 실제 주행과 동일하게 완주 — "GS 기반 렌더링이 표준 시뮬레이터 렌더링 대비 시각적 격차를 줄이고, 실제 주행 참조와의 행동 유사성을 개선한다"

## 📊 주요 실험 및 결과 (Experiments & Results)
- **사용 데이터셋 / 벤치마크**: Fortuna 연구 차량(개조된 VW Passat Variant GTE, 카메라+Velodyne LiDAR+GNSS/INS)으로 촬영한 뮌헨 주택가 450m 실주행 시나리오, 총 3,144개 리플레이 프레임
- **주요 성과**: 위 Chapter 3의 IoU·행동 충실도 결과가 이 논문의 핵심 실험이며, 이미지 생성 품질(3.2절)에 대한 정량 지표는 논문에 아직 "TO DO"로 표시되어 있어 이번 버전에는 포함되지 않음

## 💡 결론 및 시사점 (Conclusion & Insights)
Key Contributions가 논문 저자의 주장이라면, 여기서는 다 읽고 난 뒤의 평가와 실무 적용 관점을 담습니다.
- "시뮬레이션 전용 베이스라인은 실패하는데 GS 증강판은 실제 주행처럼 성공한다"는 결과는, 회귀 테스트에서 렌더링 품질이 실제로 정성적 판정(성공/실패)을 뒤집을 수 있다는 것을 명확히 보여준 사례로, NuRec류 신경 재구성이 왜 회귀 테스트 인프라에 필수적인지를 직접적으로 뒷받침하는 근거 자료로 쓸 수 있음
- CARLA라는 기존 오픈소스 시뮬레이터의 물리·맵 파이프라인은 그대로 두고 "카메라 관측"만 신경 렌더링으로 교체하는 구조라, 처음부터 새 시뮬레이터를 만들지 않고도 기존 CARLA 기반 테스트 자산을 재활용하면서 sim-to-real 격차만 줄이고 싶을 때 참고할 만한 아키텍처
- **한계점 및 아쉬운 점**:
  - 현재는 정적 장면 재구성에 집중되어 있어 동적 객체(다른 차량, 보행자)의 재구성/재생은 다루지 않음
  - 카메라 기반 end-to-end 모델(DAVE-2) 하나로만 검증되어, LiDAR 기반이나 더 복잡한 모듈형 스택에서도 같은 결론이 성립할지는 추가 검증이 필요
  - 단일 시나리오(뮌헨 주택가 450m)로만 평가되어 일반화 가능성은 아직 제한적이며, 이미지 생성 품질에 대한 정량 지표도 이번 버전 논문에는 빠져 있음

---

*관련 논문: [CARLA](/posts/papers/CARLA-An-Open-Urban-Driving-Simulator/), [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/)*
