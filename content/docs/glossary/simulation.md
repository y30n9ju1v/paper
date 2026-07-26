---
title: "시뮬레이션 용어집"
date: 2026-07-26T16:15:00+09:00
draft: false
weight: 40
categories: ["Glossary"]
---

`autonomous-driving/simulation/` 폴더의 논문들 — [CARLA](/posts/papers/carla-an-open-urban-driving-simulator/), [LiDARsim](/posts/papers/lidarsim-realistic-lidar-simulation-by-leveraging-the-real-world/), [Street Gaussians](/posts/papers/street-gaussians-modeling-dynamic-urban-scenes/), [DrivingGaussian](/posts/papers/driving-gaussian-composite-gaussian-splatting/), [OmniRe](/posts/papers/omnire-omni-urban-scene-reconstruction/), [HUGSIM](/posts/papers/hugsim-real-time-photorealistic-closed-loop-simulator/), [Instant NuRec](/posts/papers/instant-nurec-feed-forward-3d-gaussian-reconstruction-for-driving-scene-simulation/), [Cam2Sim](/posts/papers/cam2sim-neural-scenario-reconstruction-for-closed-loop-autonomous-driving-simulation/) — 에서 반복적으로 등장하는 핵심 용어를 정리합니다.

## 기초 개념 (사전지식)

논문 본문에서는 당연히 알고 있다고 가정하고 넘어가는 시뮬레이션/테스트 배경지식입니다.

- **ADS (Autonomous Driving System)**: 자율주행 소프트웨어 스택 전체(인지+예측+계획+제어)를 가리키는 말. 이 논문들에서 "ADS가 시뮬레이터 안에서 주행한다"는 것은, 테스트 대상이 되는 자율주행 소프트웨어를 실제 대신 시뮬레이션 환경에 넣어 돌린다는 뜻
- **시나리오(Scenario) vs 씬(Scene)**: 씬은 정적인 한 장면(도로, 건물 배치 등)을, 시나리오는 그 씬 위에서 시간에 따라 벌어지는 사건(다른 차가 끼어든다, 보행자가 튀어나온다 등)까지 포함한 개념 — 회귀 테스트는 보통 "시나리오" 단위로 정의됨
- **회귀 테스트(Regression Test)**: 소프트웨어를 수정한 뒤, 이전에는 잘 되던 것들이 여전히 잘 되는지 다시 검증하는 테스트. 자율주행에서는 "새 모델/코드가 기존에 통과하던 시나리오들에서 여전히 안전하게 주행하는가"를 자동으로 반복 검증하는 것을 의미
- **에이전트(Agent)**: 시뮬레이션 안에서 스스로 판단해 행동하는 개체 — 테스트 대상 ADS(ego agent)뿐 아니라, 주변의 다른 차량·보행자도 각자의 행동 규칙에 따라 움직이는 에이전트로 취급됨
- **재현성(Reproducibility)**: 같은 조건으로 여러 번 실행해도 같은 결과가 나오는 성질. 실도로 테스트는 재현이 거의 불가능하지만, 시뮬레이션은 초기 조건을 고정하면 완벽히 재현 가능 — 회귀 테스트가 시뮬레이션을 필요로 하는 핵심 이유
- **센서 시뮬레이션 vs 행동 시뮬레이션**: 센서 시뮬레이션은 "카메라/LiDAR가 이 장면에서 어떤 데이터를 만들어낼까"를 재현하는 것(이 폴더 논문 대부분의 주제)이고, 행동 시뮬레이션은 "다른 차량·보행자가 어떻게 반응할까"를 재현하는 것 — 둘 다 갖춰야 완전한 폐루프 테스트가 됨
- **디지털 트윈(Digital Twin)**: 실제 장소·차량을 최대한 그대로 복제한 가상 버전. NuRec 계열 재구성 기술이 지향하는 목표 중 하나로, 실제 도로를 촬영한 데이터로 그 도로의 디지털 트윈을 만들어 반복 테스트에 사용

## 평가 방식론

- **오픈루프 vs 클로즈드루프 (Open-loop vs Closed-loop)**: 오픈루프 평가는 사전 수집된 데이터로 알고리즘을 테스트하지만, 알고리즘의 결정이 이후 관측에 전혀 영향을 주지 않는다 — 그저 현재 상태를 유지하기만 해도 좋은 성능처럼 보이는 함정이 있음. 클로즈드루프는 시뮬레이터가 에이전트의 제어 명령에 실시간으로 반응해, 실제 주행처럼 오류가 누적되는 피드백 루프를 형성함
- **Sim-to-Real 도메인 갭**: 시뮬레이터에서 학습·평가한 모델이 실제 환경에서는 성능이 떨어지는 현상. 가상 환경의 텍스처·조명·기하가 현실과 다르면 ADS가 인지하는 입력 분포 자체가 달라져 행동이 달라짐
- **행동 충실도 (Behavior Fidelity)**: 시뮬레이션에서의 주행 행동(완주율, 궤적, 조향 패턴 등)이 실제 주행과 얼마나 유사한지를 측정하는 관점 — 렌더링 화질 지표(PSNR 등)와는 별개로, "실제로 같은 결정을 내리는가"를 직접 검증
- **롱테일 이벤트 (Long-tail Events)**: 희귀하지만 안전에 치명적인 시나리오. 실도로 주행만으로는 충분한 샘플을 모을 수 없어 시뮬레이션/재구성이 필요한 핵심 이유

## 신경 재구성 기반 시뮬레이션 (NeRF/3DGS 계열)

- **Per-scene Optimization**: 장면 하나마다 처음부터 최적화를 다시 수행하는 전통적인 NeRF/3DGS 재구성 방식. 화질은 높지만 장면당 수 분~수 시간이 걸려, 매일 수백만 클립을 수집하는 자율주행 플릿 규모에는 비용이 감당되지 않음
- **Feed-Forward 재구성**: 최적화 없이 학습된 네트워크의 한 번의 forward pass만으로 3D 장면(가우시안 등)을 예측하는 방식. Per-scene 대비 수백~수천 배 빠르지만, 학습 분포를 크게 벗어난 장면에는 일반화가 제한적일 수 있음 (Instant NuRec)
- **정적/동적 레이어 분리 (Static/Dynamic Decomposition)**: 장면을 고정된 배경(건물, 도로)과 움직이는 객체(차량, 보행자)로 나눠 각각 다른 방식으로 모델링하는 접근. 정적 배경은 한 번만 재구성하면 되고, 동적 객체만 시간에 따른 궤적/변형을 추가로 다루면 되므로 효율적
  - **Rigid Nodes**: 강체(차량 등)를 캐노니컬 공간의 가우시안 + 시간별 6-DoF 포즈로 표현
  - **SMPL Nodes**: 보행자를 인체 파라메트릭 모델(SMPL, 형태+관절 자세)로 구동해 관절 단위 제어를 지원
  - **Deformable Nodes**: 자전거 등 정형화되지 않은 물체를 변형 네트워크로 표현하는 템플릿 없는 방식
  - (OmniRe, Street Gaussians, DrivingGaussian이 각기 다른 조합으로 이 분리를 구현)
- **Sky Node / Sky Cubemap**: 하늘처럼 무한히 먼 영역은 3D 가우시안 대신 별도의 환경 텍스처(구면/큐브맵)로 표현하는 관행
- **LiDAR Prior**: LiDAR로 얻은 정확한 깊이 정보를 가우시안 초기화에 활용해 기하학적 정확도를 높이는 기법
- **캐노니컬 공간 (Canonical Space)**: 시간·포즈에 무관한 객체의 표준 좌표계 — 여기서 학습된 물체를 매 프레임의 실제 포즈로 변환해 장면에 배치
- **레이어드 3DGS 표현**: 정적 가우시안, 동적(궤적 기반) 가우시안, 하늘 큐브맵, 카메라별 색보정(ISP)까지 포함한, 그 자체로 시뮬레이션에 바로 쓸 수 있는 완전한 장면 표현 (Instant NuRec)

## LiDAR 시뮬레이션

- **Surfel (Surface Element)**: 점이 아니라 디스크 형태를 가진 렌더링 기본 단위. 중심 위치와 법선 방향을 가져 폐색(가려짐) 처리와 충돌 검사에 효율적 (LiDARsim)
- **Raydrop**: LiDAR 레이가 물체에 닿았는데도 반사 신호가 임계값 이하라 수신기가 감지하지 못해 포인트가 생성되지 않는 현상 — 실제 LiDAR의 노이즈 특성을 시뮬레이션에 반영할 때 중요
- **극좌표 이미지 그리드 (Polar Image Grid)**: LiDAR 스캔을 (수직 채널 수) × (수평 해상도)의 2D 이미지로 표현하는 방식 — CNN(U-Net 등)으로 처리하기 쉬운 형태로 LiDAR 데이터를 재구성

## 게임 엔진 기반 시뮬레이터 (CARLA)

- **클라이언트-서버 구조**: 서버가 물리 시뮬레이션·렌더링을 담당하고, 클라이언트(에이전트)가 제어 명령을 보내고 센서 데이터를 받는 구조
- **의사 센서 (Pseudo-sensor)**: RGB 카메라 외에 지면 진실 깊이(ground-truth depth), 의미론적 분할 맵처럼 시뮬레이터만이 제공할 수 있는 "정답" 형태의 센서 출력

## Gaussian Splatting을 시뮬레이터에 결합하는 도구

- **"실행이 곧 검증"(Execution-as-Validation)**: 배포된 ADS가 렌더링된 카메라 스트림을 그대로 입력받아 운전하므로, 폐루프 주행이 성공적으로 끝나는지 자체가 생성된 관측(렌더링)의 실용적 품질 검증이 된다는 설계 철학 (Cam2Sim)
- **좌표계 정합 (Coordinate Alignment)**: 신경 재구성 도구(Nerfstudio 등)가 복원한 카메라 포즈와 시뮬레이터(CARLA 등)의 좌표계를 일치시키는 과정 — 이게 어긋나면 시뮬레이터의 물리·에이전트 움직임과 렌더링이 서로 다른 곳을 가리키게 됨
- **하늘 마스크 (Sky Mask)**: 3DGS 학습 시 하늘 영역을 제외해, 모델이 도로·건물 같은 정적 장면 콘텐츠에 집중하도록 만드는 전처리

## 평가 지표

- **PDM Score / HD-Score / ADS**: 충돌 없음(No Collision), 주행 가능 영역 준수(Drivable Area Compliance), 충돌까지 남은 시간(TTC), 승차감(Comfort) 등을 조합한 폐루프 시뮬레이션 종합 점수 (NAVSIM, DriveArena, HUGSIM 등에서 변형되어 쓰임)
- **Route Completion (경로 완주율)**: 충돌·이탈 없이 목표 경로를 끝까지 주행한 비율 — 폐루프 평가에서 특히 중요 (충돌 시 시뮬레이션이 즉시 종료되므로)
- **IoU (도로/자동차/배경)**: 재구성된 시나리오의 기하학적 정확도를 세그멘테이션 클래스별 겹침 비율로 측정하는 지표 (Cam2Sim)
