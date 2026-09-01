---
title: "OpenCalib: A Multi-sensor Calibration Toolbox for Autonomous Driving"
date: 2026-09-01T09:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "Calibration"]
tags: ["Autonomous Driving", "Calibration", "LiDAR", "Camera", "Radar", "IMU", "Sensor Fusion", "Toolbox"]
year: 2022
references: []
---

## 💡 한 줄 요약
LiDAR·카메라·레이더·IMU 등 자율주행 차량의 모든 센서 조합에 대해 수동/자동/공장/온라인 캘리브레이션을 지원하는 오픈소스 툴박스 **OpenCalib**을 공개하고, CARLA 시뮬레이션 기반 벤치마크 데이터셋으로 정확도를 검증했다.

## 📌 개요 및 핵심 기여 (Key Contributions)
- **저자**: Guohang Yan, Liu Zhuochun, Chengjie Wang, Chunlei Shi, Pengjin Wei, Xinyu Cai, Tao Ma, Zhizheng Liu, Zebin Zhong, Yuqian Liu, Ming Zhao, Zheng Ma, Yikang Li (Shanghai AI Laboratory / SenseTime, Autonomous Driving Group)
- **발행년도**: 2022 (arXiv:2205.14087)
- **코드**: [PJLab-ADG/SensorsCalibration](https://github.com/PJLab-ADG/SensorsCalibration)
- **주요 기여점**:
  1. **전 시나리오 커버리지**: 수동(Manual), 자동(Automatic), 공장(Factory), 온라인(Online) 4가지 운용 시나리오와 LiDAR-카메라, LiDAR-LiDAR, LiDAR-IMU, 레이더-LiDAR, 레이더-카메라, 카메라-IMU 등 사실상 모든 센서 조합의 캘리브레이션을 단일 툴박스로 통합
  2. **타겟 유무에 따른 이중 파이프라인**: 체커보드·원형 홀 보드 등 물리적 타겟을 쓰는 정밀 캘리브레이션과, 차선·전봇대·지면 등 주행 환경의 자연 특징만으로 캘리브레이션하는 타겟리스(target-less) 방식을 모두 제공해 실차 배포 제약(정비소 방문 불가 등)에 대응
  3. **의존성 제거 및 견고화**: OpenCV의 체커보드 코너 검출기에 의존하지 않는 자체 캘리브레이션 보드 인식 알고리즘을 새로 설계해 다양한 조도·각도에서도 안정적으로 동작
  4. **CARLA 기반 벤치마크 데이터셋 공개**: 시뮬레이터가 제공하는 고정밀 Ground Truth 포즈를 활용해 정량 평가가 어려운 캘리브레이션 문제에 대한 검증 기반 마련

## 🎯 관련 연구 흐름 및 기존 한계 (Related Work & Motivation)
- **연구 흐름**: 초기 캘리브레이션 연구는 Zhang의 체커보드 기반 카메라 내부 파라미터 추정법처럼 단일 센서·단일 알고리즘에 집중되었고, 이후 LiDAR-카메라 외부 파라미터 추정(체커보드+원형 홀 보드 결합, PnP 기반 정합), 타겟리스 방식(차선·에지 등 자연 특징 정합), 센서 내 상태 추정(카메라-IMU, LiDAR-IMU 시간/회전 오프셋 온라인 추정) 등으로 각 센서 쌍마다 독립적인 알고리즘이 산발적으로 제안되어 왔다. OpenCalib은 이런 개별 연구 성과를 하나의 실전 배포 가능한 툴박스로 통합한 첫 시도다.
- **기존 한계점**:
  1. **완결된 오픈소스 툴박스 부재**: 자율주행에 필요한 다양한 센서 조합·운용 시나리오(공장 출하 시점, 정비소, 주행 중)를 모두 아우르는 공개 캘리브레이션 소프트웨어가 없어, 각 회사·연구팀이 유사한 알고리즘을 반복 구현해야 했다.
  2. **타겟 기반 방식의 실전 제약**: 체커보드 등 물리적 타겟을 이용한 방식은 정확하지만, 차량이 정비소를 방문해야 하므로 대규모 차량 운용이나 주행 중 센서 드리프트 보정에는 적용하기 어렵다.
  3. **정량 평가 기준 부재**: 실제 도로에서는 LiDAR-카메라 등 외부 파라미터의 절대 정답(Ground Truth)을 구하기 어려워, 대부분의 선행 연구가 정성적(시각적) 검증에 그쳤다.
- **이 논문의 접근 방식**: 타겟 기반·타겟리스 알고리즘을 모두 구현해 상황별로 선택 가능하게 하고, CARLA 시뮬레이터로 정답 포즈가 알려진 합성 데이터를 생성해 정량 평가의 기반을 마련한다.

## 📑 목차
- Chapter 1: 수동 타겟리스 캘리브레이션 (LiDAR-카메라, LiDAR-LiDAR, 레이더-LiDAR/카메라)
- Chapter 2: 자동 타겟 기반 캘리브레이션 (카메라 내부 파라미터, LiDAR-카메라 결합 보드)
- Chapter 3: 자동 타겟리스 캘리브레이션 (IMU 헤딩, 도로 기반 LiDAR-카메라, LiDAR-IMU, 다중 LiDAR)
- Chapter 4: 공장 및 온라인 캘리브레이션
- Chapter 5: CARLA 벤치마크 데이터셋 및 평가

## 🛠️ Chapter 1: 수동 타겟리스 캘리브레이션

### 1. 요약
정비소나 특수 장비 없이, 사람이 GUI에서 슬라이더/키보드로 회전·이동 값을 조정하며 두 센서 데이터를 눈으로 정합시키는 가장 기초적인 방식입니다. LiDAR-카메라(포인트 클라우드를 이미지에 투영), LiDAR-LiDAR(포인트 클라우드 정합), 레이더-LiDAR(2D 레이더 포인트와 3D 포인트 클라우드 정합), 레이더-카메라(호모그래피 기반 BEV 정합)의 4개 도구가 제공됩니다.

### 2. 수식 및 설명

LiDAR 포인트 $\mathbf{p}_L \in \mathbb{R}^3$를 카메라 좌표계로 옮긴 뒤 이미지 평면에 투영하는 기본 식은 다음과 같습니다.

$$\mathbf{p}_C = \mathbf{R} \mathbf{p}_L + \mathbf{t}$$

$$\mathbf{p}_{\text{img}} = \mathbf{K} \cdot \pi(\mathbf{p}_C)$$

- **$\mathbf{R}, \mathbf{t}$**: LiDAR → 카메라 외부 파라미터(회전·이동)로, 사용자가 GUI에서 직접 조정하는 대상
- **$\mathbf{K}$**: 카메라 내부 파라미터 행렬 (초점거리, 주점)
- **$\pi(\cdot)$**: 3D → 2D 원근 투영 함수

사용자는 투영된 포인트 클라우드가 이미지 상의 실제 물체(차선, 차량 윤곽 등)와 겹치도록 $\mathbf{R}, \mathbf{t}$를 조정하며, 툴은 이 결과를 즉시 시각화해 실시간 피드백을 제공합니다.

---

## 🛠️ Chapter 2: 자동 타겟 기반 캘리브레이션

### 1. 요약
체커보드로 카메라 내부 파라미터를 구하고, 체커보드에 원형 구멍을 뚫은 결합 보드(Joint Board)로 카메라-LiDAR 외부 파라미터를 동시에 정밀 추정합니다. 타겟 기반 방식이므로 반복 재현성과 정확도가 가장 높습니다.

### 2. 수식 및 설명

**카메라 내부 파라미터**: Zhang의 방법으로 여러 각도에서 촬영한 체커보드로 초점거리·주점·왜곡 계수를 추정하며, 아래 왜곡 평가 지표로 품질을 검증합니다.

$$e_{\text{rms}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} d(\mathbf{x}_i, l)^2}, \qquad e_{\max} = \max_i d(\mathbf{x}_i, l)$$

- **$d(\mathbf{x}_i, l)$**: 실제로는 직선이어야 할 체커보드 격자선 위의 샘플 점 $\mathbf{x}_i$와, 이를 최소자승으로 피팅한 직선 $l$ 사이의 거리 — 왜곡 보정이 잘 될수록 0에 가까워짐

**LiDAR-카메라 결합 캘리브레이션**: 체커보드 코너의 재투영 오차 $J_{\text{board}}$와, 원형 홀 중심의 재투영 오차 $J_{\text{lidar}}$를 가중합한 목적 함수를 최소화합니다.

$$J_{\text{sum}} = \alpha \cdot J_{\text{board}} + \beta \cdot J_{\text{lidar}}, \qquad (\alpha=1,\ \beta=60)$$

$\beta$를 $\alpha$보다 훨씬 크게 준 것은, LiDAR 포인트가 이미지 코너보다 훨씬 희소하고 노이즈에 민감해 상대적으로 더 강하게 맞춰야 최종 정합 품질이 안정적이기 때문입니다.

---

## 🛠️ Chapter 3: 자동 타겟리스 캘리브레이션

### 1. 요약
물리적 타겟 없이 주행 환경 자체의 특징(차선, 지면, 전봇대 등)을 이용합니다. IMU 헤딩 오프셋 보정, 시맨틱 세그멘테이션(BiSeNet-V2)으로 차선·기둥을 추출해 맞추는 도로 기반 LiDAR-카메라 캘리브레이션, 슬라이딩 윈도우 기반 LiDAR-IMU 캘리브레이션, 지면 정합과 NICP(Normal ICP)를 결합한 다중 LiDAR 캘리브레이션이 포함됩니다.

### 2. 수식 및 설명

**IMU 헤딩 보정**: 차량이 직선 구간을 달릴 때 GPS 기반 진행 방향 $\gamma_{gd}$와 IMU가 보고하는 방향 $\gamma_{IMU}$의 차이를 구간별로 평균 내어 오프셋을 구합니다.

$$\gamma_{\text{offset}} = \frac{1}{|S_l|} \sum_{i \in S_l} (\gamma_{gd,i} - \gamma_{IMU,i})$$

- **$S_l$**: 직선으로 판단된(회전이 거의 없는) 구간의 프레임 집합. B-spline으로 궤적을 스무딩해 직선 구간을 안정적으로 판별합니다.

**LiDAR-IMU 캘리브레이션**: 슬라이딩 윈도우 내에서 평면/에지로 분류된 포인트들이 정렬됐을 때 분산이 최소가 되도록, 즉 국소 좌표계에서의 공분산 행렬 고유값을 최소화하는 방향으로 외부 파라미터를 최적화합니다. 여러 좌표 변환이 체인으로 연결되므로 좌표계 연쇄 법칙(coordinate chain rule)을 이용해 자코비안을 유도합니다.

**다중 LiDAR 캘리브레이션**: 먼저 각 LiDAR가 관측한 지면의 법선 벡터 사이의 외적(cross product)으로 대략적인 회전축을 구해 초기 정렬을 맞춘 뒤, 두 포인트 클라우드가 겹치는 복셀(공간을 옥트리로 분할)의 부피를 최소화하는 방향으로 NICP를 적용해 정밀 정렬합니다.

---

## 🛠️ Chapter 4: 공장 및 온라인 캘리브레이션

### 1. 요약
공장 출하 단계에서는 소실점(vanishing point)과 호모그래피로 카메라를, 원형 홀 보드로 LiDAR를 빠르게 캘리브레이션합니다. 체커보드·원형 보드·수직 보드·ArUco 마커·원형 홀·AprilTag 등 6종의 보드를 지원해 생산 라인 환경에 맞춰 선택할 수 있습니다. 온라인 캘리브레이션은 차량이 실제로 주행하는 동안 카메라-IMU, LiDAR-IMU의 시간/회전 오프셋을 지속적으로 재추정하고, 레이더의 경우 정지된 물체(예: 정차 차량)를 기준으로 차량 중심 대비 요(yaw) 각도 오차를 온라인으로 보정합니다.

### 2. 핵심 개념
- **공장 캘리브레이션**: 짧은 시간에 대량의 차량을 처리해야 하므로, 표준화된 보드와 고정된 촬영 위치를 전제로 속도와 재현성을 최적화
- **온라인 캘리브레이션**: 주행 중 진동·충격 등으로 발생하는 센서 간 미세한 오프셋 드리프트를 정비소 방문 없이 실시간으로 추적·보정 — 대규모 차량 플릿(fleet) 운영에 필수적인 유지보수 요소

---

## 📊 주요 실험 및 결과 (Experiments & Results)
- **벤치마크 데이터**: CARLA 시뮬레이터로 Hesai Pandar64 LiDAR(10Hz), Basler acA1920-40gc 카메라(10Hz, FOV 30°/60°/120° 세 종류), Delphi ESR 2.5 레이더(20Hz), IMU(100Hz)를 모사해 카메라 내부 파라미터, 온라인 캘리브레이션, 레이더 외부 파라미터 시나리오를 각각 구성 — 시뮬레이터가 제공하는 정답 포즈를 Ground Truth로 사용
- **평가 방식의 한계**: 실차 데이터에 대해서는 정답 포즈를 구하기 어려워, 논문 스스로도 "정량 평가에는 Ground Truth가 필요하다"고 인정하며 실차 결과는 포인트 클라우드-이미지 정합 품질, 다중 프레임 지도 작성(mapping) 품질, 서로 다른 LiDAR 간 물체 정합 일관성 등 시각적 검증에 의존
- **카메라 왜곡 평가**: 직선 피팅 기반 RMS/최대 오차 지표로 내부 파라미터 캘리브레이션 품질을 정량화한 것이 실질적으로 가장 명확한 수치 지표
- **실차 검증**: Hesai Pandar64, Basler 카메라, Delphi 레이더, NovAtel GNSS로 구성된 실제 차량 플랫폼에서 툴박스 전체 파이프라인이 정상 동작함을 확인

## 💡 결론 및 시사점 (Conclusion & Insights)

### 1. 결론
OpenCalib은 "센서 캘리브레이션은 자율주행 시스템의 기초"라는 문제의식 아래, 그동안 개별 논문·사내 코드로 흩어져 있던 자율주행 캘리브레이션 알고리즘들을 하나의 오픈소스 툴박스로 통합한 최초의 시도입니다. 인식(perception)·센서 융합·측위(localization/mapping) 등 후속 파이프라인의 정확도는 결국 이 캘리브레이션 정밀도에 상한이 걸리기 때문에, 화려한 신규 알고리즘 제안보다 실전 배포 가능한 완결성에 방점을 둔 실용적 기여가 크다.

### 2. 실무적 시사점
- **연구/개발 재사용성**: BEVFusion, CenterPoint 등 멀티센서 인식 모델을 실차에 적용하려는 팀이 캘리브레이션 알고리즘을 직접 구현할 필요 없이 바로 활용 가능
- **플릿 운영 관점**: 온라인·타겟리스 캘리브레이션 도구는 차량이 대수가 많아질수록 정비소 방문 비용이 커지는 문제를 실질적으로 완화

### 3. 한계점 및 아쉬운 점
- 실차 데이터에 대한 정량적 정확도(Ground Truth 대비 오차)를 제시하지 못하고 시각적 정성 평가에 크게 의존한다 — 저자들도 향후 시뮬레이터 기반 대규모 정량 벤치마크를 공개하겠다고 예고
- 논문 시점 기준 v0.1 릴리스로, 학습 기반(딥러닝) 캘리브레이션 방법은 "정밀도가 중요한 캘리브레이션 과제에는 아직 덜 안정적"이라며 채택하지 않음 — 최신 학습 기반 기법과의 정량 비교는 부재
- 타겟 기반 방식이 가장 정확하지만 특정 배포 시나리오(대규모 플릿, 원격지 차량)에서는 여전히 실용성이 떨어짐

---

## 참고 자료
- [공식 GitHub 저장소 (PJLab-ADG/SensorsCalibration)](https://github.com/PJLab-ADG/SensorsCalibration)
- [arXiv 논문 (arXiv:2205.14087)](https://arxiv.org/abs/2205.14087)

*관련 논문: [PointPillars](/posts/papers/pointpillars-fast-encoders-object-detection-point-clouds/), [CenterPoint](/posts/papers/centerpoint-center-based-3d-object-detection-and-tracking/), [BEVFusion](/posts/papers/bevfusion-multi-task-multi-sensor-fusion/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/), [Waymo Open Dataset](/posts/papers/waymo-open-dataset/)*
