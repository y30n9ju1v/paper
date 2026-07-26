---
title: "Novel View Synthesis 용어집"
date: 2026-07-26T16:05:00+09:00
draft: false
weight: 20
categories: ["Glossary"]
---

이 폴더의 논문들([NeRF](/posts/papers/nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis/), [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [3DGUT](/posts/papers/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting/), [3D Gaussian Ray Tracing](/posts/papers/3d-gaussian-ray-tracing/), [4D Gaussian Splatting](/posts/papers/4d-gaussian-splatting/), [Difix3D+](/posts/papers/difix3d-plus/), [3DGS-QA](/posts/papers/3dgs-qa-perceptual-quality-assessment-of-3d-gaussian-splatting/), [Evaluating Human Perception of NVS](/posts/papers/evaluating-human-perception-of-novel-view-synthesis-gs-nerf-dynamic-scenes/))에서 반복적으로 등장하는 핵심 용어를 정리합니다.

## 기초 개념 (사전지식)

논문 본문에서는 당연히 알고 있다고 가정하고 넘어가는 카메라·그래픽스 배경지식입니다.

- **Novel View Synthesis (새 시점 합성)**: 몇 장의 사진(기존 시점)만으로, 카메라가 놓인 적 없는 새로운 위치·각도에서 본 모습을 만들어내는 문제. "몇 장의 사진으로 그 장면을 3D로 이해해서, 다른 각도에서 다시 찍은 것처럼 보여준다"는 것이 목표
- **카메라 외부 파라미터(Extrinsics) / 포즈(Pose)**: 카메라가 3D 공간 어디에, 어느 방향을 보고 있는지를 나타내는 위치+회전 정보. "카메라가 지금 서 있는 자리와 바라보는 방향"
- **카메라 내부 파라미터(Intrinsics)**: 초점거리, 렌즈 왜곡 등 카메라 자체의 광학적 특성 — 같은 3D 점이 센서 위 어느 픽셀에 맺히는지를 결정
- **투영(Projection)**: 3D 공간의 한 점을 카메라 내부·외부 파라미터를 이용해 2D 이미지의 픽셀 좌표로 변환하는 계산. 반대로 2D 픽셀에서 3D 위치를 복원하려면 깊이 정보가 추가로 필요함(2D→3D는 정보가 부족한 "ill-posed" 문제)
- **월드 좌표계 vs 카메라 좌표계**: 장면 전체를 기준으로 한 고정 좌표계(월드)와, 각 카메라 자신을 기준으로 한 좌표계(카메라)가 있고, 이 둘을 서로 변환하는 것이 포즈(extrinsics)의 역할
- **SfM (Structure-from-Motion) / COLMAP**: 여러 장의 사진만 보고, 각 사진을 찍은 카메라의 포즈와 장면의 대략적인 3D 점 위치를 동시에 역산해내는 고전적인 컴퓨터 비전 기법. NeRF/3DGS 학습 전 단계에서 카메라 포즈를 얻는 데 널리 쓰임 (COLMAP은 이를 구현한 대표적인 오픈소스 도구)
- **RGB / Depth / Point Cloud**: RGB는 색상만 있는 일반 사진, Depth는 각 픽셀까지의 거리 정보가 추가된 이미지, Point Cloud는 3D 좌표를 가진 점들의 집합(주로 LiDAR나 depth 센서로 얻음) — 이 세 가지가 3D 재구성에서 가장 흔히 쓰이는 입력 형태
- **실시간(Real-time) 렌더링**: 사람이 끊김을 느끼지 못하는 속도(보통 30fps 이상)로 이미지를 계속 그려내는 것. NeRF가 느렸던 이유, 3DGS가 주목받은 이유가 바로 이 실시간성 여부
- **상관계수(Correlation Coefficient)의 직관**: 두 값(예: 어떤 지표의 점수와 사람이 매긴 점수)이 "같이 오르고 같이 내리는 정도"를 -1~1 사이의 숫자로 나타낸 것. 1에 가까우면 둘이 거의 같이 움직인다는(=지표가 사람 판단을 잘 예측한다는) 뜻, 0에 가까우면 둘이 서로 무관하다는 뜻. PLCC/SRCC/KRCC 같은 지표 평가 용어들이 모두 이 개념 위에 정의됨
- **평가자(관찰자, Observer) 기반 실험**: 알고리즘 성능을 사람이 아니라 여러 명의 실제 사람에게 직접 보여주고 점수를 매기게 해서 "정답"으로 삼는 실험 방식. 시력·색각 검사 같은 사전 검증과, 신뢰할 수 없는 응답을 걸러내는 사후 스크리닝을 거치는 경우가 많음

## 공통 렌더링 개념

- **체적 렌더링 (Volume Rendering)**: 카메라 광선을 따라 여러 지점의 색상·밀도를 누적해 최종 픽셀 색을 계산하는 렌더링 방식. NeRF는 연속 밀도 함수로, 3DGS는 이산적인 가우시안 불투명도로 이 계산을 수행하지만 수학적 형태는 동일함
- **알파 합성 (Alpha Compositing)**: $C = \sum_i T_i \alpha_i c_i,\ T_i = \prod_{j<i}(1-\alpha_j)$ — 앞의 물체를 통과하고 남은 빛의 투과율($T_i$)만큼만 뒤의 색이 더해지는 방식. "색유리 여러 장을 겹쳐 보는" 직관과 같음
- **불투명도 $\alpha$ / 투과율 $T$**: $\alpha$는 해당 지점의 불투명함 정도, $T$는 거기까지 도달하기 전 앞의 모든 지점을 통과하고 남은 빛의 비율
- **미분 가능 렌더링 (Differentiable Rendering)**: 렌더링 과정 전체가 미분 가능해, 정답 이미지와의 픽셀 오차를 역전파해 장면 표현(MLP 가중치 또는 가우시안 파라미터)을 직접 최적화할 수 있음

## NeRF 계열

- **Implicit Neural Representation**: 좌표(위치·방향)를 입력받아 해당 지점의 물리량(색상·밀도)을 출력하는 신경망으로 장면 전체를 표현하는 방식 — 복셀/메시 같은 명시적 구조 없이 MLP 가중치 안에 장면이 압축됨
- **Positional Encoding**: MLP가 저주파 함수를 선호하는 경향(spectral bias)을 극복하기 위해, 입력 좌표를 사인/코사인의 여러 주파수로 매핑해 고주파 디테일(날카로운 경계, 텍스처)을 학습하기 쉽게 만드는 기법
- **Hierarchical (Coarse-to-Fine) Sampling**: 레이 위에 균일하게 샘플링하면 빈 공간에 샘플이 낭비되므로, Coarse 네트워크로 중요한 영역을 먼저 파악한 뒤 그 영역에 Fine 샘플을 집중시키는 2단계 샘플링 전략

## 3D Gaussian Splatting 계열

- **3D 가우시안**: 위치 $\mu$, 공분산 $\Sigma$(=회전 $R$ × 스케일 $S$), 불투명도 $\alpha$, 색상(구면조화 계수)을 갖는 명시적·미분 가능한 장면 표현 단위. 중심에서 멀어질수록 밀도가 지수적으로 감소하는 타원체
- **이방성 공분산 (Anisotropic Covariance)**: 방향에 따라 크기가 다른 공분산으로, 원판·바늘 같은 임의의 가우시안 형태를 표현 — 회전×스케일 분해($\Sigma=RSS^TR^T$)로 항상 유효한 공분산을 보장
- **적응적 밀도 제어**: 학습 중 기울기가 큰(정보가 부족한) 영역은 가우시안을 분할·복제하고, 기여가 적은 가우시안은 제거해 가우시안 개수와 분포를 스스로 조정
- **타일 기반 래스터화**: 이미지를 16×16 타일로 나눠 GPU 캐시 효율과 병렬성을 극대화하는 3DGS의 핵심 렌더링 기법. 가우시안 키를 (타일, 깊이)로 인코딩해 단일 Radix 정렬로 깊이 순서를 처리
- **조기 종료 (Early Termination)**: 픽셀의 누적 불투명도가 임계값(보통 0.9999)을 넘으면 남은 가우시안 계산을 생략해 렌더링을 가속하는 기법
- **EWA Splatting**: 3D 가우시안을 2D 이미지 평면에 투영할 때 투영 함수를 Jacobian으로 1차 선형근사하는 기존 3DGS의 방법 — 왜곡이 큰 카메라(어안 등)일수록 근사 오차가 커진다는 한계가 있음
- **Unscented Transform (UT)**: EWA splatting의 선형화 대신, 가우시안을 7개의 대표점(Sigma point)으로 근사해 실제 비선형 투영 함수에 직접 통과시키는 방법. Jacobian 없이 임의의 카메라 모델(어안, 롤링 셔터)을 지원 가능하게 함 ([3DGUT](/posts/papers/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting/))
- **Secondary Ray / 하이브리드 렌더링**: 1차 광선(카메라→장면)은 빠른 래스터화로 처리하고, 반사·굴절 같은 2차 광선만 광선 추적으로 넘겨 정확도와 속도를 동시에 얻는 방식
- **BVH (Bounding Volume Hierarchy)**: 물체들을 계층적 바운딩 박스 트리로 구성해 광선-물체 교차를 $O(\log N)$으로 가속하는 자료구조 — 3D Gaussian Ray Tracing이 각 가우시안을 이 구조에 넣기 위한 대리 기하(프록시)를 씌움

## 동적 장면 (4D)

- **Canonical Space**: 모든 타임스텝의 기준이 되는 정규 공간 — 객체의 "기본 형태"를 여기 저장하고, 각 시점의 모습은 이 기준으로부터의 변형(deformation)으로 표현
- **Deformation Field**: 시간을 입력받아 각 가우시안의 위치·회전·크기 변형량을 예측하는 신경망. 프레임마다 별도의 3DGS를 만드는 대신, 하나의 canonical 가우시안 집합 + 변형 필드로 동적 장면을 표현해 메모리를 O(tN)에서 O(N+F)로 줄임
- **HexPlane 인코딩**: 4D(x,y,z,t) 시공간을 6개의 2D 평면(xy, xz, yz, xt, yt, zt)으로 분해해 표현하는 메모리 효율적인 인코딩 방식

## 화질 개선 / 생성 모델과의 결합

- **단일 단계 확산 모델 (Single-step Diffusion)**: 일반 확산 모델(수백~수천 스텝)과 달리, 한 번의 forward pass로 이미지를 생성/보정하는 증류된 확산 모델(SD-Turbo 등) — 재구성 렌더링의 아티팩트를 실시간에 가깝게 제거하는 데 활용 ([Difix3D+](/posts/papers/difix3d-plus/))
- **크로스-뷰 참조 혼합 (Cross-view Reference Mixing)**: 여러 참조 뷰의 정보를 attention으로 섞어, 개별 뷰를 따로 보정할 때 생기는 3D 불일치를 줄이는 기법

## 평가 지표

- **PSNR / SSIM**: 픽셀 단위 재구성 정확도(PSNR, dB)와 구조적 유사도(SSIM, 0~1) — 둘 다 높을수록 좋음
- **LPIPS**: 딥러닝 기반 지각적 유사도 지표 — 사람이 느끼는 화질 차이에 더 가깝고, 낮을수록 좋음
- **FID (Fréchet Inception Distance)**: 실제 이미지 분포와 생성/재구성 이미지 분포 사이의 유사도 — 낮을수록 좋음

## 지각 품질 평가 방법론 (Perceptual Quality Assessment)

PSNR/SSIM 같은 지표가 "사람 눈에 실제로 실사처럼 보이는가"를 얼마나 잘 대변하는지를 검증하는 별도의 연구 분야에서 쓰이는 용어입니다.

- **주관 품질 평가 (Subjective Quality Assessment)**: 알고리즘이 아니라 사람이 직접 이미지·영상을 보고 점수를 매겨 "정답"으로 삼는 평가 방식. 이렇게 모은 사람의 점수를 기준으로, 자동화된 지표(PSNR 등)가 그 점수를 얼마나 잘 예측하는지를 검증함
- **MOS (Mean Opinion Score)**: 여러 평가자가 매긴 점수의 평균값. 주관 품질 평가에서 "이 방법이 얼마나 실사처럼 보였는가"를 하나의 숫자로 요약한 것
- **참조 기반(Full-Reference, FR) vs 무참조(No-Reference, NR) 지표**: 참조 기반은 정답 이미지와 비교해서 점수를 매기는 방식(PSNR, SSIM 등), 무참조는 정답 없이 이미지/장면 자체만 보고 품질을 판단하는 방식 — 실제 서비스 환경은 정답이 없는 경우가 많아 무참조 지표의 실용성이 더 큼
- **PLCC / SRCC / KRCC**: 어떤 자동화된 지표의 점수가 사람의 MOS와 얼마나 일치하는지를 재는 상관계수들(선형/순위/켄달 순위 상관) — 1에 가까울수록 그 지표가 사람의 판단을 잘 예측한다는 뜻
- **SAMVIQ (Subjective Assessment Methodology for Video Quality, ITU-T BT.1788)**: 평가자가 모든 영상을 한 번 훑어본 뒤, 자유롭게 되돌아가며 0~100점을 매기는 표준화된 주관 평가 실험 프로토콜
- **3D 가우시안 원시 데이터 기반 평가**: 렌더링된 2D 이미지를 거치지 않고, 가우시안의 위치·색상 등 원시 파라미터 자체에서 저하 신호를 직접 탐지해 품질을 예측하는 방식 — 렌더링 비용 없이 대량의 재구성 결과를 빠르게 선별(triage)하는 데 유용 ([3DGS-QA](/posts/papers/3dgs-qa-perceptual-quality-assessment-of-3d-gaussian-splatting/))
