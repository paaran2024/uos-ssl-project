# CATANet-L 기반 Student Model 아키텍처 설계

이 문서는 `CATANet-L`을 Teacher 모델로 사용하여 **Dynamic Pruning**과 **Feature Distillation**을 통해 경량화된 Student 모델을 개발하는 프로젝트의 폴더 구조와 설계 철학을 정의합니다.

## 1. 목표

- **Teacher Model**: `CATANet-L` (크고 성능이 좋은 원본 모델)
- **Student Model**: `CATANet-L`을 경량화한 빠르고 효율적인 모델 (CATANet-XS)
- **핵심 기술**:
  1.  **Dynamic Pruning**: 훈련 과정 혹은 훈련 후에 모델의 불필요한 부분을 동적으로 제거하여 모델 사이즈를 줄입니다.
  2.  **Feature Distillation**: Student 모델이 Teacher 모델의 최종 결과뿐만 아니라, 중간 계층의 특징(Feature Map)을 모방하도록 훈련하여 성능 손실을 최소화합니다.

## 2. 제안 폴더 구조

`CATANet`의 모듈성과 `FaKD`의 역할 분리 구조를 차용하여 다음과 같은 폴더 구조를 제안합니다. `ai` 폴더 내에 새로운 `student_model` 디렉토리를 생성하여 프로젝트를 진행합니다.

```
ai/
├── student_model/
│   ├── archs/
│   │   ├── student_arch.py         # Student 모델의 기본 신경망 구조 정의
│   │   └── components/
│   │       └── pruning_layers.py   # Dynamic Pruning을 지원하는 커스텀 레이어
│   ├── data/                       # 데이터셋 폴더 (원본 CATANet의 것을 공유 또는 링크)
│   ├── losses/
│   │   ├── distillation_loss.py    # Teacher-Student 간 Feature 유사도를 측정하는 손실 함수
│   │   └── pruning_loss.py         # Pruning을 촉진하기 위한 손실 함수 (선택 사항)
│   ├── models/
│   │   └── distillation_model.py   # Distillation 훈련 과정을 총괄하는 메인 모델 클래스
│   ├── options/
│   │   └── train_student_distill.yml # Student 모델 훈련(Distillation)을 위한 설정 파일
│   ├── pruning/
│   │   └── strategy.py             # 동적 Pruning 기준 및 실행 전략 정의
│   ├── teacher/
│   │   └── loader.py               # 사전 훈련된 CATANet-L Teacher 모델을 불러오는 유틸리티
│   ├── pretrained/
│   │   ├── teacher/
│   │   │   └── CATANet-L.pth       # Teacher 모델 가중치
│   │   └── student/                # 훈련된 Student 모델의 가중치를 저장할 위치
│   ├── train.py                    # 훈련 시작을 위한 메인 스크립트
│   └── inference.py                # 훈련된 Student 모델을 사용한 추론 스크립트
│
├── references/
│   ├── CATANet-main/
│   └── FaKD/
├── architecture.md                 # (본 문서)
└── ...
```

## 3. 구조 설계 해설

- **`student_model/`**: 새로운 Student 모델 개발을 위한 독립적인 프로젝트 폴더입니다. 기존 참고자료와 분리하여 명확성을 유지합니다.

- **`archs/`**: `CATANet`의 구조를 따라 모델의 신경망 구조를 정의합니다.

  - `student_arch.py`: Pruning이 적용되기 전의 기본 학생 모델 구조를 정의합니다.
  - `components/pruning_layers.py`: 일반적인 레이어(Conv2D 등)를 Pruning 로직이 추가된 커스텀 레이어로 래핑(wrapping)하여, Pruning을 모듈화하고 재사용성을 높입니다.

- **`losses/`**: 훈련에 사용될 손실 함수를 분리하여 관리합니다.

  - `distillation_loss.py`: Teacher와 Student의 중간 Feature Map 간의 차이를 계산하는 로직을 담당합니다. (예: L1, L2, 또는 FaKD에서 사용된 손실 함수)

- **`models/`**: 훈련의 전체적인 흐름을 제어하는 상위 모델 클래스를 정의합니다.

  - `distillation_model.py`: Teacher 모델과 Student 모델을 모두 멤버로 포함합니다. 훈련 시 데이터가 입력되면, Teacher와 Student를 각각 통과시켜 Feature Map을 얻고, `distillation_loss`를 계산하여 최종 손실을 반환하는 등 전체 훈련 파이프라인을 총괄합니다.

- **`pruning/`**: Dynamic Pruning 알고리즘의 핵심 로직을 별도 폴더로 분리합니다.

  - `strategy.py`: 어떤 뉴런이나 가중치를 '중요하지 않다'고 판단할 것인지에 대한 기준(예: L1-norm)과, 이를 바탕으로 모델에서 실제 제거하는 로직을 포함합니다. `distillation_model`이 이 모듈을 호출하여 Pruning을 수행합니다.

- **`teacher/`**: Teacher 모델을 깔끔하게 불러오기 위한 유틸리티 폴더입니다.

  - `loader.py`: `pretrained/teacher/`에 저장된 가중치를 이용해 `CATANet-L` 모델을 생성하고, 훈련에 방해되지 않도록 `eval` 모드로 설정하고 가중치를 고정(freeze)하는 역할을 담당합니다.

- **`options/`**: `CATANet`의 장점인 설정 파일 기반의 실험 관리를 그대로 사용합니다. 학생 모델의 구조, 러닝 레이트, distillation loss의 가중치 등 모든 하이퍼파라미터를 `.yml` 파일로 관리하여 재현성을 확보합니다.

- **`pretrained/`**: 가중치 파일을 역할(Teacher/Student)에 따라 명확하게 분리하여 관리의 용이성을 높입니다.

- **`train.py` / `inference.py`**: 각각 훈련과 추론을 시작하는 명확한 진입점(entry point) 역할을 합니다.
