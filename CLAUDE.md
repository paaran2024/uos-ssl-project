# UOS SSL Project - CATAScaler

## Project Overview
- **Name**: CATAScaler (Image/Video Super-Resolution with Pruned CATANet-L)
- **Institution**: 서울시립대학교 (Seoul National University of Science and Technology)
- **Purpose**: CATANet-L 모델에 pruning + knowledge distillation 적용하여 50% MAC 감소된 최적화 super-resolution 모델 구축
- **Branch**: `ai` (현재 개발 브랜치), `main` (안정 브랜치)

## Architecture

```
uos-ssl-project/
├── ai/                    # AI 모델 최적화 프레임워크
│   ├── _references/       # 외부 레퍼런스 (CATANet, FaKD, OPTIN)
│   ├── models/            # 모델 아키텍처 정의
│   ├── prune/             # Pruning 알고리즘 (OPTIN 기반)
│   ├── analysis/          # 모델 분석 및 재구축 스크립트
│   ├── evals/             # 평가 함수
│   ├── utils/             # 유틸리티 함수
│   ├── basicsr/           # BasicSR 프레임워크 어댑터
│   ├── weights/           # 사전 학습된 모델 가중치
│   └── datasets/          # 테스트 데이터셋 (Set5, Set14)
│
└── app/                   # Flutter 크로스 플랫폼 앱
    └── lib/               # Dart 소스 코드
```

## Key Technologies

### AI Module
- **Framework**: PyTorch
- **Model**: CATANet-L (Channel Attention-based aTtention Network)
- **Pruning**: OPTIN framework
- **Knowledge Distillation**: FaKD (Feature Affinity KD)
- **Dependencies**: numpy, scipy, scikit-image, opencv-python, Pillow, tqdm

### Mobile App
- **Framework**: Flutter 3.9.2+
- **Language**: Dart
- **Platforms**: Android, iOS, macOS, Windows, Linux, Web
- **Key Packages**: image_picker, video_player

## Core Commands

### AI Module
```bash
# 의존성 설치
pip install -r ai/requirements.txt

# Pruning 실행
python ai/run_catanet_pruning.py --config ai/config_catanet.yml

# Fine-tuning (Knowledge Distillation)
python ai/finetune_pruned_model.py --config ai/config_catanet.yml \
    --teacher_weights ai/weights/CATANet-L_x2.pth \
    --pruned_weights ai/weights/catanet_pruned.pth

# Pruned 모델 재구축
python ai/analysis/rebuild_pruned_model.py

# 성능 분석
python ai/analysis/calculate_performance.py
```

### Flutter App
```bash
cd app
flutter pub get          # 의존성 설치
flutter run              # 실행
flutter build android    # Android APK 빌드
flutter build ios        # iOS 빌드
```

### Windows 자동화
```batch
ssl-final-ai-activator.bat
```

## Key Files

### AI Module
| File | Purpose |
|------|---------|
| `run_catanet_pruning.py` | Pruning 워크플로우 메인 진입점 |
| `finetune_pruned_model.py` | KD를 사용한 pruned 모델 fine-tuning |
| `config_catanet.yml` | Pruning 설정 (MAC 제약: 50%) |
| `models/catanet_arch.py` | CATANet-L 신경망 아키텍처 |
| `prune/main_prune.py` | OPTIN pruning 알고리즘 구현 |
| `analysis/rebuild_pruned_model.py` | 물리적으로 축소된 모델 재구축 |

### Flutter App
| File | Purpose |
|------|---------|
| `lib/main.dart` | 앱 진입점, 탭 네비게이션 |
| `lib/picture_tab.dart` | 이미지 선택 및 처리 UI |
| `lib/video_tab.dart` | 비디오 선택 및 처리 UI |

## Workflow
1. **Pruning**: OPTIN으로 CATANet-L의 MAC 50% 감소
2. **Fine-tuning**: FaKD로 성능 손실 복구
3. **Rebuild**: 물리적으로 축소된 모델 아키텍처 재구축
4. **Analysis**: Original vs Pruned 성능 비교
5. **Deploy**: Flutter 앱으로 모바일 배포

## Configuration
- `ai/config_catanet.yml`: Pruning 설정 (모델 경로, 데이터셋, MAC 제약)
- `app/pubspec.yaml`: Flutter 패키지 설정
- `app/analysis_options.yaml`: Dart 린팅 규칙

## Notes
- 모델 가중치 파일(`.pth`)은 2-2.2MB 크기
- DIV2K 학습 데이터셋은 별도 다운로드 필요 (250GB+)
- Set5, Set14는 테스트용 경량 데이터셋
