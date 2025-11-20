import torch
import torch.nn as nn
from ai.models.catanet_arch import CATANet
import os

# 이 스크립트는 CATANet-L 모델을 로드하는 커스텀 로더 역할을 합니다.
# OPTIN 프레임워크의 gen_vision_model.py를 직접 수정하는 대신,
# ai 폴더 내에서 독립적으로 모델 로딩 로직을 관리합니다.

def get_catanet_teacher_model(weights_path: str = None, upscale: int = 4):
    """
    CATANet-L 교사 모델을 생성하고 (선택적으로) 사전 훈련된 가중치를 로드합니다.

    Args:
        weights_path (str, optional): 사전 훈련된 가중치 파일(.pth)의 경로.
                                      None이면 무작위로 초기화된 모델을 반환합니다.
        upscale (int): 모델의 업스케일링 비율 (예: 2, 3, 4). CATANet 아키텍처에 전달됩니다.

    Returns:
        torch.nn.Module: CATANet-L 모델 인스턴스.
    """
    print(f"CATANet-L 모델을 생성합니다. Upscale 비율: {upscale}")

    # 1. CATANet 모델 아키텍처 인스턴스 생성
    # ai/models/catanet_arch.py에 정의된 CATANet 클래스를 사용합니다.
    # in_chans는 입력 이미지 채널 수 (RGB의 경우 3), upscale은 초해상도 비율입니다.
    model = CATANet(in_chans=3, upscale=upscale)

    # 2. 사전 훈련된 가중치 로드 (선택 사항)
    if weights_path:
        if os.path.exists(weights_path):
            print(f"사전 훈련된 가중치를 로드합니다: {weights_path}")
            # 가중치 파일을 로드하고 모델의 state_dict에 적용합니다.
            # map_location='cpu'를 사용하여 GPU가 없는 환경에서도 로드할 수 있도록 합니다.
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            print("가중치 로드 성공.")
        else:
            print(f"경고: 지정된 가중치 파일 '{weights_path}'를 찾을 수 없습니다. "
                  "무작위로 초기화된 모델을 사용합니다.")
    else:
        print("가중치 경로가 제공되지 않았습니다. 무작위로 초기화된 모델을 사용합니다.")

    # 모델을 평가 모드로 설정합니다. (프루닝 시에는 중요)
    model.eval()

    return model
