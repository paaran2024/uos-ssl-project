import torch
import torch.onnx
import os
import sys

# 프로젝트 루트 경로 설정 (models 폴더를 찾기 위함)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.catanet_arch import CATANet

def run():
    # ================= [설정 영역] =================
    # 변환할 자식 모델(CATANet-XS)의 가중치 경로
    # (가장 성능이 좋은 finetuned 버전을 기본으로 설정)
    WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'weights', 'catanet_finetuned_feature_kd.pth')
    
    # 저장할 ONNX 파일명
    OUTPUT_ONNX_PATH = os.path.join(os.path.dirname(__file__), 'catanet_child_x4.onnx')
    
    # 업스케일 배율 (x4 가정)
    UPSCALE_FACTOR = 4
    # ==============================================

    print(f"--- [Step 1] PyTorch -> ONNX 변환 시작 ---")
    print(f"대상 모델: {WEIGHTS_PATH}")

    if not os.path.exists(WEIGHTS_PATH):
        print(f"오류: 가중치 파일을 찾을 수 없습니다.")
        return

    # 1. 모델 구조 생성 (CPU 모드)
    device = torch.device('cpu')
    model = CATANet(upscale=UPSCALE_FACTOR)
    model.to(device)

    # 2. 가중치 로드
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    
    # 'params' 키가 있으면 내부를, 없으면 전체를 사용
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint

    # 자식 모델이므로 일부 구조가 다를 수 있어 strict=False로 유연하게 로드
    try:
        model.load_state_dict(state_dict, strict=False)
        print("가중치 로드 완료 (strict=False 적용됨)")
    except Exception as e:
        print(f"가중치 로드 실패: {e}")
        return

    model.eval()

    # 3. 더미 데이터 생성 (입력 크기 예시: 1x3x64x64)
    dummy_input = torch.randn(1, 3, 64, 64, device=device)

    # 4. ONNX로 내보내기
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_ONNX_PATH,
        export_params=True,
        opset_version=11,           # 호환성이 좋은 버전 11 사용
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={              # 입력 크기가 변해도 동작하도록 설정
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"변환 완료: {OUTPUT_ONNX_PATH}")

if __name__ == "__main__":
    run()
