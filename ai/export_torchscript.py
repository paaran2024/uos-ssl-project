import torch
import torch.utils.mobile_optimizer
import os
import sys

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 원본 모델 아키텍처 로드
import models.catanet_arch as arch_module

def run():
    # ================= [설정 영역] =================
    # 가중치 파일 경로 (x2 모델)
    WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'weights', 'catanet_finetuned_feature_kd.pth')
    
    # 출력 파일 경로 (.ptl 확장자 사용 권장 - PyTorch Lite Interpreter용)
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'catanet_child_x2.ptl')
    
    UPSCALE_FACTOR = 2
    # 트레이싱을 위한 예시 입력 크기
    # SR 모델은 보통 Fully Convolutional이라 입력 크기가 달라도 되지만, 
    # TorchScript 변환 시 특정 크기로 Trace를 뜨면 내부 연산 그래프가 그 크기에 고정될 수 있습니다.
    # 모바일에서 주로 사용할 패치 크기나 해상도를 지정하는 것이 좋습니다.
    INPUT_SIZE = 128 
    # ==============================================

    print(f"--- [New Direction] PyTorch -> TorchScript (Mobile) 변환 시작 ---")

    device = torch.device('cpu')
    
    # 1. 모델 로드
    print("1. 모델 및 가중치 로드 중...")
    model = arch_module.CATANet(upscale=UPSCALE_FACTOR)
    model.to(device)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"오류: 가중치 파일이 없습니다: {WEIGHTS_PATH}")
        return

    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = checkpoint['params'] if 'params' in checkpoint else checkpoint
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print("   - 가중치 로드 완료")
    except Exception as e:
        print(f"   - [경고] 가중치 로드 중 이슈 발생: {e}")

    model.eval()

    # 2. TorchScript 변환 (Tracing)
    print(f"2. Tracing 수행 (Input: {INPUT_SIZE}x{INPUT_SIZE})...")
    # 임의의 입력 텐서 생성
    example_input = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)

    try:
        # Tracing을 통해 연산 그래프 기록
        traced_script_module = torch.jit.trace(model, example_input)
        
        # 3. 모바일 최적화 (Mobile Optimizer)
        print("3. 모바일용 최적화 (optimize_for_mobile)...")
        optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_script_module)
        
        # 4. 파일 저장 (Lite Interpreter 포맷)
        print(f"4. 저장 중: {OUTPUT_PATH}")
        # _save_for_lite_interpreter는 최신 PyTorch Mobile 런타임에서 필수입니다.
        optimized_model._save_for_lite_interpreter(OUTPUT_PATH)
        
        file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
        print(f"=== 변환 성공! ===")
        print(f"파일 위치: {OUTPUT_PATH}")
        print(f"파일 크기: {file_size_mb:.2f} MB")
        print("이제 이 .ptl 파일을 Flutter 프로젝트의 assets에 넣으시면 됩니다.")
        
    except Exception as e:
        print(f"변환 실패: {e}")
        print("팁: 모델 내부의 동적 제어 흐름이나 einops 등 서드파티 연산이 Tracing과 호환되지 않을 수 있습니다.")

if __name__ == "__main__":
    run()
