import onnx
from onnx_tf.backend import prepare
import os

def run():
    # ================= [설정 영역] =================
    # Step 1에서 생성된 ONNX 파일 경로
    INPUT_ONNX_PATH = os.path.join(os.path.dirname(__file__), 'catanet_child_x4.onnx')
    
    # 저장할 TensorFlow SavedModel 폴더명
    OUTPUT_TF_PATH = os.path.join(os.path.dirname(__file__), 'catanet_child_x4_tf')
    # ==============================================

    print(f"--- [Step 2] ONNX -> TensorFlow 변환 시작 ---")

    if not os.path.exists(INPUT_ONNX_PATH):
        print(f"오류: ONNX 파일이 없습니다: {INPUT_ONNX_PATH}")
        print("step1_pth2onnx.py를 먼저 실행해주세요.")
        return

    # 1. ONNX 모델 로드
    onnx_model = onnx.load(INPUT_ONNX_PATH)
    onnx.checker.check_model(onnx_model) # 모델 무결성 확인
    print("ONNX 모델 로드 및 검사 완료")

    # 2. TensorFlow 모델로 변환 및 저장
    # SavedModel 형식으로 디렉토리에 저장됩니다.
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(OUTPUT_TF_PATH)
    
    print(f"변환 완료: {OUTPUT_TF_PATH}")

if __name__ == "__main__":
    run()
