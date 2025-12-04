import tensorflow as tf
import os

def run():
    # ================= [설정 영역] =================
    # Step 2에서 생성된 TensorFlow 폴더 경로
    INPUT_TF_PATH = os.path.join(os.path.dirname(__file__), 'catanet_child_x4_tf')
    
    # 저장할 최종 TFLite 파일명
    OUTPUT_TFLITE_PATH = os.path.join(os.path.dirname(__file__), 'catanet_child_x4.tflite')
    # ==============================================

    print(f"--- [Step 3] TensorFlow -> TFLite 변환 시작 ---")

    if not os.path.exists(INPUT_TF_PATH):
        print(f"오류: TensorFlow 모델 폴더가 없습니다: {INPUT_TF_PATH}")
        print("step2_onnx2tf.py를 먼저 실행해주세요.")
        return

    # 1. Converter 설정 (SavedModel 불러오기)
    converter = tf.lite.TFLiteConverter.from_saved_model(INPUT_TF_PATH)

    # 2. 호환성 옵션 설정
    # 일부 고급 연산은 기본 TFLite에서 지원하지 않을 수 있어, TensorFlow 연산을 허용합니다.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # 기본 TFLite 연산
        tf.lite.OpsSet.SELECT_TF_OPS    # Flex Delegate (TensorFlow 연산 허용)
    ]
    
    # (선택) 모델 최적화: 용량을 줄이려면 아래 주석을 해제하세요.
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 3. 변환 수행
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"변환 중 오류 발생: {e}")
        return

    # 4. 파일 저장
    with open(OUTPUT_TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    file_size_mb = os.path.getsize(OUTPUT_TFLITE_PATH) / (1024 * 1024)
    print(f"변환 완료: {OUTPUT_TFLITE_PATH}")
    print(f"최종 모델 크기: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    run()
