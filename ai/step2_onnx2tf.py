import onnx
import sys
from onnx import TensorProto

# ==============================================================================
# [Monkey Patching] onnx-tf와 최신 onnx 버전 간의 호환성 문제 해결
# 'onnx.mapping' 모듈이 삭제되었으므로, 이를 가짜로 만들어줍니다.
# ==============================================================================

if not hasattr(onnx, 'mapping'):
    class MappingProxy:
        # onnx-tf가 주로 사용하는 TENSOR_TYPE_TO_NP_TYPE 매핑을 복구
        # (참고: 실제 매핑은 훨씬 복잡하지만, 주요 타입만 매핑해도 작동할 수 있음)
        TENSOR_TYPE_TO_NP_TYPE = {
            TensorProto.FLOAT: 'float32',
            TensorProto.UINT8: 'uint8',
            TensorProto.INT8: 'int8',
            TensorProto.UINT16: 'uint16',
            TensorProto.INT16: 'int16',
            TensorProto.INT32: 'int32',
            TensorProto.INT64: 'int64',
            TensorProto.STRING: 'object',
            TensorProto.BOOL: 'bool',
            TensorProto.FLOAT16: 'float16',
            TensorProto.DOUBLE: 'float64',
            TensorProto.UINT32: 'uint32',
            TensorProto.UINT64: 'uint64',
        }
        
        # 반대 매핑 (NP_TYPE_TO_TENSOR_TYPE)
        NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in TENSOR_TYPE_TO_NP_TYPE.items()}
        
    # 가짜 모듈 등록
    sys.modules['onnx.mapping'] = MappingProxy
    onnx.mapping = MappingProxy
    print("Patching: 'onnx.mapping' module injected for compatibility.")

# 패치 후에 onnx-tf 임포트
from onnx_tf.backend import prepare
import os

def run():
    # ================= [설정 영역] =================
    INPUT_ONNX_PATH = os.path.join(os.path.dirname(__file__), 'catanet_child_x2.onnx')
    OUTPUT_TF_PATH = os.path.join(os.path.dirname(__file__), 'catanet_child_x2_tf')
    # ==============================================

    print(f"--- [Step 2] ONNX -> TensorFlow 변환 시작 (x2) ---")

    if not os.path.exists(INPUT_ONNX_PATH):
        print(f"오류: ONNX 파일이 없습니다: {INPUT_ONNX_PATH}")
        print("step1_pth2onnx.py를 먼저 실행해주세요.")
        return

    # 1. ONNX 모델 로드
    onnx_model = onnx.load(INPUT_ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX 모델 로드 및 검사 완료")

    # 2. TensorFlow 모델로 변환 및 저장
    try:
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(OUTPUT_TF_PATH)
        print(f"변환 완료: {OUTPUT_TF_PATH}")
    except Exception as e:
        print(f"변환 중 오류 발생: {e}")
        # 추가적인 힌트 제공
        if "mapping" in str(e):
             print("여전히 매핑 오류가 발생한다면 onnx-tf 라이브러리 자체 수정이 필요할 수 있습니다.")

if __name__ == "__main__":
    run()
