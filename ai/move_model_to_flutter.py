import os
import shutil

def run():
    # 1. 생성된 .ptl 모델 경로 (ai 폴더 내)
    SOURCE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'catanet_child_x2.ptl')
    
    # 2. 이동할 목적지 경로 (Flutter 앱 assets 폴더)
    # ../app/assets/models 폴더를 타겟으로 설정
    FLUTTER_APP_ROOT = os.path.join(os.path.dirname(__file__), '..', 'app')
    DEST_DIR = os.path.join(FLUTTER_APP_ROOT, 'assets', 'models')
    DEST_MODEL_PATH = os.path.join(DEST_DIR, 'catanet_child_x2.ptl')

    print("--- [Model Transfer] Flutter 앱으로 모델 이동 ---")

    # 소스 파일 확인
    if not os.path.exists(SOURCE_MODEL_PATH):
        print(f"[오류] 원본 모델 파일이 없습니다: {SOURCE_MODEL_PATH}")
        print("먼저 'export_torchscript.py'를 실행하여 모델을 생성해주세요.")
        return

    # 목적지 폴더 생성 (없으면 생성)
    if not os.path.exists(DEST_DIR):
        print(f"assets/models 폴더 생성 중: {DEST_DIR}")
        os.makedirs(DEST_DIR, exist_ok=True)

    # 파일 복사
    try:
        shutil.copy2(SOURCE_MODEL_PATH, DEST_MODEL_PATH)
        print(f"[성공] 모델 복사 완료!")
        print(f" - 원본: {SOURCE_MODEL_PATH}")
        print(f" - 대상: {DEST_MODEL_PATH}")
        print("\n[다음 단계]")
        print("1. 'app/pubspec.yaml' 파일을 열고 assets 섹션에 다음 줄을 추가하세요:")
        print("   assets:")
        print("     - assets/models/catanet_child_x2.ptl")
    except Exception as e:
        print(f"[오류] 파일 복사 실패: {e}")

if __name__ == "__main__":
    run()
