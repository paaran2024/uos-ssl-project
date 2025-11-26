@echo off
REM =================================================================================
REM SSL-FINAL-AI-ACTIVATOR
REM =================================================================================
REM 이 스크립트는 CATANet 모델의 프루닝, 파인튜닝, 재구성, 분석까지의
REM 전체 워크플로우를 자동으로 실행합니다.
REM
REM 실행 방법:
REM 1. Anaconda Prompt 또는 명령 프롬프트(CMD)를 엽니다.
REM 2. `D:\Project\uos-ssl-project` 경로로 이동합니다. (cd D:\Project\uos-ssl-project)
REM 3. `ssl-final-ai-activator.bat`를 입력하고 Enter 키를 누릅니다.
REM =================================================================================

echo [INFO] 전체 AI 모델 최적화 및 분석 워크플로우를 시작합니다.
echo [INFO] 작업 디렉토리를 'ai' 폴더로 변경합니다.
cd ai

REM --- 1. 이전 결과 파일 정리 ---
echo.
echo [STEP 1/6] 이전 실행 결과 파일들을 정리합니다...
del /q weights\catanet_pruned.pth
del /q weights\catanet_pruning_masks.pth
del /q weights\finetuned_*.pth
del /q weights\rebuilt_*.pth
del /q storage\vision\DIV2K\CATANet-L\*.pkl
del /q tradeoff_results.csv
del /q tradeoff_visualization.png
echo [SUCCESS] 파일 정리 완료.

REM --- 2. 모델 프루닝 ---
echo.
echo [STEP 2/6] 모델 프루닝을 시작합니다. (run_catanet_pruning.py)
python run_catanet_pruning.py --config config_catanet.yml
if %errorlevel% neq 0 (
    echo [ERROR] 모델 프루닝 중 오류가 발생했습니다. 스크립트를 중단합니다.
    exit /b %errorlevel%
)
echo [SUCCESS] 모델 프루닝 완료. 'catanet_pruned.pth' 및 마스크 파일 생성됨.

REM --- 3. 지식 증류 방식별 파인튜닝 ---
echo.
echo [STEP 3/6] 지식 증류 방식별 모델 파인튜닝을 시작합니다. (finetune_pruned_model.py)

echo.
echo [INFO] 3-1. Output Distillation 방식으로 파인튜닝...
python finetune_pruned_model.py --config config_catanet.yml --teacher_weights weights/CATANet-L_x2.pth --pruned_weights weights/catanet_pruned.pth --save_path weights/finetuned_output.pth --distillation_type output --epochs 10 --lr 1e-4 --alpha 0.8
if %errorlevel% neq 0 ( echo [ERROR] Output KD 중 오류 발생! & exit /b %errorlevel% )

echo.
echo [INFO] 3-2. Feature Distillation 방식으로 파인튜닝...
python finetune_pruned_model.py --config config_catanet.yml --teacher_weights weights/CATANet-L_x2.pth --pruned_weights weights/catanet_pruned.pth --save_path weights/finetuned_feature.pth --distillation_type feature --epochs 10 --lr 1e-4 --alpha 0.5 --beta 1
if %errorlevel% neq 0 ( echo [ERROR] Feature KD 중 오류 발생! & exit /b %errorlevel% )

echo.
echo [INFO] 3-3. FaKD 방식으로 파인튜닝...
python finetune_pruned_model.py --config config_catanet.yml --teacher_weights weights/CATANet-L_x2.pth --pruned_weights weights/catanet_pruned.pth --save_path weights/finetuned_fakd.pth --distillation_type fakd --epochs 10 --lr 1e-4 --alpha 0.5 --beta 100
if %errorlevel% neq 0 ( echo [ERROR] FaKD 중 오류 발생! & exit /b %errorlevel% )
echo [SUCCESS] 모든 파인튜닝 완료.

REM --- 4. 압축 모델 재구성 ---
echo.
echo [STEP 4/6] 파인튜닝된 모델들을 물리적으로 작은 모델로 재구성합니다. (analysis/rebuild_pruned_model.py)

echo.
echo [INFO] 4-1. Output KD 모델 재구성...
python analysis/rebuild_pruned_model.py --config config_catanet.yml --masks weights/catanet_pruning_masks.pth --source_weights weights/finetuned_output.pth --save_path weights/rebuilt_output.pth
if %errorlevel% neq 0 ( echo [ERROR] Output 모델 재구성 중 오류 발생! & exit /b %errorlevel% )

echo.
echo [INFO] 4-2. Feature KD 모델 재구성...
python analysis/rebuild_pruned_model.py --config config_catanet.yml --masks weights/catanet_pruning_masks.pth --source_weights weights/finetuned_feature.pth --save_path weights/rebuilt_feature.pth
if %errorlevel% neq 0 ( echo [ERROR] Feature 모델 재구성 중 오류 발생! & exit /b %errorlevel% )

echo.
echo [INFO] 4-3. FaKD 모델 재구성...
python analysis/rebuild_pruned_model.py --config config_catanet.yml --masks weights/catanet_pruning_masks.pth --source_weights weights/finetuned_fakd.pth --save_path weights/rebuilt_fakd.pth
if %errorlevel% neq 0 ( echo [ERROR] FaKD 모델 재구성 중 오류 발생! & exit /b %errorlevel% )
echo [SUCCESS] 모든 모델 재구성 완료.

REM --- 5. 가중치 희소성 검사 ---
echo.
echo [STEP 5/6] 생성된 모든 모델의 가중치 희소성을 검사합니다. (analysis/inspect_weights.py)
python analysis/inspect_weights.py --files weights/CATANet-L_x2.pth weights/catanet_pruned.pth weights/finetuned_fakd.pth weights/rebuilt_fakd.pth
if %errorlevel% neq 0 ( echo [ERROR] 가중치 검사 중 오류 발생! & exit /b %errorlevel% )
echo [SUCCESS] 가중치 검사 완료.

REM --- 6. 최종 성능 비교 및 시각화 ---
echo.
echo [STEP 6/6] 모든 모델의 성능을 최종 비교하고 트레이드오프 그래프를 생성합니다. (analysis/visualize_tradeoff.py)
python analysis/visualize_tradeoff.py
if %errorlevel% neq 0 ( echo [ERROR] 최종 분석 중 오류 발생! & exit /b %errorlevel% )
echo [SUCCESS] 최종 분석 및 그래프 생성 완료.

echo.

echo =================================================================================
echo [WORKFLOW COMPLETE] 모든 작업이 성공적으로 완료되었습니다.
echo 'ai' 폴더에서 'tradeoff_results.csv'와 'tradeoff_visualization.png' 파일을 확인하세요.
echo =================================================================================

cd ..
pause
