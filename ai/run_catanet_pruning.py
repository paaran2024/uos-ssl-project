# -*- coding: utf-8 -*-
import argparse
import yaml
import torch
import os

# --- 커스텀 모듈 및 OPTIN 라이브러리 임포트 ---

# 1. 우리가 만든 커스텀 CATANet-L 모델 로더를 임포트합니다.
from ai.scripts.load_catanet import get_catanet_teacher_model

# 2. OPTIN 프레임워크의 핵심 프루닝 및 평가 함수를 임포트합니다.
# references 폴더에 있지만, 수정 없이 라이브러리처럼 사용합니다.
from ai.references.OPTIN.prune.main_prune import pruneModel
from ai.references.OPTIN.evals.gen_eval import evalModel
from ai.references.OPTIN.data.scripts.gen_dataset import generateDataset
from ai.references.OPTIN.utils.utility import calculateComplexity

# 이 스크립트는 CATANet-L 모델에 대한 프루닝 전체 과정을 실행하는 메인 스크립트입니다.
# OPTIN의 main.py 역할을 하지만, 우리가 만든 커스텀 모델을 사용하도록 수정되었습니다.

def main():
    # --- 1. 설정 파일 로드 ---
    parser = argparse.ArgumentParser(description="CATANet-L 모델을 위한 커스텀 프루닝 실행 스크립트")
    parser.add_argument("--config", required=True, help="프루닝 설정을 담은 YAML 파일 경로")
    args = parser.parse_args()

    with open(args.config, "r") as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
    
    # YAML 파일 내용을 args 객체에 병합합니다.
    for key, value in yaml_config.items():
        setattr(args, key, value)

    print("--- 설정 파일 로드 완료 ---")
    print(f"모델: {args.model_name}, 데이터셋: {args.dataset}")
    print(f"프루닝 제약 조건 (MAC): {args.mac_constraint}")
    print("--------------------------\n")

    # --- 2. 교사 모델(CATANet-L) 로드 ---
    # 우리가 만든 load_catanet.py 스크립트를 사용하여 모델을 로드합니다.
    # 나중에 실제 훈련된 가중치 파일 경로를 config 파일에 명시해야 합니다.
    teacher_model = get_catanet_teacher_model(weights_path=args.weights_path, upscale=args.scale)
    
    # 모델의 설정(config)을 가져옵니다. OPTIN의 pruneModel 함수에 필요합니다.
    # CATANet에는 Hugging Face의 config 객체가 없으므로, 직접 딕셔너리를 만들어줍니다.
    # 이 값들은 CATANet 아키텍처에 맞춰 설정해야 합니다.
    model_config = {
        "num_attention_heads": teacher_model.heads,
        "intermediate_size": teacher_model.mlp_dim, # mlp_dim을 intermediate_size로 간주
        "hidden_size": teacher_model.dim,
        "num_hidden_layers": teacher_model.block_num,
    }
    args.model_config = model_config
    
    print(f"--- 교사 모델 '{args.model_name}' 로드 완료 ---")
    num_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"모델 파라미터 수: {num_params / 1e6:.2f}M")
    print("-----------------------------------\n")

    # --- 3. 데이터셋 로드 ---
    # OPTIN의 데이터셋 생성 스크립트를 재사용합니다.
    # config 파일에 따라 학습 및 검증 데이터셋을 로드합니다.
    train_dataset, val_dataset, args = generateDataset(args)
    print("--- 데이터셋 로드 완료 ---\n")

    # --- 4. 프루닝 실행 ---
    # pruneModel 함수에 필요한 프로퍼티들을 준비합니다.
    # 이 값들은 모델 아키텍처와 데이터에 따라 결정됩니다.
    prunedProps = {
        "num_att_head": model_config["num_attention_heads"],
        "inter_size": model_config["intermediate_size"],
        "hidden_size": model_config["hidden_size"],
        "num_layers": model_config["num_hidden_layers"],
        "patch_size": 128 + 1 # 예시 값, 실제 데이터셋의 시퀀스 길이에 맞춰야 함
    }
    
    print("--- 모델 프루닝 시작 ---")
    # OPTIN의 핵심 프루닝 함수를 호출합니다.
    # 이 함수는 교사 모델을 분석하여 어떤 뉴런과 헤드를 제거할지 결정하고,
    # 프루닝 마스크(pruningParams)를 반환합니다.
    pruningParams, baselineComplexity, prunedComplexity = pruneModel(args, teacher_model, train_dataset, model_config)
    print("--- 모델 프루닝 완료 ---\n")

    # --- 5. 결과 평가 ---
    print("--- 프루닝된 모델 성능 평가 시작 ---")
    flop_reduction_amount = 100 - (prunedComplexity / baselineComplexity * 100.0)
    print(f"FLOPs 감소율: {flop_reduction_amount:.2f}%")
    
    # OPTIN의 평가 함수를 호출하여 원본 모델과 프루닝된 모델의 성능을 비교합니다.
    # `evalModel`은 내부적으로 원본 모델과 프루닝 마스크를 적용한 모델을 각각 평가합니다.
    baselinePerformance, finalPerformance = evalModel(args, teacher_model, train_dataset, val_dataset, pruningParams, prunedProps)
    print("--- 성능 평가 완료 ---\n")

    # --- 6. 최종 결과 출력 ---
    print("--- 최종 결과 ---")
    print(f"원본 모델 성능: {baselinePerformance}")
    print(f"프루닝된 모델 성능: {finalPerformance}")
    print(f"FLOPs 감소율: {flop_reduction_amount:.2f}%")
    print(f"FLOPs 비율: {prunedComplexity / baselineComplexity * 100.0:.2f}%")
    print("------------------")

if __name__ == "__main__":
    main()
