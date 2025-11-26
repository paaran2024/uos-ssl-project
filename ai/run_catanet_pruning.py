# -*- coding: utf-8 -*-
import argparse
import yaml
import torch
import os

# --- 커스텀 모듈 및 OPTIN 라이브러리 임포트 ---

# 1. 우리가 만든 커스텀 CATANet-L 모델 로더를 임포트합니다.
from scripts.load_catanet import get_catanet_teacher_model

# 2. OPTIN 프레임워크의 핵심 프루닝 및 평가 함수를 임포트합니다.
from prune.main_prune import pruneModel
from evals.gen_eval import evalModel
from data.scripts.gen_dataset import generateDataset
from utils.utility import calculateComplexity

# 이 스크립트는 CATANet-L 모델에 대한 프루닝 전체 과정을 실행하는 메인 스크립트입니다.
# OPTIN의 main.py 역할을 하지만, 우리가 만든 커스텀 모델을 사용하도록 수정되었습니다.

def apply_and_save_pruned_model(model, pruning_params, save_path):
    """
    프루닝 마스크를 모델에 적용하고, 프루닝된 모델의 state_dict를 저장합니다.
    현재는 MLP의 뉴런 프루닝만 적용합니다.

    Args:
        model (nn.Module): 원본 교사 모델 (실제 네트워크, 예: CATANet).
        pruning_params (dict): 'neuron_mask'를 포함하는 프루닝 파라미터 딕셔너리.
        save_path (str): 프루닝된 모델의 state_dict를 저장할 경로.
    """
    print(f"--- 프루닝된 모델 가중치 저장 시작 ---")
    print(f"저장 경로: {save_path}")

    # 원본 모델의 state_dict를 복사하여 수정합니다.
    new_state_dict = model.state_dict()
    
    # 프루닝 파라미터에서 뉴런 마스크를 가져옵니다.
    # 마스크 값이 1이면 유지, 0이면 프루닝 대상입니다.
    neuron_mask = pruning_params.get('neuron_mask')

    if neuron_mask is None:
        print("경고: 뉴런 마스크를 찾을 수 없습니다. 모델을 저장하지 않습니다.")
        return

    # 모델의 각 블록을 순회하며 프루닝을 적용합니다.
    for i in range(model.block_num):
        # 뉴런 마스크에서 현재 레이어에서 프루닝할 뉴런의 인덱스를 찾습니다.
        # 마스크 값이 0인 뉴런이 프루닝 대상입니다.
        pruned_neurons = (neuron_mask[i] == 0).nonzero(as_tuple=True)[0]

        if len(pruned_neurons) == 0:
            continue # 이 레이어에서는 프루닝할 뉴런이 없음

        # 프루닝할 뉴런의 인덱스를 사용하여 가중치를 0으로 만듭니다.
        # ConvFFN의 fc1, fc2 레이어가 대상입니다.
        fc1_weight_name = f'blocks.{i}.0.mlp.fn.fc1.weight'
        fc1_bias_name = f'blocks.{i}.0.mlp.fn.fc1.bias'
        
        if fc1_weight_name in new_state_dict:
            new_state_dict[fc1_weight_name][pruned_neurons, :] = 0
            new_state_dict[fc1_bias_name][pruned_neurons] = 0

        # fc2: Linear(mlp_dim, dim) -> 입력 뉴런에 해당하는 열(column)을 0으로 만듭니다.
        fc2_weight_name = f'blocks.{i}.0.mlp.fn.fc2.weight'
        if fc2_weight_name in new_state_dict:
            new_state_dict[fc2_weight_name][:, pruned_neurons] = 0

    # 수정된 state_dict를 파일로 저장합니다.
    torch.save(new_state_dict, save_path)
    print(f"--- 프루닝된 모델 가중치 저장 완료 ---")


def main():
    # --- 1. 설정 파일 로드 ---
    parser = argparse.ArgumentParser(description="CATANet-L 모델을 위한 커스텀 프루닝 실행 스크립트")
    parser.add_argument("--config", required=True, help="프루닝 설정을 담은 YAML 파일 경로")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
    
    # YAML 파일 내용을 args 객체에 병합합니다.
    for key, value in yaml_config.items():
        setattr(args, key, value)

    print("--- 설정 파일 로드 완료 ---")
    print(f"모델: {args.model_name}, 데이터셋: {args.dataset}")
    print(f"프루닝 제약 조건 (MAC): {args.mac_constraint}")
    print("--------------------------\n")

    # --- 2. 교사 모델(CATANet-L) 로드 ---
    # `get_catanet_teacher_model`은 `CATANet` nn.Module 자체를 반환합니다.
    teacher_model = get_catanet_teacher_model(weights_path=args.weights_path, upscale=args.scale)
    
    # 모델의 설정(config)을 가져옵니다.
    # MODIFIED: `teacher_model`이 실제 네트워크이므로 `.net_g` 없이 직접 접근합니다.
    model_config = {
        "num_attention_heads": teacher_model.heads,
        "intermediate_size": teacher_model.mlp_dim,
        "hidden_size": teacher_model.dim,
        "num_hidden_layers": teacher_model.block_num,
    }
    args.model_config = model_config
    
    print(f"--- 교사 모델 '{args.model_name}' 로드 완료 ---")
    num_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"모델 파라미터 수: {num_params / 1e6:.2f}M")
    print("-----------------------------------\n")

    # --- 3. 데이터셋 로드 ---
    train_dataset, val_dataset, args = generateDataset(args)
    print("--- 데이터셋 로드 완료 ---\n")

    # --- 4. 프루닝 실행 ---
    prunedProps = {
        "num_att_head": model_config["num_attention_heads"],
        "inter_size": model_config["intermediate_size"],
        "hidden_size": model_config["hidden_size"],
        "num_layers": model_config["num_hidden_layers"],
    }
    
    print("--- 모델 프루닝 시작 ---")
    # `pruneModel`은 `CATANet` 모듈을 직접 받습니다.
    pruningParams, baselineComplexity, prunedComplexity = pruneModel(args, teacher_model, train_dataset, model_config)
    print("--- 모델 프루닝 완료---\n")

    # --- 5. 프루닝된 모델 저장 ---
    pruned_model_save_path = os.path.join('weights', 'catanet_pruned.pth')
    # MODIFIED: `teacher_model`이 실제 네트워크이므로 `.net_g` 없이 직접 전달합니다.
    apply_and_save_pruned_model(teacher_model, pruningParams, pruned_model_save_path)
    print("\n")

    # --- 6. 결과 평가 ---
    print("--- 프루닝된 모델 성능 평가 시작 ---")
    # `prunedComplexity` 와 `baselineComplexity`는 이제 숫자 값일 것으로 예상됩니다.
    if isinstance(baselineComplexity, dict): baselineComplexity = baselineComplexity.get('MAC', 1)
    if isinstance(prunedComplexity, dict): prunedComplexity = prunedComplexity.get('MAC', 1)
    flop_reduction_amount = 100 - (prunedComplexity / baselineComplexity * 100.0)
    print(f"FLOPs 감소율: {flop_reduction_amount:.2f}%")
    
    # `evalModel`은 `CATANet` 모듈을 직접 받습니다.
    baselinePerformance, finalPerformance = evalModel(args, teacher_model, train_dataset, val_dataset, pruningParams, prunedProps)
    print("--- 성능 평가 완료---\n")

    # --- 7. 최종 결과 출력 ---
    print("--- 최종 결과 ---")
    print(f"원본 모델 성능: {baselinePerformance}")
    print(f"프루닝된 모델 성능: {finalPerformance}")
    print(f"FLOPs 감소율: {flop_reduction_amount:.2f}%")
    print(f"FLOPs 비율: {prunedComplexity / baselineComplexity * 100.0:.2f}%")
    print("------------------")

if __name__ == "__main__":
    main()