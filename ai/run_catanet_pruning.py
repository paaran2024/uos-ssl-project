# -*- coding: utf-8 -*-
import argparse
import yaml
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import redirect_stdout
import io

# --- 커스텀 모듈 및 OPTIN 라이브러리 임포트 ---
from scripts.load_catanet import get_catanet_teacher_model
from prune.main_prune import pruneModel
from evals.gen_eval import evalModel, _apply_pruning_in_memory
from data.scripts.gen_dataset import generateDataset
from utils.utility import calculateComplexity

# 이 스크립트는 CATANet-L 모델에 대한 프루닝 전체 과정을 실행하는 메인 스크립트입니다.
# MODIFIED: 여러 mac_constraint 값에 대한 반복 테스트 및 결과 시각화 기능 추가
# MODIFIED: 매 실행 전 .pkl 랭킹 캐시 파일 자동 삭제 기능 추가

def apply_and_save_pruned_model(model, pruning_params, save_path):
    """
    프루닝 마스크를 적용하고 프루닝된 state_dict를 저장합니다.
    """
    print(f"--- 프루닝된 모델 가중치 저장 시작 ---")
    print(f"저장 경로: {save_path}")
    pruned_state_dict = _apply_pruning_in_memory(model, pruning_params)
    torch.save({'params': pruned_state_dict}, save_path)
    print(f"--- 프루닝된 모델 가중치 저장 완료 ---\n")

def visualize_tradeoff_results(csv_path="tradeoff_results.csv", image_path="tradeoff_visualization.png"):
    """
    CSV 파일로부터 트레이드오프 결과를 읽어 시각화하고 이미지로 저장합니다.
    """
    if not os.path.exists(csv_path):
        print(f"[경고] 시각화할 결과 파일 '{csv_path}'를 찾을 수 없습니다.")
        return

    print(f"--- '{csv_path}' 파일로부터 트레이드오프 그래프 생성 ---")
    df = pd.read_csv(csv_path)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # GFLOPs vs PSNR
    sns.lineplot(data=df, x='GFLOPs', y='PSNR', marker='o', ax=ax, label='PSNR vs. GFLOPs')
    sns.scatterplot(data=df, x='GFLOPs', y='PSNR', s=100, ax=ax, color='red')

    ax.set_xlabel('GFLOPs (Lower is Better)', fontsize=12)
    ax.set_ylabel('PSNR (dB) (Higher is Better)', fontsize=12)
    ax.set_title('PSNR vs. Complexity Trade-off for CATANet Pruning', fontsize=16)

    # 각 점에 mac_constraint 값 표시
    for i, row in df.iterrows():
        ax.text(row['GFLOPs'], row['PSNR'] + 0.01, f"mac={row['MAC Constraint']:.2f}", fontsize=9, ha='center')
    
    # GFLOPs 축을 역순으로 표시하여 오른쪽으로 갈수록 좋은 모델(더 적은 GFLOPs)이 되게 함
    ax.invert_xaxis()
    ax.grid(True)
    plt.savefig(image_path, dpi=300)
    print(f"트레이드오프 그래프가 '{image_path}'에 저장되었습니다.")
    # plt.show()


def run_single(args):
    """
    단일 mac_constraint 값에 대해 프루닝 및 평가를 실행합니다.
    """
    # --- 1. 교사 모델(CATANet-L) 로드 ---
    # 매번 새로운 모델을 로드하여 이전 프루닝의 영향을 받지 않도록 합니다.
    teacher_model = get_catanet_teacher_model(weights_path=args.weights_path, upscale=args.scale)
    
    model_config = {
        "num_attention_heads": teacher_model.heads,
        "intermediate_size": teacher_model.mlp_dim,
        "hidden_size": teacher_model.dim,
        "num_hidden_layers": teacher_model.block_num,
    }
    args.model_config = model_config
    
    print(f"--- 교사 모델 '{args.model_name}' 로드 완료 ---")
    num_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"모델 파라미터 수: {num_params / 1e6:.2f}M\n")

    # --- 2. 데이터셋 로드 ---
    # 데이터셋은 한 번만 로드해도 됩니다. 단, generateDataset이 args를 수정할 수 있으므로 복사해서 사용합니다.
    temp_args = argparse.Namespace(**vars(args))
    train_dataset, val_dataset, temp_args = generateDataset(temp_args)

    # --- 3. 프루닝 실행 ---
    print("--- 모델 프루닝 시작 ---")
    pruningParams, baselineComplexity, prunedComplexity = pruneModel(temp_args, teacher_model, train_dataset, model_config)
    print("--- 모델 프루닝 완료 ---\n")

    # --- 4. 프루닝된 모델 및 마스크 저장 ---
    # 가장 마지막 mac_constraint에 대한 결과만 저장하거나, 필요에 따라 경로를 동적으로 변경할 수 있습니다.
    pruned_model_save_path = os.path.join('weights', f'catanet_pruned_mac_{args.mac_constraint:.2f}.pth')
    pruning_masks_save_path = os.path.join('weights', f'catanet_pruning_masks_mac_{args.mac_constraint:.2f}.pth')
    
    print(f"프루닝 마스크를 '{pruning_masks_save_path}'에 저장합니다...")
    torch.save(pruningParams, pruning_masks_save_path)
    print("마스크 저장 완료.")
    apply_and_save_pruned_model(teacher_model, pruningParams, pruned_model_save_path)

    # --- 5. 결과 평가 ---
    print("--- 프루닝된 모델 성능 평가 시작 ---")
    if isinstance(baselineComplexity, dict): baselineComplexity = baselineComplexity.get('MAC', 1)
    if isinstance(prunedComplexity, dict): prunedComplexity = prunedComplexity.get('MAC', 1)
    
    pruned_gflops = (prunedComplexity * 2) / 1e9

    # evalModel의 print 출력을 가로채서 깔끔하게 만듭니다.
    f = io.StringIO()
    with redirect_stdout(f):
        baselinePerformance, finalPerformance = evalModel(temp_args, teacher_model, train_dataset, val_dataset, pruningParams, model_config)
    eval_output = f.getvalue() # 필요시 로그 파일에 저장 가능

    print("--- 성능 평가 완료 ---")

    # --- 6. 결과 반환 ---
    result = {
        'MAC Constraint': args.mac_constraint,
        'PSNR': finalPerformance.get('PSNR', 0),
        'SSIM': finalPerformance.get('SSIM', 0),
        'GFLOPs': pruned_gflops,
        'Params (M)': sum(p.numel() for p in _apply_pruning_in_memory(teacher_model, pruningParams).values()) / 1e6
    }
    
    print("\n--- 이번 실행 결과 요약 ---")
    print(f"  - MAC 제약: {result['MAC Constraint']:.2f}")
    print(f"  - 최종 PSNR: {result['PSNR']:.4f}")
    print(f"  - 최종 GFLOPs: {result['GFLOPs']:.4f}")
    print("-" * 28 + "\n")

    return result


def main():
    # --- 설정 파일 로드 ---
    parser = argparse.ArgumentParser(description="CATANet-L 모델을 위한 커스텀 프루닝 실행 스크립트")
    parser.add_argument("--config", default="config_catanet.yml", help="프루닝 설정을 담은 YAML 파일 경로")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)
    
    for key, value in yaml_config.items():
        setattr(args, key, value)

    print("--- 설정 파일 로드 완료 ---")
    print(f"모델: {args.model_name}, 데이터셋: {args.dataset}\n")
    
    # --- mac_constraint 값에 따라 분기 ---
    mac_constraints = args.mac_constraint
    if not isinstance(mac_constraints, list):
        mac_constraints = [mac_constraints]

    if len(mac_constraints) > 1:
        print(f"--- 트레이드오프 테스트 모드 시작 ---")
        print(f"테스트할 MAC 제약 조건: {mac_constraints}\n")
    else:
        print(f"--- 단일 실행 모드 시작 ---")
        print(f"MAC 제약 조건: {mac_constraints[0]}\n")

    all_results = []
    
    # --- .pkl 파일 경로 설정 ---
    storage_dir = f"storage/{args.task_name}/{args.dataset}/{args.model_name}"
    os.makedirs(storage_dir, exist_ok=True) # 폴더가 없으면 생성
    head_ranking_file = os.path.join(storage_dir, "head_ranking_body.pkl")
    neuron_ranking_file = os.path.join(storage_dir, "neuron_ranking_body.pkl")

    # --- 메인 루프 ---
    for mac in mac_constraints:
        # --- [추가됨] 이전 랭킹(.pkl) 파일 삭제 ---
        print(f"--- MAC 제약: {mac} 실행 전, 이전 랭킹 파일 삭제 ---")
        try:
            os.remove(head_ranking_file)
            print(f"삭제 완료: {head_ranking_file}")
        except FileNotFoundError:
            print(f"삭제할 파일 없음 (정상): {head_ranking_file}")
        try:
            os.remove(neuron_ranking_file)
            print(f"삭제 완료: {neuron_ranking_file}")
        except FileNotFoundError:
            print(f"삭제할 파일 없음 (정상): {neuron_ranking_file}")
        print("-" * 50)
        # --- 삭제 로직 끝 ---

        # 각 실행을 위한 args 객체 복사 및 수정
        run_args = argparse.Namespace(**vars(args))
        run_args.mac_constraint = mac
        
        result = run_single(run_args)
        all_results.append(result)

    # --- 최종 결과 처리 ---
    if len(all_results) > 1:
        print("--- 모든 테스트 실행 완료 ---")
        results_df = pd.DataFrame(all_results)
        
        # 결과를 GFLOPs 기준으로 정렬
        results_df = results_df.sort_values(by='GFLOPs', ascending=False)
        
        print("최종 결과 요약:")
        print(results_df.to_string(index=False))

        # CSV 저장
        output_csv = 'tradeoff_results.csv'
        results_df.to_csv(output_csv, index=False)
        print(f"\n결과가 '{output_csv}'에 저장되었습니다.")

        # 시각화
        visualize_tradeoff_results(csv_path=output_csv)
    else:
        print("--- 단일 실행 완료 ---")

if __name__ == "__main__":
    main()
