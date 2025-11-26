import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import os

# --- 커스텀 모듈 임포트 ---
# 독립적인 실행을 위해 프로젝트 루트를 기준으로 필요한 모듈을 임포트합니다.
from scripts.load_catanet import get_catanet_teacher_model
from data.scripts.gen_dataset import generateDataset
from basicsr.data import build_dataset
from basicsr.metrics import calculate_psnr
from basicsr.utils import tensor2img
# [개선] Feature Distillation을 위해 후킹 클래스 임포트
from utils.catanet_hooks import CATANetModelHooking

"""
finetune_pruned_model.py: 지식 증류(Knowledge Distillation)를 사용하여
                           가지치기된(pruned) 모델을 파인튜닝하는 스크립트입니다.

이 스크립트는 두 단계로 구성됩니다:
1.  (현재 구현) Output Distillation: 교사 모델의 최종 출력(이미지)을 학생 모델이 모방하도록 학습합니다.
2.  (추후 확장) Feature Distillation: 교사 모델의 중간 피처(feature)를 학생 모델이 모방하도록 학습합니다.

실행 방법 (ai/ 디렉토리에서):
    python finetune_pruned_model.py --config config_catanet.yml \
                                     --teacher_weights weights/CATANet-L_x2.pth \
                                     --pruned_weights weights/catanet_pruned.pth \
                                     --save_path weights/catanet_finetuned.pth \
                                     --epochs 10 \
                                     --lr 1e-4 \
                                     --alpha 0.8
"""

def main():
    # --- 0. 스크립트 실행 위치 보정 ---
    # 스크립트가 어디에서 실행되든, 이 파일이 위치한 폴더를 기준으로 동작하도록
    # 현재 작업 디렉토리를 변경합니다. Colab/로컬 환경 간의 경로 차이를 해결합니다.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory changed to: {os.getcwd()}")

    # --- 1. 인자 파싱 및 설정 ---
    parser = argparse.ArgumentParser(description="Pruned CATANet Fine-tuning with Knowledge Distillation")
    parser.add_argument("--config", required=True, help="모델 및 데이터셋 설정을 담은 YAML 파일")
    parser.add_argument("--teacher_weights", required=True, help="원본 교사 모델 가중치 경로")
    parser.add_argument("--pruned_weights", required=True, help="가지치기된 학생 모델 가중치 경로")
    parser.add_argument("--save_path", required=True, help="파인튜닝된 모델을 저장할 경로")
    parser.add_argument("--epochs", type=int, default=10, help="파인튜닝 에폭 수")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률 (Learning Rate)")
    parser.add_argument("--alpha", type=float, default=0.8, help="Output Distillation Loss의 가중치")
    # [개선] Feature Distillation을 위한 인자 추가
    parser.add_argument("--distillation_type", type=str, default="output", choices=["output", "feature"], help="증류 타입 (feature 선택 시 output과 feature 모두 사용)")
    parser.add_argument("--beta", type=float, default=0.5, help="Feature Distillation Loss의 가중치")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # YAML 파일 내용을 args 객체에 병합
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. 데이터셋 로드 ---
    # `generateDataset`은 학습용 데이터 로더만 생성하므로, 검증용은 별도로 생성합니다.
    train_opt = config['datasets']['train']
    val_opt = config['datasets']['val']

    # [수정] 데이터셋 설정에 upscale 비율(scale)을 명시적으로 추가합니다.
    # PairedImageDataset에서 'scale' 키를 요구하기 때문입니다.
    train_opt['scale'] = args.scale
    val_opt['scale'] = args.scale

    # [수정] 데이터셋 설정에 phase를 명시적으로 추가합니다.
    # PairedImageDataset에서 'phase' 키를 요구하기 때문입니다.
    train_opt['phase'] = 'train'
    val_opt['phase'] = 'val'
    
    train_set = build_dataset(train_opt)
    val_set = build_dataset(val_opt)

    train_loader = DataLoader(
        train_set,
        batch_size=train_opt['batch_size_per_gpu'],
        shuffle=True,
        num_workers=train_opt['num_worker_per_gpu'],
        sampler=None,
        pin_memory=True
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)
    print(f"데이터셋 로드 완료. Train: {len(train_set)}개, Val: {len(val_set)}개")

    # --- 3. 모델 로드 ---
    # 교사 모델 로드 (학습되지 않음)
    print("교사 모델 로딩...")
    teacher_model = get_catanet_teacher_model(weights_path=args.teacher_weights, upscale=args.scale).to(device)
    teacher_model.eval()

    # 학생 모델 로드 (학습 대상)
    print("학생 모델 로딩 (가지치기된 가중치)...")
    student_model = get_catanet_teacher_model(weights_path=None, upscale=args.scale).to(device)
    pruned_state = torch.load(args.pruned_weights, map_location=device)
    if 'params' in pruned_state:
        pruned_state = pruned_state['params']
    student_model.load_state_dict(pruned_state, strict=False) # strict=False로 하여 완벽히 일치하지 않아도 로드
    student_model.train()
    
    # [개선] Feature Distillation을 위한 모델 후킹
    teacher_hook = CATANetModelHooking(args=None, model=teacher_model)
    student_hook = CATANetModelHooking(args=None, model=student_model)

    # [수정] 피처 증류 활성화 시 forward hook을 즉시 등록하여 경고를 방지합니다.
    if args.distillation_type == 'feature':
        teacher_hook.apply_mask_and_hooks()
        student_hook.apply_mask_and_hooks()
    
    print("모델 로드 및 후킹 완료.")

    # --- 4. 옵티마이저 및 손실 함수 설정 ---
    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr)
    l1_loss = torch.nn.L1Loss().to(device)
    
    best_psnr = 0.0
    
    # --- 5. 파인튜닝 학습 루프 ---
    for epoch in range(1, args.epochs + 1):
        student_model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")

        for batch in pbar:
            lq = batch['lq'].to(device)
            gt = batch['gt'].to(device)

            # --- 손실 계산 ---
            optimizer.zero_grad()
            
            # 교사/학생 모델로부터 출력과 피처맵을 동시에 추출합니다.
            with torch.no_grad():
                teacher_output, teacher_fms = teacher_hook.forwardPass(lq)
            
            student_output, student_fms = student_hook.forwardPass(lq)

            # 1. Output Distillation (기본)
            loss_task = l1_loss(student_output, gt)
            loss_distill_output = l1_loss(student_output, teacher_output)
            loss = (1 - args.alpha) * loss_task + args.alpha * loss_distill_output
            
            # 2. Feature Distillation (선택 사항)
            if args.distillation_type == 'feature':
                loss_distill_feature = 0
                if student_fms and teacher_fms:
                    for student_fm, teacher_fm in zip(student_fms, teacher_fms):
                        loss_distill_feature += l1_loss(student_fm, teacher_fm)
                    loss += args.beta * loss_distill_feature
                else:
                    print("Warning: Could not retrieve feature maps for feature distillation.")


            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch} 완료. 평균 Loss: {total_loss / len(train_loader):.4f}")

        # --- 6. 검증 (Validation) ---
        student_model.eval()
        current_psnr = 0
        val_pbar = tqdm(val_loader, desc="Validating", unit="image")
        for batch in val_pbar:
            lq, gt = batch['lq'].to(device), batch['gt'].to(device)
            with torch.no_grad():
                student_output = student_model(lq)
            output_img, gt_img = tensor2img(student_output), tensor2img(gt)
            current_psnr += calculate_psnr(output_img, gt_img, crop_border=args.scale, test_y_channel=True)
        
        avg_psnr = current_psnr / len(val_loader)
        print(f"검증 완료. 평균 PSNR: {avg_psnr:.4f}")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(student_model.state_dict(), args.save_path)
            print(f"최고 성능 달성! 모델을 '{args.save_path}'에 저장했습니다. (PSNR: {best_psnr:.4f})")
            
        # 각 에폭 후 훅 제거 및 재설정 (메모리 누수 방지 및 상태 초기화)
        teacher_hook.purge_hooks()
        student_hook.purge_hooks()
        if epoch < args.epochs: # 마지막 에폭 후에는 필요 없음
             teacher_hook.apply_mask_and_hooks()
             student_hook.apply_mask_and_hooks()

    print("\n--- 파인튜닝 완료 ---")
    print(f"최고 PSNR: {best_psnr:.4f}")
    print(f"최종 모델은 '{args.save_path}'에 저장되었습니다.")


if __name__ == '__main__':
    main()
