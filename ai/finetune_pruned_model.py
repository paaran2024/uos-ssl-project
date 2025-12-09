import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F # FaKD 구현을 위해 추가
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

# --- 커스텀 모듈 임포트 ---
from scripts.load_catanet import get_catanet_teacher_model
from basicsr.data import build_dataset
from basicsr.metrics import calculate_psnr
from basicsr.utils import tensor2img
from utils.catanet_hooks import CATANetModelHooking

# --- 결과 저장 경로 ---
RESULTS_DIR = "results"
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")

"""
finetune_pruned_model.py: 지식 증류(Knowledge Distillation)를 사용하여
                           가지치기된(pruned) 모델을 파인튜닝하는 스크립트입니다.

이 스크립트는 이제 세 가지 증류 방식을 지원합니다:
1.  Output Distillation: 교사 모델의 최종 출력을 학생 모델이 모방합니다. (기본)
2.  Feature Distillation: 교사 모델의 중간 피처맵을 학생 모델이 직접 모방합니다.
3.  FaKD: 교사 모델 피처맵의 구조적 관계(Affinity)를 학생 모델이 모방합니다.

실행 방법 (ai/ 디렉토리에서):
    # FaKD 사용 예시
    python finetune_pruned_model.py --config config_catanet.yml \
                                     --teacher_weights weights/CATANet-L_x2.pth \
                                     --pruned_weights weights/catanet_pruned.pth \
                                     --save_path weights/catanet_finetuned_fakd.pth \
                                     --distillation_type fakd \
                                     --beta 100 
"""

def ensure_results_dirs():
    """결과 저장 디렉토리 생성"""
    for subdir in ['loss', 'psnr', 'comparison']:
        os.makedirs(os.path.join(PLOTS_DIR, subdir), exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)


def get_log_path(distillation_type):
    """증류 타입에 따른 로그 파일 경로 반환"""
    return os.path.join(LOGS_DIR, f"{distillation_type}_kd.csv")


def load_existing_log(log_path):
    """기존 로그 파일 로드 (이어서 학습용)"""
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        start_epoch = int(df['epoch'].max()) + 1
        print(f"📂 기존 로그 발견! {start_epoch-1} 에폭부터 이어서 학습합니다.")
        return df.to_dict('records'), start_epoch
    return [], 1


def save_log(log_data, log_path):
    """로그 데이터를 CSV로 저장"""
    df = pd.DataFrame(log_data)
    df.to_csv(log_path, index=False)


def generate_individual_plots(log_path, distillation_type):
    """개별 모델의 Loss/PSNR 그래프 생성"""
    df = pd.read_csv(log_path)

    # Loss 그래프
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['total_loss'], label='Total Loss', color='blue')
    ax.plot(df['epoch'], df['task_loss'], label='Task Loss', color='green', alpha=0.7)
    ax.plot(df['epoch'], df['distill_loss'], label='Distill Loss', color='red', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{distillation_type.upper()} KD - Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'loss', f'{distillation_type}_kd_loss.png'), dpi=150)
    plt.close()

    # PSNR 그래프
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['val_psnr'], label='Val PSNR', color='purple', linewidth=2)
    ax.axhline(y=df['val_psnr'].max(), color='red', linestyle='--', alpha=0.5, label=f'Best: {df["val_psnr"].max():.2f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'{distillation_type.upper()} KD - Validation PSNR')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'psnr', f'{distillation_type}_kd_psnr.png'), dpi=150)
    plt.close()

    print(f"📊 {distillation_type} 개별 그래프 저장 완료!")


def generate_comparison_plots():
    """3가지 KD 방법 비교 그래프 생성 (1000 에폭 완료 시에만)"""
    kd_types = ['output', 'feature', 'fakd']
    colors = {'output': 'blue', 'feature': 'green', 'fakd': 'red'}

    # 모든 로그 파일 확인
    all_logs = {}
    all_complete = True

    for kd_type in kd_types:
        log_path = get_log_path(kd_type)
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            if len(df) >= 1000:
                all_logs[kd_type] = df
            else:
                print(f"⏳ {kd_type} KD: {len(df)}/1000 에폭 (미완료)")
                all_complete = False
        else:
            print(f"⚠️ {kd_type} KD 로그 파일 없음")
            all_complete = False

    if not all_complete:
        print("❌ 3가지 KD 방법 모두 1000 에폭 완료 후 비교 그래프가 생성됩니다.")
        return False

    print("✅ 모든 KD 방법 1000 에폭 완료! 비교 그래프 생성 중...")

    # PSNR 비교 그래프
    fig, ax = plt.subplots(figsize=(12, 7))
    for kd_type, df in all_logs.items():
        ax.plot(df['epoch'], df['val_psnr'], label=f'{kd_type.upper()} KD',
                color=colors[kd_type], linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Knowledge Distillation Methods Comparison - PSNR', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'comparison', 'all_psnr_comparison.png'), dpi=200)
    plt.close()

    # Loss 비교 그래프
    fig, ax = plt.subplots(figsize=(12, 7))
    for kd_type, df in all_logs.items():
        ax.plot(df['epoch'], df['total_loss'], label=f'{kd_type.upper()} KD',
                color=colors[kd_type], linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Knowledge Distillation Methods Comparison - Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'comparison', 'all_loss_comparison.png'), dpi=200)
    plt.close()

    # 최종 요약 CSV 생성
    summary_data = []
    for kd_type, df in all_logs.items():
        summary_data.append({
            'method': f'{kd_type}_kd',
            'final_psnr': df['val_psnr'].iloc[-1],
            'best_psnr': df['val_psnr'].max(),
            'best_epoch': df.loc[df['val_psnr'].idxmax(), 'epoch'],
            'final_loss': df['total_loss'].iloc[-1],
            'min_loss': df['total_loss'].min()
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(SUMMARY_DIR, 'training_summary.csv'), index=False)

    print("📊 비교 그래프 및 요약 저장 완료!")
    print(f"   - {os.path.join(PLOTS_DIR, 'comparison', 'all_psnr_comparison.png')}")
    print(f"   - {os.path.join(PLOTS_DIR, 'comparison', 'all_loss_comparison.png')}")
    print(f"   - {os.path.join(SUMMARY_DIR, 'training_summary.csv')}")

    return True


def calculate_fakd_loss(fm_teacher, fm_student):
    """
    Feature-Affinity based Knowledge Distillation (FaKD) 손실을 계산합니다.
    피처맵의 2차 통계 정보(Gram 행렬)를 비교합니다.
    """
    # fm_teacher, fm_student shape: (B, C, H, W)
    
    # 1. 피처맵을 (B, C, H*W) 형태로 재구성합니다.
    b, c, h, w = fm_teacher.shape
    fm_teacher_reshaped = fm_teacher.view(b, c, h * w)
    fm_student_reshaped = fm_student.view(b, c, h * w)

    # 2. 채널 차원을 따라 L2 정규화를 수행합니다.
    fm_teacher_normalized = F.normalize(fm_teacher_reshaped, p=2, dim=1)
    fm_student_normalized = F.normalize(fm_student_reshaped, p=2, dim=1)

    # 3. Gram 행렬 (Affinity Matrix)을 계산합니다.
    # (B, C, N) -> (B, N, C)로 전치 후 행렬 곱셈
    affinity_teacher = torch.bmm(fm_teacher_normalized.transpose(1, 2), fm_teacher_normalized)
    affinity_student = torch.bmm(fm_student_normalized.transpose(1, 2), fm_student_normalized)
    
    # 4. 두 Affinity 행렬 간의 L1 손실을 계산합니다.
    loss = F.l1_loss(affinity_student, affinity_teacher)
    
    return loss

def main():
    # --- 0. 스크립트 실행 위치 보정 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory changed to: {os.getcwd()}")

    # --- 결과 디렉토리 생성 ---
    ensure_results_dirs()

    # --- 1. 인자 파싱 및 설정 ---
    parser = argparse.ArgumentParser(description="Pruned CATANet Fine-tuning with Knowledge Distillation")
    parser.add_argument("--config", required=True, help="모델 및 데이터셋 설정을 담은 YAML 파일")
    parser.add_argument("--teacher_weights", required=True, help="원본 교사 모델 가중치 경로")
    parser.add_argument("--pruned_weights", required=True, help="가지치기된 학생 모델 가중치 경로")
    parser.add_argument("--save_path", required=True, help="파인튜닝된 모델을 저장할 경로")
    parser.add_argument("--epochs", type=int, default=10, help="파인튜닝 에폭 수")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--alpha", type=float, default=0.8, help="Output Distillation Loss 가중치")
    parser.add_argument("--distillation_type", type=str, default="output", choices=["output", "feature", "fakd"], help="증류 타입 선택")
    parser.add_argument("--beta", type=float, default=0.5, help="Feature/FaKD Distillation Loss 가중치")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 로그 파일 설정 (이어서 학습 지원) ---
    log_path = get_log_path(args.distillation_type)
    log_data, start_epoch = load_existing_log(log_path)
    end_epoch = start_epoch + args.epochs - 1
    print(f"📊 학습 범위: {start_epoch} ~ {end_epoch} 에폭")

    # --- 2. 데이터셋 로드 ---
    train_opt, val_opt = config['datasets']['train'], config['datasets']['val']
    train_opt['scale'], val_opt['scale'] = args.scale, args.scale
    train_opt['phase'], val_opt['phase'] = 'train', 'val'
    
    train_set = build_dataset(train_opt)
    val_set = build_dataset(val_opt)

    train_loader = DataLoader(train_set, batch_size=train_opt['batch_size_per_gpu'], shuffle=True, num_workers=train_opt.get('num_worker_per_gpu', 4), pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)
    print(f"데이터셋 로드 완료. Train: {len(train_set)}개, Val: {len(val_set)}개")

    # --- 3. 모델 로드 ---
    print("교사 모델 로딩...")
    teacher_model = get_catanet_teacher_model(weights_path=args.teacher_weights, upscale=args.scale).to(device)
    teacher_model.eval()

    print("학생 모델 로딩 (가지치기된 가중치)...")
    student_model = get_catanet_teacher_model(weights_path=None, upscale=args.scale).to(device)
    pruned_state = torch.load(args.pruned_weights, map_location=device)['params']
    student_model.load_state_dict(pruned_state, strict=False)
    student_model.train()
    
    teacher_hook = CATANetModelHooking(args=None, model=teacher_model)
    student_hook = CATANetModelHooking(args=None, model=student_model)

    # MODIFIED: 'feature' 또는 'fakd'일 때 hook을 등록
    if args.distillation_type in ['feature', 'fakd']:
        print(f"'{args.distillation_type}' 증류를 위해 hook을 활성화합니다.")
        teacher_hook.apply_mask_and_hooks()
        student_hook.apply_mask_and_hooks()
    
    print("모델 로드 및 후킹 완료.")

    # --- 4. 옵티마이저 및 손실 함수 설정 ---
    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr)
    l1_loss = torch.nn.L1Loss().to(device)

    # 기존 로그에서 best_psnr 복원
    best_psnr = max([d['val_psnr'] for d in log_data], default=0.0)
    if best_psnr > 0:
        print(f"📈 기존 최고 PSNR: {best_psnr:.4f}")

    # --- 5. 파인튜닝 학습 루프 ---
    for epoch in range(start_epoch, end_epoch + 1):
        student_model.train()
        total_loss = 0
        total_task_loss = 0
        total_distill_loss = 0
        total_intermediate_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{end_epoch}", unit="batch")

        for batch in pbar:
            lq, gt = batch['lq'].to(device), batch['gt'].to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_output, teacher_fms = teacher_hook.forwardPass(lq)
            student_output, student_fms = student_hook.forwardPass(lq)

            loss_task = l1_loss(student_output, gt)
            loss_distill_output = l1_loss(student_output, teacher_output)
            loss = (1 - args.alpha) * loss_task + args.alpha * loss_distill_output

            intermediate_loss = 0
            if args.distillation_type == 'feature' or args.distillation_type == 'fakd':
                if not (student_fms and teacher_fms):
                    print("경고: 피처맵을 가져올 수 없어 중간 증류를 건너뜁니다.")
                else:
                    for student_fm, teacher_fm in zip(student_fms, teacher_fms):
                        if args.distillation_type == 'feature':
                            intermediate_loss += l1_loss(student_fm, teacher_fm)
                        elif args.distillation_type == 'fakd':
                            intermediate_loss += calculate_fakd_loss(teacher_fm, student_fm)
                    loss += args.beta * intermediate_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_task_loss += loss_task.item()
            total_distill_loss += loss_distill_output.item()
            if isinstance(intermediate_loss, torch.Tensor):
                total_intermediate_loss += intermediate_loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        avg_task_loss = total_task_loss / len(train_loader)
        avg_distill_loss = total_distill_loss / len(train_loader)
        avg_intermediate_loss = total_intermediate_loss / len(train_loader)
        print(f"Epoch {epoch} 완료. 평균 Loss: {avg_loss:.4f}")

        # --- 6. 검증 (Validation) ---
        student_model.eval()
        current_psnr = 0
        with torch.no_grad():
            for batch in val_loader:
                lq, gt = batch['lq'].to(device), batch['gt'].to(device)
                student_output = student_model(lq)
                output_img, gt_img = tensor2img(student_output), tensor2img(gt)
                current_psnr += calculate_psnr(output_img, gt_img, crop_border=args.scale, test_y_channel=True)

        avg_psnr = current_psnr / len(val_loader)
        print(f"검증 완료. 평균 PSNR: {avg_psnr:.4f}")

        # --- 로그 기록 ---
        log_data.append({
            'epoch': epoch,
            'total_loss': avg_loss,
            'task_loss': avg_task_loss,
            'distill_loss': avg_distill_loss,
            'intermediate_loss': avg_intermediate_loss,
            'val_psnr': avg_psnr,
            'lr': args.lr,
            'alpha': args.alpha,
            'beta': args.beta
        })
        save_log(log_data, log_path)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(student_model.state_dict(), args.save_path)
            print(f"최고 성능 달성! 모델을 '{args.save_path}'에 저장했습니다. (PSNR: {best_psnr:.4f})")

        # 훅 재설정 (메모리 누수 방지)
        if args.distillation_type in ['feature', 'fakd']:
            teacher_hook.purge_hooks()
            student_hook.purge_hooks()
            if epoch < end_epoch:
                 teacher_hook.apply_mask_and_hooks()
                 student_hook.apply_mask_and_hooks()

    print(f"\n--- 파인튜닝 완료 ---\n최고 PSNR: {best_psnr:.4f}")
    print(f"최종 모델은 '{args.save_path}'에 저장되었습니다.")
    print(f"📊 로그 저장됨: {log_path}")

    # --- 7. 그래프 생성 ---
    current_total_epochs = len(log_data)
    print(f"\n📈 현재 총 에폭: {current_total_epochs}/1000")

    # 개별 그래프는 항상 생성
    generate_individual_plots(log_path, args.distillation_type)

    # 1000 에폭 달성 시 비교 그래프 생성 시도
    if current_total_epochs >= 1000:
        print("🎉 1000 에폭 달성! 비교 그래프 생성을 시도합니다...")
        generate_comparison_plots()


if __name__ == '__main__':
    main()