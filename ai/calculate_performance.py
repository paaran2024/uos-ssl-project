import torch
import torch.utils.data
import subprocess
import sys
import os
from tqdm import tqdm
import numpy as np

# --- 1. 의존성 설치 및 임포트 ---

# thop 설치
try:
    from thop import profile
except ImportError:
    print("thop 라이브러리를 설치합니다...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "thop"])
    from thop import profile

# 프로젝트 커스텀 모듈 임포트
from scripts.load_catanet import get_catanet_teacher_model
from basicsr.archs.catanet_arch import CATANet
from basicsr.data import build_dataset
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import tensor2img

# --- 2. 헬퍼 함수 ---

def calculate_model_complexity(model, input_tensor):
    """주어진 모델의 FLOPs와 총 파라미터를 계산합니다."""
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = macs * 2
    return flops, params

def count_nonzero_parameters(model):
    """모델의 0이 아닌 파라미터 수를 계산합니다."""
    return sum(p.count_nonzero().item() for p in model.parameters())

def calculate_model_quality(model, dataloader, device):
    """주어진 모델의 PSNR/SSIM을 데이터셋에 대해 평가합니다."""
    model.eval()
    model.to(device)
    
    total_psnr = 0
    total_ssim = 0
    pbar = tqdm(dataloader, desc=f'Evaluating {model.__class__.__name__}', unit='image')

    for batch in pbar:
        lq, gt = batch['lq'].to(device), batch['gt'].to(device)
        with torch.no_grad():
            output = model(lq)
        
        output_img = tensor2img(output)
        gt_img = tensor2img(gt)
        
        total_psnr += calculate_psnr(output_img, gt_img, crop_border=2, test_y_channel=True)
        total_ssim += calculate_ssim(output_img, gt_img, crop_border=2, test_y_channel=True)
        
        pbar.set_postfix({
            'PSNR': f"{total_psnr / len(pbar):.4f}",
            'SSIM': f"{total_ssim / len(pbar):.4f}"
        })

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    return avg_psnr, avg_ssim


# --- 3. 메인 스크립트 ---

def main():
    # --- 설정 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory changed to: {os.getcwd()}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    original_model_path = 'weights/CATANet-L_x2.pth'
    finetuned_model_path = 'weights/catanet_finetuned_feature_kd.pth'
    upscale = 2

    # --- 복잡도 계산 (FLOPs, Params) ---
    input_h, input_w = 64, 64 
    complexity_input_tensor = torch.randn(1, 3, input_h, input_w)
    print(f"복잡도(FLOPs/Params)는 {input_h}x{input_w} 크기의 입력 텐서를 기준으로 계산합니다.\n")

    # 원본 모델
    original_model = get_catanet_teacher_model(weights_path=original_model_path, upscale=upscale)
    original_flops, original_params = calculate_model_complexity(original_model, complexity_input_tensor)
    original_nonzero_params = count_nonzero_parameters(original_model)

    # 파인튜닝된 모델
    finetuned_model = CATANet(upscale=upscale)
    finetuned_state = torch.load(finetuned_model_path, map_location='cpu')
    if 'params' in finetuned_state:
        finetuned_state = finetuned_state['params']
    finetuned_model.load_state_dict(finetuned_state)
    finetuned_flops, finetuned_params = calculate_model_complexity(finetuned_model, complexity_input_tensor)
    finetuned_nonzero_params = count_nonzero_parameters(finetuned_model)

    # --- 품질 평가 (PSNR, SSIM) ---
    print("\n품질(PSNR/SSIM)은 Set5 데이터셋을 기준으로 평가합니다.")
    dataset_opt = {
        'name': 'Set5',
        'type': 'PairedImageDataset',
        'dataroot_gt': 'datasets/Set5/HR',
        'dataroot_lq': 'datasets/Set5/LR_bicubic/X2',
        'filename_tmpl': '{}x2',
        'io_backend': {'type': 'disk'},
        'scale': upscale,
        'phase': 'val'
    }
    
    try:
        test_set = build_dataset(dataset_opt)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    except Exception as e:
        print(f"\n[오류] 테스트 데이터셋을 로드할 수 없습니다: {e}")
        print("프로젝트의 'datasets/Set5' 경로에 테스트 데이터가 올바르게 위치해 있는지 확인하세요.")
        print("성능 평가는 복잡도 계산만으로 제한됩니다.\n")
        test_loader = None

    original_psnr, original_ssim = (None, None)
    finetuned_psnr, finetuned_ssim = (None, None)

    if test_loader:
        original_psnr, original_ssim = calculate_model_quality(original_model, test_loader, device)
        finetuned_psnr, finetuned_ssim = calculate_model_quality(finetuned_model, test_loader, device)
        
    # --- 결과 종합 및 출력 ---
    flops_reduction = (original_flops - finetuned_flops) / original_flops * 100
    params_reduction = (original_params - finetuned_params) / original_params * 100
    nonzero_params_reduction = (original_nonzero_params - finetuned_nonzero_params) / original_nonzero_params * 100 if original_nonzero_params > 0 else 0
    psnr_change = finetuned_psnr - original_psnr if all(x is not None for x in [original_psnr, finetuned_psnr]) else None
    ssim_change = finetuned_ssim - original_ssim if all(x is not None for x in [original_ssim, finetuned_ssim]) else None
    
    print("\n--- 최종 성능 종합 비교 ---")
    header = f"{ 'Metric':<20} | { 'Original Model':<20} | { 'Finetuned Model':<20} | { 'Change':<15}"
    print(header)
    print("-" * (len(header) + 5))
    
    if all(x is not None for x in [original_psnr, finetuned_psnr]):
        print(f"{ 'PSNR (dB)':<20} | {original_psnr:<20.4f} | {finetuned_psnr:<20.4f} | {psnr_change:+.4f}")
    if all(x is not None for x in [original_ssim, finetuned_ssim]):
        print(f"{ 'SSIM':<20} | {original_ssim:<20.4f} | {finetuned_ssim:<20.4f} | {ssim_change:+.4f}")
        
    print(f"{ 'GFLOPs':<20} | {original_flops / 1e9:<20.4f} | {finetuned_flops / 1e9:<20.4f} | {f'-{flops_reduction:.2f}%'}")
    print(f"{ 'Total Params (M)':<20} | {original_params / 1e6:<20.4f} | {finetuned_params / 1e6:<20.4f} | {f'-{params_reduction:.2f}%'}")
    print(f"{ 'Non-Zero Params (M)':<20} | {original_nonzero_params / 1e6:<20.4f} | {finetuned_nonzero_params / 1e6:<20.4f} | {f'-{nonzero_params_reduction:.2f}%'}")
    print("-" * (len(header) + 5))
    print("\n* PSNR/SSIM의 'Change'는 (Finetuned - Original) 값을 의미합니다 (높을수록 좋음).")
    print("* GFLOPs/Params의 'Change'는 감소율을 의미합니다 (클수록 좋음).")
    print("* Non-Zero Params (M): 값이 0이 아닌 파라미터의 수 (희소성/프루닝 효과 측정).")


if __name__ == '__main__':
    main()