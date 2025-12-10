#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SET5 데이터셋 평가 스크립트

CATANet 모델의 Super Resolution 성능을 PSNR/SSIM으로 평가합니다.
원본(.pth) 모델과 변환된(.ptl) 모델을 비교할 수 있습니다.

Usage:
    python scripts/evaluate_set5.py --weights weights/CATANet-L_x2.pth
    python scripts/evaluate_set5.py --ptl weights/CATANet-L_x2_128.ptl
    python scripts/evaluate_set5.py --weights weights/CATANet-L_x2.pth --ptl weights/CATANet-L_x2_128.ptl
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.catanet_arch import CATANet


@dataclass
class ImageResult:
    """단일 이미지 평가 결과"""
    name: str
    psnr: float
    ssim: float
    size: Tuple[int, int]


@dataclass
class EvaluationResult:
    """전체 평가 결과"""
    model_name: str
    images: List[ImageResult]

    @property
    def avg_psnr(self) -> float:
        return np.mean([img.psnr for img in self.images])

    @property
    def avg_ssim(self) -> float:
        return np.mean([img.ssim for img in self.images])


class ImageMetrics:
    """이미지 품질 메트릭 계산 클래스"""

    @staticmethod
    def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
        """RGB를 YCbCr로 변환 (Y 채널만 반환)"""
        return 16 + (65.481 * img[:, :, 0] +
                     128.553 * img[:, :, 1] +
                     24.966 * img[:, :, 2]) / 255.0

    @staticmethod
    def calculate_psnr(sr: np.ndarray, hr: np.ndarray, border: int = 2) -> float:
        """
        PSNR 계산 (Y 채널, border crop)

        Args:
            sr: Super Resolution 이미지 [H, W, C], 범위 [0, 1]
            hr: High Resolution 원본 이미지 [H, W, C], 범위 [0, 1]
            border: 가장자리 제거 픽셀 수

        Returns:
            PSNR 값 (dB)
        """
        y_sr = ImageMetrics.rgb_to_ycbcr(sr * 255.0)
        y_hr = ImageMetrics.rgb_to_ycbcr(hr * 255.0)

        # Border crop
        y_sr = y_sr[border:-border, border:-border]
        y_hr = y_hr[border:-border, border:-border]

        mse = np.mean((y_sr - y_hr) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(255.0 ** 2 / mse)

    @staticmethod
    def calculate_ssim(sr: np.ndarray, hr: np.ndarray, border: int = 2) -> float:
        """
        SSIM 계산 (Y 채널, border crop)

        Args:
            sr: Super Resolution 이미지 [H, W, C], 범위 [0, 1]
            hr: High Resolution 원본 이미지 [H, W, C], 범위 [0, 1]
            border: 가장자리 제거 픽셀 수

        Returns:
            SSIM 값 (0~1)
        """
        y_sr = ImageMetrics.rgb_to_ycbcr(sr * 255.0)
        y_hr = ImageMetrics.rgb_to_ycbcr(hr * 255.0)

        # Border crop
        y_sr = y_sr[border:-border, border:-border]
        y_hr = y_hr[border:-border, border:-border]

        # SSIM 상수
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu_sr = np.mean(y_sr)
        mu_hr = np.mean(y_hr)
        sigma_sr_sq = np.var(y_sr)
        sigma_hr_sq = np.var(y_hr)
        sigma_sr_hr = np.mean((y_sr - mu_sr) * (y_hr - mu_hr))

        ssim = ((2 * mu_sr * mu_hr + C1) * (2 * sigma_sr_hr + C2)) / \
               ((mu_sr ** 2 + mu_hr ** 2 + C1) * (sigma_sr_sq + sigma_hr_sq + C2))

        return ssim


class SET5Evaluator:
    """SET5 데이터셋 평가 클래스"""

    # SET5 이미지 목록 (LR 파일명, HR 파일명, 크기)
    IMAGES = [
        ('babyx2.png', 'baby.png'),
        ('birdx2.png', 'bird.png'),
        ('butterflyx2.png', 'butterfly.png'),
        ('headx2.png', 'head.png'),
        ('womanx2.png', 'woman.png'),
    ]

    def __init__(self, data_dir: str = 'datasets/benchmark', dataset_name: str = 'Set5', scale: int = 2):
        """
        Args:
            data_dir: 벤치마크 데이터셋 루트 경로
            dataset_name: 데이터셋 이름 (Set5, Set14 등)
            scale: 업스케일 배율
        """
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.scale = scale
        self.hr_dir = self.data_dir / 'HR' / dataset_name
        self.lr_dir = self.data_dir / 'LR' / 'LRBI' / dataset_name / f'x{scale}'

    @staticmethod
    def load_image(path: Path) -> torch.Tensor:
        """이미지를 텐서로 로드"""
        img = Image.open(path).convert('RGB')
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """텐서를 numpy 배열로 변환"""
        arr = tensor.squeeze(0).permute(1, 2, 0).numpy()
        return np.clip(arr, 0, 1)

    def evaluate_model(self, model, model_name: str) -> EvaluationResult:
        """
        모델 평가 실행

        Args:
            model: PyTorch 모델 또는 TorchScript 모델
            model_name: 결과 출력용 모델 이름

        Returns:
            EvaluationResult 객체
        """
        results = []

        for lr_name, hr_name in self.IMAGES:
            lr_path = self.lr_dir / lr_name
            hr_path = self.hr_dir / hr_name

            if not lr_path.exists():
                continue

            # 이미지 로드
            lr_tensor = self.load_image(lr_path)
            hr_np = self.tensor_to_numpy(self.load_image(hr_path))

            # 추론
            try:
                with torch.no_grad():
                    sr_tensor = model(lr_tensor)
                sr_np = self.tensor_to_numpy(sr_tensor)

                # HR 크기에 맞춤
                if sr_np.shape[:2] != hr_np.shape[:2]:
                    sr_np = sr_np[:hr_np.shape[0], :hr_np.shape[1], :]

                # 메트릭 계산
                psnr = ImageMetrics.calculate_psnr(sr_np, hr_np)
                ssim = ImageMetrics.calculate_ssim(sr_np, hr_np)

                results.append(ImageResult(
                    name=hr_name.replace('.png', ''),
                    psnr=psnr,
                    ssim=ssim,
                    size=lr_tensor.shape[2:]
                ))
            except RuntimeError as e:
                # PTL 크기 불일치 오류
                results.append(ImageResult(
                    name=hr_name.replace('.png', ''),
                    psnr=float('nan'),
                    ssim=float('nan'),
                    size=lr_tensor.shape[2:]
                ))

        return EvaluationResult(model_name=model_name, images=results)


def load_original_model(weights_path: str, upscale: int = 2):
    """원본 PyTorch 모델 로드"""
    model = CATANet(in_chans=3, upscale=upscale)
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('params', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_ptl_model(ptl_path: str):
    """PTL 모델 로드"""
    return torch.jit.load(ptl_path, map_location='cpu')


def print_results(result: EvaluationResult) -> None:
    """평가 결과 출력"""
    print(f'\n[{result.model_name}]')
    print('-' * 50)
    print(f'{"Image":<12} {"Size":<12} {"PSNR (dB)":<12} {"SSIM":<10}')
    print('-' * 50)

    for img in result.images:
        size_str = f'{img.size[0]}x{img.size[1]}'
        if np.isnan(img.psnr):
            print(f'{img.name:<12} {size_str:<12} {"N/A":<12} {"N/A":<10}')
        else:
            print(f'{img.name:<12} {size_str:<12} {img.psnr:<12.4f} {img.ssim:<10.4f}')

    print('-' * 50)
    valid_images = [img for img in result.images if not np.isnan(img.psnr)]
    if valid_images:
        avg_psnr = np.mean([img.psnr for img in valid_images])
        avg_ssim = np.mean([img.ssim for img in valid_images])
        print(f'{"Average":<12} {"":<12} {avg_psnr:<12.4f} {avg_ssim:<10.4f}')


def main():
    parser = argparse.ArgumentParser(
        description='SET5 데이터셋으로 CATANet 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/evaluate_set5.py --weights weights/CATANet-L_x2.pth
    python scripts/evaluate_set5.py --ptl weights/CATANet-L_x2_128.ptl
    python scripts/evaluate_set5.py --weights weights/CATANet-L_x2.pth --ptl weights/CATANet-L_x2_128.ptl
        """
    )
    parser.add_argument('--weights', type=str, default=None,
                        help='원본 .pth 가중치 파일')
    parser.add_argument('--ptl', type=str, default=None,
                        help='변환된 .ptl 파일')
    parser.add_argument('--upscale', type=int, default=2,
                        help='업스케일 배율')
    parser.add_argument('--data_dir', type=str, default='datasets/benchmark',
                        help='벤치마크 데이터셋 루트 경로')
    parser.add_argument('--dataset', type=str, default='Set5',
                        help='데이터셋 이름 (Set5, Set14 등)')

    args = parser.parse_args()

    # 기본값 설정
    if args.weights is None and args.ptl is None:
        args.weights = 'weights/CATANet-L_x2.pth'

    print('=' * 50)
    print(f'{args.dataset} Super Resolution 평가 (x{args.upscale})')
    print('=' * 50)

    evaluator = SET5Evaluator(args.data_dir, args.dataset, args.upscale)
    results = []

    # 원본 모델 평가
    if args.weights:
        print(f'\n원본 모델 로드: {args.weights}')
        model = load_original_model(args.weights, args.upscale)
        result = evaluator.evaluate_model(model, 'Original (.pth)')
        print_results(result)
        results.append(result)

    # PTL 모델 평가
    if args.ptl:
        print(f'\nPTL 모델 로드: {args.ptl}')
        ptl_model = load_ptl_model(args.ptl)
        result = evaluator.evaluate_model(ptl_model, 'Mobile (.ptl)')
        print_results(result)
        results.append(result)

    # 비교 요약
    if len(results) == 2:
        print('\n' + '=' * 50)
        print('비교 요약')
        print('=' * 50)
        orig, ptl = results
        valid_orig = [img for img in orig.images if not np.isnan(img.psnr)]
        valid_ptl = [img for img in ptl.images if not np.isnan(img.psnr)]

        print(f'Original: {np.mean([i.psnr for i in valid_orig]):.4f} dB (평균 PSNR)')
        print(f'PTL:      {np.mean([i.psnr for i in valid_ptl]):.4f} dB (평균 PSNR)')

    return results


if __name__ == '__main__':
    main()
