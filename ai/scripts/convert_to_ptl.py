#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CATANet PyTorch Mobile 변환 스크립트

PyTorch 모델(.pth)을 모바일용 PTL 파일로 변환합니다.
torch.jit.trace를 사용하여 지정된 입력 크기에 최적화된 모델을 생성합니다.

Usage:
    python scripts/convert_to_ptl.py --input_size 128
    python scripts/convert_to_ptl.py --input_size 256 --output weights/custom.ptl

Note:
    - trace 방식은 지정된 입력 크기에서만 동작합니다.
    - 모바일 앱에서 사용할 이미지 크기에 맞춰 변환하세요.
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from models.catanet_arch import CATANet


class PTLConverter:
    """CATANet을 PyTorch Mobile 형식으로 변환하는 클래스"""

    def __init__(self, weights_path: str, upscale: int = 2):
        """
        Args:
            weights_path: 원본 .pth 가중치 파일 경로
            upscale: 업스케일 배율 (2, 3, 4)
        """
        self.weights_path = Path(weights_path)
        self.upscale = upscale
        self.model = None

    def load_model(self) -> None:
        """모델 로드 및 가중치 적용"""
        self.model = CATANet(in_chans=3, upscale=self.upscale)

        checkpoint = torch.load(self.weights_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('params', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def convert(self, input_size: int, output_path: str) -> dict:
        """
        PTL 파일로 변환

        Args:
            input_size: 입력 이미지 크기 (정사각형)
            output_path: 출력 PTL 파일 경로

        Returns:
            변환 결과 정보 딕셔너리
        """
        if self.model is None:
            self.load_model()

        # TorchScript trace
        example_input = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            traced_model = torch.jit.trace(
                self.model,
                example_input,
                check_trace=False  # 비결정적 연산(argsort) 때문에 필요
            )

        # 모바일 최적화 및 저장
        optimized_model = optimize_for_mobile(traced_model)
        optimized_model._save_for_lite_interpreter(output_path)

        # 결과 정보
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        output_size = input_size * self.upscale

        return {
            'input_size': input_size,
            'output_size': output_size,
            'file_size_mb': file_size_mb,
            'output_path': output_path
        }


def main():
    parser = argparse.ArgumentParser(
        description='CATANet을 PyTorch Mobile(.ptl)로 변환',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/convert_to_ptl.py --input_size 128
    python scripts/convert_to_ptl.py --input_size 256 --upscale 2
    python scripts/convert_to_ptl.py --input_size 1024 --output weights/CATANet_1024.ptl
        """
    )
    parser.add_argument('--weights', type=str, default='weights/CATANet-L_x2.pth',
                        help='원본 가중치 파일 경로')
    parser.add_argument('--upscale', type=int, default=2, choices=[2, 3, 4],
                        help='업스케일 배율')
    parser.add_argument('--input_size', type=int, default=128,
                        help='입력 이미지 크기 (정사각형)')
    parser.add_argument('--output', type=str, default=None,
                        help='출력 파일 경로 (기본: weights/CATANet-L_x{upscale}_{size}.ptl)')

    args = parser.parse_args()

    # 출력 경로 설정
    if args.output is None:
        args.output = f'weights/CATANet-L_x{args.upscale}_{args.input_size}.ptl'

    # 변환 실행
    print(f'CATANet PTL 변환')
    print(f'=' * 50)
    print(f'입력 가중치: {args.weights}')
    print(f'입력 크기:   {args.input_size}x{args.input_size}')
    print(f'업스케일:    x{args.upscale}')
    print(f'=' * 50)

    converter = PTLConverter(args.weights, args.upscale)
    converter.load_model()
    print(f'[1/3] 모델 로드 완료')

    print(f'[2/3] TorchScript 변환 중...')
    result = converter.convert(args.input_size, args.output)

    print(f'[3/3] 저장 완료')
    print(f'=' * 50)
    print(f'출력 파일:   {result["output_path"]}')
    print(f'파일 크기:   {result["file_size_mb"]:.2f} MB')
    print(f'입력 크기:   {result["input_size"]}x{result["input_size"]}')
    print(f'출력 크기:   {result["output_size"]}x{result["output_size"]}')
    print(f'=' * 50)

    return result


if __name__ == '__main__':
    main()
