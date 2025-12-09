# -*- coding: utf-8 -*-
"""
CATANet PyTorch 모델을 TFLite로 변환하는 스크립트
참조: https://github.com/cornpip/pt_to_tflite

변환 경로: PyTorch -> ONNX -> TensorFlow SavedModel -> TFLite
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.catanet_arch import CATANet


class CATANetONNXWrapper(nn.Module):
    """
    ONNX 변환을 위한 CATANet 래퍼.
    동적 연산을 정적 연산으로 대체하여 ONNX 호환성 확보.
    """

    def __init__(self, model: CATANet):
        super().__init__()
        self.model = model
        self.upscale = model.upscale
        self.dim = model.dim
        self.block_num = model.block_num
        self.patch_size = model.patch_size

    def forward(self, x):
        # 업스케일링된 기본 이미지 (residual)
        if self.upscale != 1:
            base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        else:
            base = x

        # 얕은 특징 추출
        x = self.model.first_conv(x)

        # 깊은 특징 추출 (간소화)
        residual_feat = x
        for i in range(self.block_num):
            block_residual = x
            tab, lrsa = self.model.blocks[i]

            # TAB 간소화: 전체 셀프 어텐션으로 대체
            x = self._simplified_tab_forward(x, tab)

            # LRSA 간소화: 전체 셀프 어텐션으로 대체
            x = self._simplified_lrsa_forward(x, lrsa)

            x = block_residual + self.model.mid_convs[i](x)

        x = residual_feat + x

        # 이미지 재구성
        if self.upscale == 4:
            out = self.model.lrelu(self.model.pixel_shuffle(self.model.upconv1(x)))
            out = self.model.lrelu(self.model.pixel_shuffle(self.model.upconv2(out)))
        elif self.upscale == 1:
            out = x
        else:
            out = self.model.lrelu(self.model.pixel_shuffle(self.model.upconv(x)))

        out = self.model.last_conv(out) + base
        return out

    def _simplified_tab_forward(self, x, tab):
        """TAB을 간소화된 셀프 어텐션으로 대체"""
        b, c, h, w = x.shape
        # (B, C, H, W) -> (B, H*W, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        residual = x_flat

        x_norm = tab.norm(x_flat)

        # 셀프 어텐션
        q = tab.iasa_attn.to_q(x_norm)
        k = tab.iasa_attn.to_k(x_norm)
        v = tab.iasa_attn.to_v(x_norm)

        heads = tab.iasa_attn.heads
        head_dim = q.shape[-1] // heads

        # (B, N, heads*head_dim) -> (B, heads, N, head_dim)
        q = q.reshape(b, h * w, heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, h * w, heads, head_dim).permute(0, 2, 1, 3)
        v_head_dim = v.shape[-1] // heads
        v = v.reshape(b, h * w, heads, v_head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # (B, heads, N, head_dim) -> (B, N, heads*head_dim)
        out = out.permute(0, 2, 1, 3).reshape(b, h * w, -1)
        out = tab.iasa_attn.proj(out)

        # (B, N, C) -> (B, C, H, W)
        y = out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        y = tab.conv1x1(y)

        # residual + MLP
        x_flat = residual + y.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x_flat = tab.mlp(x_flat, x_size=(h, w)) + x_flat

        return x_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)

    def _simplified_lrsa_forward(self, x, lrsa):
        """LRSA를 간소화된 셀프 어텐션으로 대체"""
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        attn_layer, ff_layer = lrsa.layer

        # PreNorm + Attention
        x_norm = attn_layer.norm(x_flat)
        attn = attn_layer.fn

        q = attn.to_q(x_norm)
        k = attn.to_k(x_norm)
        v = attn.to_v(x_norm)

        heads = attn.heads
        head_dim = q.shape[-1] // heads

        q = q.reshape(b, h * w, heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, h * w, heads, head_dim).permute(0, 2, 1, 3)
        v_head_dim = v.shape[-1] // heads
        v = v.reshape(b, h * w, heads, v_head_dim).permute(0, 2, 1, 3)

        scale = head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.permute(0, 2, 1, 3).reshape(b, h * w, -1)
        out = attn.proj(out)

        x_flat = x_flat + out

        # PreNorm + FFN
        x_flat = ff_layer(x_flat, x_size=(h, w)) + x_flat

        return x_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)


def load_catanet(weights_path: str, upscale: int = 2) -> nn.Module:
    """CATANet 모델 로드"""
    print(f"[model] CATANet 생성 중 (upscale={upscale})")
    model = CATANet(in_chans=3, upscale=upscale)

    if weights_path and os.path.exists(weights_path):
        print(f"[model] 가중치 로드: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')

        if isinstance(checkpoint, dict) and 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        print("[model] 가중치 로드 완료")
    else:
        print(f"[model] 경고: 가중치 파일 없음 - {weights_path}")

    model.eval()
    return model


def ensure_dirs():
    """출력 디렉토리 생성"""
    for p in ["onnx", "saved_model", "tflite"]:
        os.makedirs(p, exist_ok=True)


def export_onnx(model: nn.Module, dummy_input: torch.Tensor, onnx_path: str):
    """PyTorch -> ONNX 변환"""
    import onnx

    print(f"[onnx] 변환 중...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
    )
    print(f"[onnx] 저장 완료: {onnx_path}")

    # ONNX 모델 검증
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[onnx] 모델 검증 완료")


def onnx_to_tf(onnx_path: str, saved_model_dir: str):
    """ONNX -> TensorFlow SavedModel 변환"""
    import onnx
    import tensorflow as tf
    from onnx_tf.backend import prepare

    print(f"[tf] ONNX -> SavedModel 변환 중...")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(saved_model_dir)

    # NHWC 입력을 위한 서빙 함수 재정의
    model = tf.saved_model.load(saved_model_dir)
    concrete_func = model.signatures["serving_default"]
    input_tensor = concrete_func.inputs[0]
    input_shape = input_tensor.shape.as_list()  # [1, C, H, W]

    if len(input_shape) == 4:
        nhwc_shape = [input_shape[0], input_shape[2], input_shape[3], input_shape[1]]

        @tf.function(input_signature=[tf.TensorSpec(shape=nhwc_shape, dtype=tf.float32)])
        def new_serving_fn(inputs):
            nchw_input = tf.transpose(inputs, [0, 3, 1, 2])
            outputs = concrete_func(nchw_input)
            return outputs

        tf.saved_model.save(model, saved_model_dir, signatures={"serving_default": new_serving_fn})

    print(f"[tf] SavedModel 저장 완료: {saved_model_dir}")


def tf_to_tflite(saved_model_dir: str, tflite_path: str, optimize: bool = True):
    """TensorFlow SavedModel -> TFLite 변환"""
    import tensorflow as tf

    print(f"[tflite] SavedModel -> TFLite 변환 중...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # TF Select ops 활성화 (Erf 등 TFLite에서 미지원 연산 처리)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(tflite_path) / 1024 / 1024
    print(f"[tflite] 저장 완료: {tflite_path} ({size_mb:.2f} MB)")


def convert(weights_path: str, upscale: int, input_size: int, result_name: str):
    """전체 변환 파이프라인"""
    ensure_dirs()

    device = torch.device("cpu")

    # 1) 모델 로드
    model = load_catanet(weights_path, upscale)
    wrapper = CATANetONNXWrapper(model)
    wrapper.eval().to(device)

    # 2) PyTorch -> ONNX
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    onnx_path = f"./onnx/{result_name}.onnx"
    export_onnx(wrapper, dummy_input, onnx_path)

    # 3) ONNX -> TensorFlow SavedModel
    saved_model_dir = f"./saved_model/{result_name}"
    onnx_to_tf(onnx_path, saved_model_dir)

    # 4) TensorFlow -> TFLite
    tflite_path = f"./tflite/{result_name}.tflite"
    tf_to_tflite(saved_model_dir, tflite_path)

    print(f"\n[완료] TFLite 모델: {tflite_path}")


def main():
    parser = argparse.ArgumentParser(description="CATANet을 TFLite로 변환")
    parser.add_argument("--weights", type=str, required=True, help="PyTorch 가중치 파일 경로")
    parser.add_argument("--upscale", type=int, default=2, help="업스케일 비율 (2, 3, 4)")
    parser.add_argument("--input_size", type=int, default=64, help="입력 이미지 크기")
    parser.add_argument("--result_name", type=str, default="catanet", help="출력 파일 이름")

    args = parser.parse_args()
    convert(args.weights, args.upscale, args.input_size, args.result_name)


if __name__ == "__main__":
    main()
