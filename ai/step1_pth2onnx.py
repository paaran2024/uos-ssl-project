import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import os
import sys
from einops import rearrange

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 원본 모델 모듈 임포트
import models.catanet_arch as arch_module

# ==============================================================================
# [Monkey Patching] ONNX 변환을 위해 원본 모델의 기능을 수동 구현으로 교체
# ==============================================================================

def manual_attention(q, k, v):
    dim = q.shape[-1]
    scale = dim ** -0.5
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_probs = F.softmax(attn_scores, dim=-1)
    out = torch.matmul(attn_probs, v)
    return out

def similarity_patched(x, means):
    return torch.matmul(x, means.transpose(0, 1))

class IASA_ONNX(nn.Module):
    def __init__(self, dim, qk_dim, heads, group_size):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.group_size = group_size

    def forward(self, normed_x, idx_last, k_global, v_global):
        x = normed_x
        B, N, _ = x.shape

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))

        gs = min(N, self.group_size)
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N

        paded_q = torch.cat((q, torch.flip(q[:,N-pad_n:N, :], dims=[-2])), dim=-2)
        paded_q = rearrange(paded_q, "b (ng gs) (h d) -> b ng h gs d", ng=ng, h=self.heads)
        
        paded_k_raw = torch.cat((k, torch.flip(k[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_v_raw = torch.cat((v, torch.flip(v[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        
        k_s1 = paded_k_raw[:, :ng*gs, :].view(B, ng, gs, -1)
        k_s2 = paded_k_raw[:, gs:, :].view(B, ng, gs, -1)
        k_windows = torch.cat([k_s1, k_s2], dim=2)
        
        v_s1 = paded_v_raw[:, :ng*gs, :].view(B, ng, gs, -1)
        v_s2 = paded_v_raw[:, gs:, :].view(B, ng, gs, -1)
        v_windows = torch.cat([v_s1, v_s2], dim=2)

        paded_k = rearrange(k_windows, "b ng gs (h d) -> b ng h gs d", h=self.heads)
        paded_v = rearrange(v_windows, "b ng gs (h d) -> b ng h gs d", h=self.heads)
        
        out1 = manual_attention(paded_q, paded_k, paded_v)

        k_global = k_global.reshape(1,1,*k_global.shape).expand(B,ng,-1,-1,-1)
        v_global = v_global.reshape(1,1,*v_global.shape).expand(B,ng,-1,-1,-1)
        out2 = manual_attention(paded_q, k_global, v_global)
        
        out = out1 + out2
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]

        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)
        out = self.proj(out)
        return out

class TAB_ONNX(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=3,
                 num_tokens=8, group_size=128,
                 ema_decay = 0.999):
        super().__init__()
        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.mlp = arch_module.PreNorm(dim, arch_module.ConvFFN(dim,mlp_dim))
        self.irca_attn = arch_module.IRCA(dim,qk_dim,heads)
        self.iasa_attn = IASA_ONNX(dim,qk_dim,heads,group_size)
        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        self.conv1x1 = nn.Conv2d(dim,dim,1, bias=False)

    def forward(self, x):
        _,_,h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape
        idx_last = torch.arange(N, device=x.device).reshape(1,N).expand(B,-1)
        
        if not self.initted: x_means = self.means.detach()
        else: x_means = self.means.detach()

        k_global, v_global, x_means = self.irca_attn(x, x_means)

        with torch.no_grad():
            x_norm = F.normalize(x, dim=-1)
            means_norm = F.normalize(x_means, dim=-1)
            x_scores = torch.matmul(x_norm, means_norm.transpose(0, 1))
            x_belong_idx = torch.argmax(x_scores, dim=-1)
            idx = torch.argsort(x_belong_idx, dim=-1)
            idx_last = torch.gather(idx_last, dim=-1, index=idx).unsqueeze(-1)

        y = self.iasa_attn(x, idx_last,k_global,v_global)
        y = rearrange(y,'b (h w) c->b c h w',h=h).contiguous()
        y = self.conv1x1(y)
        x = residual + rearrange(y, 'b c h w->b (h w) c')
        x = self.mlp(x, x_size=(h, w)) + x
        return rearrange(x, 'b (h w) c->b c h w',h=h)

class Attention_ONNX(nn.Module):
    def __init__(self, dim, heads, qk_dim):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        out = manual_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)

def run():
    # === 설정 (x2로 변경) ===
    WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'weights', 'catanet_finetuned_feature_kd.pth')
    OUTPUT_ONNX_PATH = os.path.join(os.path.dirname(__file__), 'catanet_child_x2.onnx') # x2로 변경
    UPSCALE_FACTOR = 2  # x2로 변경
    INPUT_SIZE = 64
    # =====================

    print("--- [Step 1] PyTorch -> ONNX 변환 (x2, Manual Attention Patching) ---")
    
    arch_module.similarity = similarity_patched
    arch_module.IASA = IASA_ONNX
    arch_module.TAB = TAB_ONNX
    arch_module.Attention = Attention_ONNX

    device = torch.device('cpu')
    model = arch_module.CATANet(upscale=UPSCALE_FACTOR)
    model.to(device)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"오류: 가중치 파일 없음 {WEIGHTS_PATH}")
        return

    print(f"가중치 로드 중... {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    state_dict = checkpoint['params'] if 'params' in checkpoint else checkpoint
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print("가중치 로드 성공 (strict=False)")
    except Exception as e:
        print(f"가중치 로드 경고: {e}")

    model.eval()

    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    
    print("ONNX 변환 중...")
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_ONNX_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"=== 변환 완료: {OUTPUT_ONNX_PATH} ===")

if __name__ == "__main__":
    run()
