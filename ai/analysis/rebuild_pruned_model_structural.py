import sys
import os

# Add the parent directory (ai) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import argparse
import yaml
from collections import OrderedDict

# --- 헬퍼: 기존 아키텍처 파일에서 필요한 클래스들을 가져옵니다 ---
# PrunedCATANetStructural이 CATANet을 상속하므로, 원본 클래스들이 필요합니다.
from basicsr.archs.catanet_arch import CATANet, TAB, LRSA, ConvFFN, Attention, IASA, PreNorm, IRCA

# =====================================================================================
# 1. 프루닝된 버전의 새로운 모듈 정의 (기존과 동일)
# =====================================================================================

class PrunedConvFFN(ConvFFN):
    def __init__(self, in_features, hidden_features=None, out_features=None, **kwargs):
        super().__init__(in_features, hidden_features, out_features, **kwargs)

class PrunedAttention(Attention):
    def __init__(self, dim, heads, qk_dim, **kwargs):
        super().__init__(dim, heads, qk_dim, **kwargs)

class PrunedIASA(IASA):
    def __init__(self, dim, qk_dim, heads, group_size, **kwargs):
        super().__init__(dim, qk_dim, heads, group_size, **kwargs)

class PrunedTAB(TAB):
    def __init__(self, dim, qk_dim, mlp_dim, heads, group_size, **kwargs):
        super(TAB, self).__init__()
        self.n_iter = kwargs.get('n_iter', 3)
        self.ema_decay = kwargs.get('ema_decay', 0.999)
        self.num_tokens = kwargs.get('num_tokens', 8)
        self.norm = nn.LayerNorm(dim)
        self.conv1x1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.mlp = PreNorm(dim, PrunedConvFFN(dim, mlp_dim))
        self.iasa_attn = PrunedIASA(dim, qk_dim, heads, group_size)
        self.irca_attn = IRCA(dim,qk_dim,heads)
        self.register_buffer('means', torch.randn(self.num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        
class PrunedLRSA(LRSA):
    def __init__(self, dim, qk_dim, mlp_dim, heads, **kwargs):
        super(LRSA, self).__init__()
        self.layer = nn.ModuleList([
            PreNorm(dim, PrunedAttention(dim, heads, qk_dim)),
            PreNorm(dim, PrunedConvFFN(dim, mlp_dim))
        ])

# =====================================================================================
# 2. [변경됨] 구조적 프루닝을 지원하는 새로운 CATANet 클래스 정의
# =====================================================================================

class PrunedCATANetStructural(CATANet):
    def __init__(self, upscale: int, mlp_dims: list, head_counts: list, args):
        # super()는 CATANet의 기본 설정을 로드하기 위해 호출되지만, blocks는 여기서 직접 재정의합니다.
        super(CATANet, self).__init__()
        
        self.setting = CATANet.setting
        self.dim = self.setting['dim']
        self.original_block_num = self.setting['block_num'] # 원본 블록 수
        self.patch_size = self.setting['patch_size']
        self.qk_dim_orig = self.setting['qk_dim']
        self.heads_orig = self.setting['heads']
        self.upscale = upscale
        
        # 원본 CATANet에서 사용하는 설정값들을 가져옵니다.
        self.n_iters = args.get('n_iters', [5]*8)
        self.num_tokens = args.get('num_tokens', [16,32,64,128,16,32,64,128])
        self.group_size = args.get('group_size', [256,128,64,32,256,128,64,32])

        self.first_conv = nn.Conv2d(3, self.dim, 3, 1, 1)

        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        
        # [변경] 살아남은 블록의 원본 인덱스를 추적하기 위한 리스트
        self.source_block_indices = [] 

        q_head_dim = self.qk_dim_orig // self.heads_orig
        
        for i in range(self.original_block_num):
            current_heads = head_counts[i]
            current_mlp_dim = mlp_dims[i]
            
            # [변경] 헤드나 뉴런이 0이면 블록을 생성하지 않고 건너뜁니다.
            if current_heads == 0 or current_mlp_dim == 0:
                print(f"INFO: 레이어 {i}는 모든 헤드/뉴런이 프루닝되어 구조에서 제외됩니다.")
                continue

            # 이 블록이 살아남았으므로, 원본 인덱스를 기록합니다.
            self.source_block_indices.append(i)

            current_qk_dim = current_heads * q_head_dim
            tab_kwargs = {
                'n_iter': self.n_iters[i], 
                'num_tokens': self.num_tokens[i], 
                'group_size': self.group_size[i], 
                'ema_decay': 0.999
            }
            
            self.blocks.append(nn.ModuleList([
                PrunedTAB(self.dim, current_qk_dim, current_mlp_dim, current_heads, **tab_kwargs),
                PrunedLRSA(self.dim, current_qk_dim, current_mlp_dim, current_heads)
            ]))
            self.mid_convs.append(nn.Conv2d(self.dim, self.dim, 3, 1, 1))

        # 업스케일링 및 최종 레이어 부분 (기존과 거의 동일)
        if upscale == 4:
            self.upconv1 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale == 2 or upscale == 3:
            self.upconv = nn.Conv2d(self.dim, self.dim * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)
    
        self.last_conv = nn.Conv2d(self.dim, 3, 1, 1)
        if upscale != 1:
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.apply(self._init_weights)

    # [변경] forward_features를 반드시 오버라이드해야 합니다.
    def forward_features(self, x):
        # len(self.blocks)는 이제 원본 블록 수가 아닌, 살아남은 블록의 수가 됩니다.
        for i in range(len(self.blocks)):
            residual = x
      
            global_attn, local_attn = self.blocks[i]
            
            # 살아남은 블록 i가 원래 몇 번째 블록이었는지 확인합니다.
            source_idx = self.source_block_indices[i]
            # 해당 원본 블록의 patch_size를 사용합니다.
            patch_size = self.patch_size[source_idx]
            
            x = global_attn(x)
            x = local_attn(x, patch_size)
            
            x = residual + self.mid_convs[i](x)
        return x

# =====================================================================================
# 3. [변경됨] 구조적 프루닝을 위한 새로운 가중치 이식 로직
# =====================================================================================

def transfer_weights_structural(rebuilt_model, source_state_dict, masks, new_head_counts, new_mlp_dims):
    new_state_dict = rebuilt_model.state_dict()
    head_mask = masks['head_mask']
    neuron_mask = masks['neuron_mask']
    
    orig_heads = CATANet.setting['heads']
    orig_qk_dim = CATANet.setting['qk_dim']
    orig_v_dim = CATANet.setting['dim']
    
    q_head_dim = orig_qk_dim // orig_heads
    v_head_dim = orig_v_dim // orig_heads
    
    # [변경] 새로운 모델의 블록 인덱스와 원본 모델의 블록 인덱스 매핑
    # rebuilt_model에서 이 정보를 가져올 수 있습니다.
    source_block_indices = rebuilt_model.source_block_indices

    for name, new_param in new_state_dict.items():
        if 'blocks' not in name:
            # 블록 외부의 파라미터는 이름이 동일하므로 그대로 복사
            if name in source_state_dict:
                new_param.data.copy_(source_state_dict[name].data)
            else:
                print(f"경고: 소스에 '{name}' 가중치가 없습니다. 건너뜁니다.")
            continue
        
        # [변경] 가중치 이식 로직 수정
        parts = name.split('.')
        dest_layer_idx = int(parts[1])
        # 새로운 모델의 dest_layer_idx를 원본 모델의 source_layer_idx로 변환
        source_layer_idx = source_block_indices[dest_layer_idx]
        
        # 원본 state_dict에서 파라미터 이름을 재구성
        source_name_parts = parts[:]
        source_name_parts[1] = str(source_layer_idx)
        source_name = '.'.join(source_name_parts)

        if source_name not in source_state_dict:
            print(f"경고: 소스에 '{source_name}' 가중치가 없습니다. 건너뜁니다.")
            continue
            
        source_param = source_state_dict[source_name]
        
        # 마스크 인덱싱은 변환된 source_layer_idx를 사용해야 함
        current_head_mask = head_mask[source_layer_idx]
        current_neuron_mask = neuron_mask[source_layer_idx]

        if 'mlp.fn.fc1' in name:
            kept_neurons = current_neuron_mask.nonzero().squeeze(-1)
            if 'weight' in name:
                new_param.data.copy_(source_param.data[kept_neurons, :])
            elif 'bias' in name:
                new_param.data.copy_(source_param.data[kept_neurons])
        elif 'mlp.fn.fc2.weight' in name:
            kept_neurons = current_neuron_mask.nonzero().squeeze(-1)
            new_param.data.copy_(source_param.data[:, kept_neurons])
        elif 'mlp.fn.dwconv' in name:
            kept_neurons = current_neuron_mask.nonzero().squeeze(-1)
            if 'depthwise_conv.0.weight' in name or 'depthwise_conv.0.bias' in name:
                new_param.data.copy_(source_param.data[kept_neurons])
        
        elif 'iasa_attn' in name or 'layer.0.fn' in name:
            kept_heads = current_head_mask.nonzero().squeeze(-1)
            
            if 'to_q.weight' in name or 'to_k.weight' in name:
                new_w = torch.cat([source_param.data[h_idx * q_head_dim : (h_idx + 1) * q_head_dim, :] for h_idx in kept_heads], dim=0)
                new_param.data.copy_(new_w)
                
            elif 'to_v.weight' in name:
                new_w = torch.cat([source_param.data[h_idx * v_head_dim : (h_idx + 1) * v_head_dim, :] for h_idx in kept_heads], dim=0)
                new_param.data.copy_(new_w)

            elif 'proj.weight' in name:
                new_w = torch.cat([source_param.data[:, h_idx * v_head_dim : (h_idx + 1) * v_head_dim] for h_idx in kept_heads], dim=1)
                new_param.data.copy_(new_w)
            elif 'irca' in name:
                new_w = torch.cat([source_param.data[h_idx * q_head_dim : (h_idx + 1) * q_head_dim, :] for h_idx in kept_heads], dim=0)
                new_param.data.copy_(new_w)
            else:
                new_param.data.copy_(source_param.data)
        
        else:
            # mid_convs 등 블록 내의 다른 파라미터들
            new_param.data.copy_(source_param.data)
            
    rebuilt_model.load_state_dict(new_state_dict)

# =====================================================================================
# 4. 메인 실행 함수
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description="프루닝된 CATANet 모델을 물리적으로 더 작은 모델로 재구성합니다. (구조적 프루닝 지원)")
    parser.add_argument('--config', type=str, required=True, help='모델 설정이 담긴 원본 config 파일')
    parser.add_argument('--masks', type=str, required=True, help='프루닝 마스크가 저장된 .pth 파일')
    parser.add_argument('--source_weights', type=str, required=True, help='파인튜닝된 희소 모델의 가중치 파일')
    parser.add_argument('--save_path', type=str, required=True, help='재구성된 모델을 저장할 경로')
    
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    print("프루닝 마스크와 소스 가중치를 로드합니다...")
    config_path = os.path.join(base_dir, args.config) if not os.path.isabs(args.config) else args.config
    masks_path = os.path.join(base_dir, args.masks) if not os.path.isabs(args.masks) else args.masks
    source_weights_path = os.path.join(base_dir, args.source_weights) if not os.path.isabs(args.source_weights) else args.source_weights
    save_path = os.path.join(base_dir, args.save_path) if not os.path.isabs(args.save_path) else args.save_path

    masks = torch.load(masks_path, map_location='cpu')
    source_data = torch.load(source_weights_path, map_location='cpu')
    source_state_dict = source_data['params'] if 'params' in source_data else source_data

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    head_mask = masks['head_mask']
    neuron_mask = masks['neuron_mask']
    
    print("새로운 소형 아키텍처의 사양을 계산합니다...")
    new_head_counts = head_mask.sum(dim=-1).int().tolist()
    new_mlp_dims = neuron_mask.sum(dim=-1).int().tolist()

    print(f"  - 새로운 헤드 수 (레이어별): {new_head_counts}")
    print(f"  - 새로운 MLP 차원 (레이어별): {new_mlp_dims}")

    print("계산된 사양으로 새로운 소형 모델을 생성합니다...")
    # [변경] 새로운 클래스 사용
    rebuilt_model = PrunedCATANetStructural(
        upscale=config.get('scale', 2),
        mlp_dims=new_mlp_dims,
        head_counts=new_head_counts,
        args=config
    )
    print("소형 모델 생성 완료.")
    print(f"  - 원본 블록 수: {rebuilt_model.original_block_num}")
    print(f"  - 재구성된 블록 수: {len(rebuilt_model.blocks)}")

    print("가중치 이식을 시작합니다...")
    # [변경] 새로운 함수 사용
    transfer_weights_structural(rebuilt_model, source_state_dict, masks, new_head_counts, new_mlp_dims)
    print("가중치 이식 완료.")

    print(f"재구성된 모델을 '{save_path}'에 저장합니다...")
    torch.save({'params': rebuilt_model.state_dict()}, save_path)
    print("저장 완료!")
    
    print("\n--- 재구성된 모델 검증 ---")
    total_params = sum(p.numel() for p in rebuilt_model.parameters())
    print(f"재구성된 모델의 총 파라미터 수: {total_params / 1e6:.4f}M")
    print("이제 'python analysis/calculate_performance.py'를 이 새로운 모델에 대해 실행하여 FLOPs 감소를 확인할 수 있습니다.")


if __name__ == '__main__':
    main()
