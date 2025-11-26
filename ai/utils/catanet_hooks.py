"""
[리팩토링 노트]
이 스크립트는 OPTIN 프레임워크의 기존 `_hooks.py`를 대체하는 파일입니다.
`CATANet` 아키텍처와 호환되도록 특별히 리팩토링되었습니다.

[원본(HuggingFace/ViT 용)과의 주요 차이점]
1.  모델 구조 탐색:
    -   원본 hook은 `model.encoder.layer`와 같은 모델 구조를 가정했습니다.
    -   이 버전은 `CATANet`의 구조에 맞춰져, `model.blocks` 및
      `model.mid_convs`를 통해 블록에 접근합니다. 여기서 `model`은
      `net_g` 네트워크 자체를 의미합니다.

2.  Forward Pass 시그니처:
    -   원본 `forwardPass`는 모델이 딕셔너리 언패킹 인자(예: `model(**batch)`)를
      받는다고 가정했으며, 이는 HuggingFace 모델에서 일반적입니다.
    -   이 버전의 `forwardPass`는 `CATANet`의 `forward` 메서드가 기대하는
      단일 텐서 `model(input_tensor)`를 받습니다.
      
3.  Hook 위치:
    -   중간 출력을 캡처하기 위한 hook 위치가 ViT 스타일의 출력 레이어에서
      `CATANet`의 주요 블록 출력을 나타내는 `mid_convs` 모듈로 변경되었습니다.
    -   뉴런 가지치기(neuron pruning)를 위한 pre-hook은 이제 `CATANet`의
      `ConvFFN` 블록 내의 활성화 함수를 대상으로 합니다.
"""
import torch
from collections import OrderedDict

class CATANetModelHooking:
    """
    `CATANet` 아키텍처를 위해 특별히 설계된 커스텀 모델 후킹 클래스입니다.
    PyTorch hook을 사용하여 마스크를 적용하고 가지치기 분석을 위한
    중간 레이어 출력을 캡처합니다.
    """
    def __init__(self, args, model=None, maskProps=None):
        super(CATANetModelHooking, self).__init__()
        
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.maskProps = maskProps
        
        self.layer_outputs = []
        self.registered_hooks = []
        self.head_mask = None
        
        if self.maskProps:
            self.apply_mask_and_hooks()

    def apply_mask_and_hooks(self):
        """maskProps에 따라 마스크를 적용하고 forward hook을 등록합니다."""
        # 중간 출력을 캡처하기 위한 메인 forward hook을 등록합니다.
        self.register_forward_hooks()
        
        # 마스크(예: 뉴런 가지치기)를 적용하기 위한 pre-hook을 등록합니다.
        if self.maskProps.get('state') == 'neuron':
            self.register_neuron_pre_hook()
        elif self.maskProps.get('state') == 'head':
            # FIX: Implement the head pruning hook registration
            self.register_head_pre_hook()
            
    def purge_hooks(self):
        """등록된 모든 hook을 제거합니다."""
        for handle in self.registered_hooks:
            handle.remove()
        self.registered_hooks = []
        self.layer_outputs = []

    def _record_layer_output_hook(self, name):
        """모듈의 출력을 기록하기 위한 hook 함수입니다."""
        def hook(module, input, output):
            self.layer_outputs.append(output)
        return hook

    def _get_head_mask_hook(self, mask, num_heads):
        """
        주어진 마스크를 사용하여 어텐션 헤드의 출력을 0으로 만드는 pre-hook을 생성합니다.
        """
        def hook(_, inputs):
            # 입력은 튜플이고, 텐서는 첫 번째 요소입니다.
            # 입력 텐서 형태: (batch_size, seq_len, total_dim)
            x = inputs[0]
            
            # 헤드 차원을 노출시키기 위해 재구성합니다.
            head_dim = x.shape[-1] // num_heads
            # (b, n, h*d) -> (b, n, h, d)
            x = x.view(x.shape[0], x.shape[1], num_heads, head_dim)
            
            # 마스크를 적용합니다.
            # 마스크 형태: (h,) -> 브로드캐스팅을 위해 (1, 1, h, 1)로 재구성
            mask_reshaped = mask.view(1, 1, num_heads, 1).to(x.device)
            x = x * mask_reshaped
            
            # 원래 형태로 재구성합니다.
            # (b, n, h, d) -> (b, n, h*d)
            x = x.view(x.shape[0], x.shape[1], -1)
            
            return (x,)
        return hook

    def register_head_pre_hook(self):
        """
        헤드 프루닝 분석을 위해 어텐션 모듈에 pre-forward hook을 등록합니다.
        """
        layer_idx = self.maskProps["layer"]
        # head_mask는 모든 레이어에 대한 것이며, 현재 레이어의 마스크를 가져옵니다.
        head_mask_for_layer = self.maskProps["mask"][layer_idx] # Shape: (num_heads,)

        # IASA와 LRSA의 어텐션 모듈에 hook을 적용해야 합니다.
        try:
            attention_modules = [
                self.model.blocks[layer_idx][0].iasa_attn, # TAB의 IASA
                self.model.blocks[layer_idx][1].layer[0].fn  # LRSA의 Attention
            ]

            for attn_module in attention_modules:
                # 최종 프로젝션 레이어에 pre-hook을 등록합니다.
                target_module = attn_module.proj
                hook_handle = target_module.register_forward_pre_hook(
                    self._get_head_mask_hook(head_mask_for_layer, attn_module.heads)
                )
                self.registered_hooks.append(hook_handle)
        except (AttributeError, IndexError) as e:
            print(f"오류: 헤드 hook을 적용할 레이어 {layer_idx}의 모듈을 찾을 수 없습니다: {e}")

    def register_forward_hooks(self):
        """
        `CATANet`의 원하는 레이어에 forward hook을 등록하여 손실 계산을 위한
        출력을 캡처합니다.
        
        `CATANet`의 경우, 각 메인 블록의 출력인 'mid_convs'의 출력을 hook합니다.
        """
        # 대상 레이어는 `forward_features` 루프 내의 `mid_convs`입니다.
        # 이 모듈들을 hook합니다.
        for i, block_module in enumerate(self.model.mid_convs):
            hook_handle = block_module.register_forward_hook(self._record_layer_output_hook(f"block_{i}"))
            self.registered_hooks.append(hook_handle)

    def register_neuron_pre_hook(self):
        """
        뉴런 마스크를 적용하기 위해 pre-forward hook을 등록합니다.
        `ConvFFN` 블록의 활성화 함수에서 특정 뉴런으로의 입력을 0으로 만듭니다.
        """
        layer_idx = self.maskProps["layer"]
        mask = self.maskProps["mask"] # 이 마스크는 MLP의 뉴런을 위한 것입니다.
        
        # 대상 모듈은 `ConvFFN`의 첫 번째 선형 레이어(fc1) 뒤의 활성화 함수입니다.
        # `catanet_arch.py`에서의 이름은 `self.act`입니다.
        # 경로: blocks -> TAB (idx 0) -> mlp -> fn (ConvFFN) -> act
        try:
            target_module = self.model.blocks[layer_idx][0].mlp.fn.act
            
            # 이 hook은 활성화 함수에 대한 입력을 마스크와 곱합니다.
            def hook(_, inputs):
                # 입력은 튜플이고, 텐서는 첫 번째 요소입니다.
                # 마스크는 입력 텐서의 모양에 맞게 브로드캐스팅되어야 합니다.
                return (inputs[0] * mask.to(self.device),)
            
            hook_handle = target_module.register_forward_pre_hook(hook)
            self.registered_hooks.append(hook_handle)
        except IndexError:
            print(f"오류: 뉴런 hook을 적용할 레이어 {layer_idx}의 모듈을 찾을 수 없습니다.")
            
    def forwardPass(self, input_tensor):
        """
        hook이 적용된 모델로 forward pass를 수행합니다.
        
        인자(Args):
            input_tensor (Tensor): 모델에 대한 입력 이미지 텐서.
        
        반환(Returns):
            tuple: 다음을 포함하는 튜플입니다:
                - output_image (Tensor): 모델의 최종 출력.
                - layer_wise_output (list): 캡처된 중간 출력들의 리스트.
        """
        self.layer_outputs = [] # 이전 출력 지우기
        with torch.no_grad():
            # `CATANet`의 forward 메서드는 딕셔너리가 아닌 단일 텐서를 기대합니다.
            output_image = self.model(input_tensor)
        
        # 캡처된 출력은 hook에 의해 채워진 self.layer_outputs에 있습니다.
        return output_image, self.layer_outputs