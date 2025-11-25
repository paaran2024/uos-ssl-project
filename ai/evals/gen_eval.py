import torch
from copy import deepcopy
from tqdm import tqdm
from os import path as osp

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import tensor2img

def _apply_pruning_in_memory(model, pruning_params):
    """
    Applies pruning by zeroing out weights in the model's state_dict.
    This function is based on the logic from `run_catanet_pruning.py`.

    NOTE: This implementation currently only handles FFN neuron pruning, as head
    pruning application logic was not present in the reference scripts.
    """
    pruned_state_dict = deepcopy(model.state_dict())
    
    neuron_mask = pruning_params.get('neuron_mask')
    if neuron_mask is None:
        print("Warning: Neuron mask not found in pruning_params. Returning original model state.")
        return pruned_state_dict

    # Traverse each block of the model and apply the neuron mask
    for i in range(model.net_g.block_num):
        # Find the indices of neurons to be pruned (where mask is 0)
        pruned_neurons = (neuron_mask[i] == 0).nonzero(as_tuple=True)[0]

        if len(pruned_neurons) == 0:
            continue  # No neurons to prune in this layer

        # Zero out the weights for the pruned neurons in the ConvFFN layers
        # fc1: Linear(dim, mlp_dim) -> Zero out rows corresponding to output neurons
        fc1_weight_name = f'net_g.blocks.{i}.0.mlp.fn.fc1.weight'
        fc1_bias_name = f'net_g.blocks.{i}.0.mlp.fn.fc1.bias'
        
        if fc1_weight_name in pruned_state_dict:
            pruned_state_dict[fc1_weight_name][pruned_neurons, :] = 0
            pruned_state_dict[fc1_bias_name][pruned_neurons] = 0

        # fc2: Linear(mlp_dim, dim) -> Zero out columns corresponding to input neurons
        fc2_weight_name = f'net_g.blocks.{i}.0.mlp.fn.fc2.weight'
        if fc2_weight_name in pruned_state_dict:
            pruned_state_dict[fc2_weight_name][:, pruned_neurons] = 0
            
    return pruned_state_dict

def _evaluate_performance(model, dataloader, args):
    """
    Helper function to evaluate a model on a given dataloader.
    Calculates average PSNR and SSIM.
    This logic is adapted from BasicSR's `nondist_validation`.
    """
    total_psnr = 0
    total_ssim = 0
    num_images = 0
    
    # Get metric options from config
    # Set default values if not specified
    val_opts = args.datasets.get('val', {})
    metric_opts = val_opts.get('metrics', {
        'psnr': {'crop_border': 4, 'test_y_channel': True},
        'ssim': {'crop_border': 4, 'test_y_channel': True}
    })
    psnr_opt = metric_opts.get('psnr', {'crop_border': 4, 'test_y_channel': True})
    ssim_opt = metric_opts.get('ssim', {'crop_border': 4, 'test_y_channel': True})

    pbar = tqdm(total=len(dataloader), unit='image', desc='Evaluating')
    
    for val_data in dataloader:
        lq_tensor = val_data['lq'].to('cuda' if torch.cuda.is_available() else 'cpu')
        gt_tensor = val_data['gt'].to('cuda' if torch.cuda.is_available() else 'cpu')

        # Inference
        with torch.no_grad():
            output_tensor = model(lq_tensor)

        # Convert tensors to numpy images (range [0, 255])
        sr_img = tensor2img(output_tensor.detach().cpu())
        gt_img = tensor2img(gt_tensor.detach().cpu())
        
        # Calculate PSNR and SSIM
        total_psnr += calculate_psnr(sr_img, gt_img, **psnr_opt)
        total_ssim += calculate_ssim(sr_img, gt_img, **ssim_opt)
        
        num_images += 1
        pbar.update(1)

    pbar.close()

    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    
    return {'psnr': avg_psnr, 'ssim': avg_ssim}

def evalModel(args, model, train_dataset, val_dataset, pruningParams, prunedProps):
    """
    Evaluates the performance of the baseline and pruned models.

    Args:
        args: Configuration arguments.
        model: The PyTorch model to be evaluated.
        val_dataset: The validation dataloader.
        pruningParams: Dictionary containing pruning masks ('neuron_mask', 'head_mask').

    Returns:
        tuple: A tuple containing:
            - baseline_performance (dict): Metrics for the original model.
            - final_performance (dict): Metrics for the pruned model.
    """
    # The 'model' passed here is actually the CATANetModel, which contains net_g
    # We need to evaluate net_g.
    
    print("--- Evaluating Baseline Model ---")
    # For evaluation, we only need the generator network `net_g`
    baseline_model = model.net_g
    baseline_performance = _evaluate_performance(baseline_model, val_dataset, args)
    print(f"Baseline Performance: PSNR={baseline_performance['psnr']:.4f}, SSIM={baseline_performance['ssim']:.4f}")

    print("\n--- Evaluating Pruned Model ---")
    # Create a deep copy to avoid modifying the original model
    pruned_model = deepcopy(baseline_model)
    
    # Get the pruned state_dict
    pruned_state_dict = _apply_pruning_in_memory(pruned_model, pruningParams)
    
    # Load the pruned weights into the copied model
    pruned_model.load_state_dict(pruned_state_dict)
    
    # Evaluate the pruned model
    final_performance = _evaluate_performance(pruned_model, val_dataset, args)
    print(f"Pruned Performance: PSNR={final_performance['psnr']:.4f}, SSIM={final_performance['ssim']:.4f}")
    
    return baseline_performance, final_performance