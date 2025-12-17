"""
finetune.py

Fine-tunes Qwen2.5-0.5B via LoRA.
"""

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type
import torch.nn.functional as F
import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NUM_TOKENS
)
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.models import load, load_vla

from prismatic.models.projectors import AlignProjector

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class FinetuneConfig:
    # fmt: off
    config_file_path: str = "openvla/openvla-7b"     # Path to necessary config files of LA-Adapter
    vlm_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)
    use_minivlm: bool = False                        # 
    resum_vla_path: str = "openvla/openvla-7b"       # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for training 
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input
    phase1_path: str = "None"

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0.1                       # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100000             # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200000                          # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    pad_future_actions: bool = False                 # If True, pads future actions with last action instead of dropping tail frames
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # LoRA
    use_lora: bool = False                           # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = False         # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Full Finetune
    use_fz: bool = False                             # If True, uses LoRA fine-tuning

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # revision version
    use_pro_version: bool = True                             # the version number
    phase: str = "Training"
    # fmt: on

    # ========== [SPATIAL FORCING] Configuration Parameters ==========
    vggt_path: str = "/home/icrlab02/vla_ws/VLA-Adapter/pretrained_models/vggt/model.pt"  # Path to VGGT model for spatial alignment
    use_spatial_forcing: bool = False                # If True, enables VGGT spatial feature alignment
    align_loss_type: str = "cosine"                  # Loss function for alignment (cosine)
    align_loss_coeff: float = 0.5                    # Coefficient for alignment loss (legacy, used when use_dual_align=False)
    vla_layers_align: int = -1                       # Which layer of VLA hidden state to align (0-indexed)
    vggt_layers_align: int = -1                      # Which layer of VGGT hidden state to align (0-indexed, legacy for use_dual_align=False)
    pooling_func: str = "bilinear"                   # Resize method for VGGT to VLA pixels (bilinear, nearest)
    use_vlm_norm: bool = False                       # Whether to use VLM normalization for vision embeddings
    use_vggt_pe: bool = True                         # Whether to use position embedding for VGGT
    share_projector: bool = True                     # If True, share projector across views; False uses per-view projectors (applies to both modes)
    
    # ========== [SPATIAL FORCING] Dual Alignment Configuration ==========
    use_dual_align: bool = False                     # If True, uses separate Frame/Global alignment (new); False uses legacy concat alignment
    frame_align_layer: int = -1                      # VGGT layer for frame-level alignment (-1 = last layer)
    global_align_layer: int = -1                     # VGGT layer for global-level alignment (-1 = last layer)
    frame_loss_coeff: float = 0.5                    # Coefficient for frame-level alignment loss
    global_loss_coeff: float = 0.5                   # Coefficient for global-level alignment loss

    # ========== [ALOHA DELTA] Delta Action Configuration ==========
    use_aloha_delta: bool = False                    # If True, convert absolute actions to delta for ALOHA datasets


def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict



def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.config_file_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.config_file_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_fz:
            run_id += f"+frozen+dropout-{cfg.lora_dropout}"
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id



def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)



def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)



def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    print(f"# trainable params in {name}: {num_params}")



def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.resum_vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
        print('loaded!!!!!!!!!')

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)



def run_forward_pass(
    vla,
    action_head,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_proprio,
    use_film,
    num_patches,
    compute_diffusion_l1=False,
    use_pro_version=True,
    vggt=None,
    align_projector=None,
    frame_align_projector=None,
    global_align_projector=None,
    preprocess_normed_images=None,
    custom_pooling=None,
    processor=None,
    cfg=None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps (int): Number of diffusion steps (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=use_film,
            )

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:,1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)

        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=current_action_mask
            )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, 
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=current_action_mask
            )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=next_actions_mask
            )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, 
            predicted_token_ids, 
            ground_truth_token_ids, 
            mask=next_actions_mask
            )
        
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )
        
    # Compute metrics for continuous action representations (L1 regression)
    else:
        # Get last layer hidden states
        multi_layer_hidden_states = []
        
        for item in output.hidden_states[0:]:
            # last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)
            text_hidden_states = item[:, num_patches:-1]
            # Get hidden states for action portion of response
            batch_size = batch["input_ids"].shape[0]
            # actions_hidden_states = text_hidden_states[:, -1, :].reshape(batch_size, 1, -1).to(torch.bfloat16)
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(batch_size, 1,NUM_TOKENS, -1).to(torch.bfloat16)
            task_latten_states = item[:, :num_patches].reshape(batch_size, 1, num_patches , -1)
            all_hidden_states = torch.cat((task_latten_states, actions_hidden_states),2)
            multi_layer_hidden_states.append(all_hidden_states)
        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim = 1)

        predicted_actions = action_head.module.predict_action(
            multi_layer_hidden_states,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            phase=cfg.phase,
            )

        action_loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)
        
        # ========== [SPATIAL FORCING] Second Layer Selection: Alignment Loss ==========
        align_loss = torch.tensor(0.0, device=device_id)
        frame_align_loss = torch.tensor(0.0, device=device_id)
        global_align_loss = torch.tensor(0.0, device=device_id)
        
        # Check if spatial forcing is enabled (supports both legacy and dual-align modes)
        sf_enabled = cfg is not None and cfg.use_spatial_forcing and vggt is not None
        sf_legacy_enabled = sf_enabled and not cfg.use_dual_align and align_projector is not None
        sf_dual_enabled = sf_enabled and cfg.use_dual_align and frame_align_projector is not None and global_align_projector is not None
        
        if sf_legacy_enabled or sf_dual_enabled:
            # Extract single layer from VLA for alignment (different from the 24 layers used above)
            vla_hidden_for_align = output.hidden_states[cfg.vla_layers_align]  # [bs, seq_len, hidden_dim]
            
            # Extract vision tokens (skip BOS token and language tokens)
            vision_length = num_patches
            boi_ids = 1  # Begin of image token index
            vision_hidden = vla_hidden_for_align[:, boi_ids:boi_ids + vision_length, :].clone()
            
            # Extract VGGT features
            vggt.eval()
            unnorm_imgs = preprocess_normed_images(
                batch['pixel_values'], 
                processor.image_processor,
                num_images_in_input=cfg.num_images_in_input
            ).to(device_id)
            
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                vggt_output = vggt(unnorm_imgs)
            
            patch_start_idx = vggt_output["patch_start_idx"]
            original_img = vggt_output["images"]
            H, W = original_img.shape[-2:]
            patch_h, patch_w = H // vggt.patch_size, W // vggt.patch_size
            
            if sf_dual_enabled:
                # ========== [DUAL ALIGN] Separate Frame and Global Alignment ==========
                # Extract frame features from specified layer (first 1024 dims)
                frame_layer_features = vggt_output["features"][cfg.frame_align_layer]  # [B, N, P+special, 2048]
                frame_hidden = frame_layer_features[:, :, patch_start_idx:, :1024]  # [B, N, P, 1024]
                
                # Extract global features from specified layer (last 1024 dims)
                global_layer_features = vggt_output["features"][cfg.global_align_layer]  # [B, N, P+special, 2048]
                global_hidden = global_layer_features[:, :, patch_start_idx:, 1024:]  # [B, N, P, 1024]
                
                # Spatial resampling for frame features
                frame_hidden_pooled = custom_pooling(
                    frame_hidden, 
                    (patch_h, patch_w), 
                    (H, W), 
                    vision_hidden,
                    cfg.pooling_func,
                    cfg.use_vggt_pe
                )  # [B, N*P_vla, 1024]
                
                # Spatial resampling for global features
                global_hidden_pooled = custom_pooling(
                    global_hidden, 
                    (patch_h, patch_w), 
                    (H, W), 
                    vision_hidden,
                    cfg.pooling_func,
                    cfg.use_vggt_pe
                )  # [B, N*P_vla, 1024]
                
                # Compute frame alignment loss
                frame_align_loss = frame_align_projector(vision_hidden, frame_hidden_pooled)
                
                # Compute global alignment loss
                global_align_loss = global_align_projector(vision_hidden, global_hidden_pooled)
                
                # Combined alignment loss
                align_loss = cfg.frame_loss_coeff * frame_align_loss + cfg.global_loss_coeff * global_align_loss
                
            else:
                # ========== [LEGACY] Concat Alignment ==========
                agg_vggt_hidden = vggt_output["features"][cfg.vggt_layers_align]
                vggt_hidden = agg_vggt_hidden[:, :, patch_start_idx:, :]  # [B, N, P, 2048]
                
                # Spatial resampling to match VLA's vision token layout
                vggt_hidden = custom_pooling(
                    vggt_hidden, 
                    (patch_h, patch_w), 
                    (H, W), 
                    vision_hidden,
                    cfg.pooling_func,
                    cfg.use_vggt_pe
                )  # [B, N*P_vla, 2048]
                
                # Reshape for per-view projector if share_projector=False
                if not cfg.share_projector:
                    B = vision_hidden.shape[0]
                    N = cfg.num_images_in_input
                    P = vision_hidden.shape[1] // N
                    D_llm = vision_hidden.shape[2]
                    D_vggt = vggt_hidden.shape[2]
                    # Reshape to [B, N, P, D] for per-view processing
                    vision_hidden = vision_hidden.reshape(B, N, P, D_llm)
                    vggt_hidden = vggt_hidden.reshape(B, N, P, D_vggt)
                
                # Compute alignment loss
                align_loss = align_projector(vision_hidden, vggt_hidden)
        # ========== [END SPATIAL FORCING ALIGNMENT] ==========
        
        # Compute total loss
        if cfg is not None and cfg.use_spatial_forcing:
            if cfg.use_dual_align:
                loss = action_loss + align_loss  # align_loss already weighted by frame/global coeffs
            else:
                loss = action_loss + cfg.align_loss_coeff * align_loss
        else:
            loss = action_loss
        
        # Update metrics
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "action_loss": action_loss.item(),  # ========== [SPATIAL FORCING] Add action_loss to metrics ==========
                "align_loss": align_loss.item() if (cfg and cfg.use_spatial_forcing) else 0.0,  # ========== [SPATIAL FORCING] Add align_loss to metrics ==========
            }
        )
        
        # Add dual-align specific metrics
        if cfg is not None and cfg.use_spatial_forcing and cfg.use_dual_align:
            metrics.update(
                {
                    "frame_align_loss": frame_align_loss.item(),
                    "global_align_loss": global_align_loss.item(),
                }
            )

        # Get detailed L1 losses for logging
        should_log_l1_loss = use_l1_regression
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            if compute_diffusion_l1:
                print('curr: ',curr_action_l1_loss.item())
                # print('next: ',next_actions_l1_loss.item())

            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return loss, metrics



def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics



def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)



def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
    new_state_dict,
    align_projector=None,
    frame_align_projector=None,
    global_align_projector=None,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)

        if cfg.use_fz:
            vla.module.save_pretrained(checkpoint_dir) # directly save checkpoint without lora
        else:
            vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if cfg.use_l1_regression and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        # ========== [SPATIAL FORCING] Save AlignProjector(s) ==========
        if cfg.use_spatial_forcing:
            if cfg.use_dual_align:
                # Dual-align mode: save frame and global projectors separately
                if frame_align_projector is not None:
                    torch.save(get_module(frame_align_projector).state_dict(), checkpoint_dir / f"frame_align_projector--{checkpoint_name_suffix}")
                    print(f"[SPATIAL FORCING] Saved FrameAlignProjector checkpoint")
                if global_align_projector is not None:
                    torch.save(get_module(global_align_projector).state_dict(), checkpoint_dir / f"global_align_projector--{checkpoint_name_suffix}")
                    print(f"[SPATIAL FORCING] Saved GlobalAlignProjector checkpoint")
            else:
                # Legacy mode: save single concat projector
                if align_projector is not None:
                    torch.save(get_module(align_projector).state_dict(), checkpoint_dir / f"align_projector--{checkpoint_name_suffix}")
                    print(f"[SPATIAL FORCING] Saved AlignProjector checkpoint")
        # ========== [END SPATIAL FORCING CHECKPOINT] ==========


        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

    # Wait for model components to be saved
    dist.barrier()

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    if cfg.use_lora and cfg.merge_lora_during_training:
        if cfg.use_minivlm:
            config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
            base_vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)  # Create a new model with configuration, the parameters are randomly initialized
            # print(new_state_dict['action_queries.weight'])
            new_state_dict['action_queries.weight'] = vla.state_dict()['module.base_model.model.action_queries.weight'].cpu()
            missing_keys, unexpected_keys = base_vla.load_state_dict(new_state_dict, strict=False)
            
        else:
            base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, trust_remote_code=False
        )


        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")
        
        # Wait for merged model to be saved
        dist.barrier()



def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
    vggt=None,
    align_projector=None,
    frame_align_projector=None,
    global_align_projector=None,
    preprocess_normed_images=None,
    custom_pooling=None,
    processor=None,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                compute_diffusion_l1=True,
                use_pro_version=cfg.use_pro_version,
                cfg=cfg,
                vggt=vggt,
                align_projector=align_projector,
                frame_align_projector=frame_align_projector,
                global_align_projector=global_align_projector,
                preprocess_normed_images=preprocess_normed_images,
                custom_pooling=custom_pooling,
                processor=processor,
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """ 

    global RAW_STATE_DICT

    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.config_file_path = cfg.config_file_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.config_file_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Initialize wandb logging
    if distributed_state.is_main_process:
        wandb.init(project=cfg.wandb_project, name=f"ft+{run_id}", mode="offline")

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect

    if model_is_on_hf_hub(cfg.config_file_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.config_file_path)
        # Overwrite VLA path
        cfg.config_file_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.config_file_path)
        check_model_logic_mismatch(cfg.config_file_path)

    # Wait for model files to be synced
    dist.barrier()

    # Load processor and VLA
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(cfg.config_file_path, trust_remote_code=True)

    if cfg.use_minivlm:
        hf_token = ''
        if 'prism-qwen25-extra-dinosiglip-224px-0_5b' in cfg.vlm_path:
            
            vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
        else:
            vlm = load_vla(
                cfg.vlm_path,
                hf_token=hf_token,
                load_for_training=True,
                )
        config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device_id)  # Create a new model with configuration, the parameters are randomly initialized
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.shape}")
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
            ]

        def rename_state_dict_keys(state_dict, replace_map):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict
        
        old_state_dict = vlm.state_dict()
        RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)
    
        missing_keys, unexpected_keys = vla.load_state_dict(RAW_STATE_DICT, strict=False)
        del old_state_dict

    else:
        RAW_STATE_DICT ={}
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=False,
            ).to(device_id)

    # Set number of images in VLA input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # vla.set_version(cfg.version)

    # ========== [SPATIAL FORCING] Initialize VGGT and AlignProjector ==========
    vggt = None
    align_projector = None
    frame_align_projector = None
    global_align_projector = None
    preprocess_normed_images = None
    custom_pooling = None
    
    if cfg.use_spatial_forcing:
        # Import SF-related modules from local spatial_forcing_components
        # Add to sys.path temporarily for imports only
        import sys
        from pathlib import Path
        _sf_components_path = str(Path(__file__).parent.parent / "spatial_forcing_components")
        if _sf_components_path not in sys.path:
            sys.path.insert(0, _sf_components_path)
        
        try:
            if distributed_state.is_main_process:
                print(f"[SPATIAL FORCING] Initializing Spatial Forcing components...")
                if cfg.use_dual_align:
                    print(f"[SPATIAL FORCING] Mode: DUAL ALIGN (Frame + Global)")
                    print(f"[SPATIAL FORCING] Config: VLA_layer={cfg.vla_layers_align}, Frame_layer={cfg.frame_align_layer}, Global_layer={cfg.global_align_layer}")
                    print(f"[SPATIAL FORCING] Loss coeffs: frame={cfg.frame_loss_coeff}, global={cfg.global_loss_coeff}")
                    print(f"[SPATIAL FORCING] Share frame projector: {cfg.share_frame_projector}")
                else:
                    print(f"[SPATIAL FORCING] Mode: LEGACY (concat alignment)")
                    print(f"[SPATIAL FORCING] Config: VLA_layer={cfg.vla_layers_align}, VGGT_layer={cfg.vggt_layers_align}, loss_coeff={cfg.align_loss_coeff}")
            
            # Import from local spatial_forcing_components
            from prismatic.models.projectors import AlignProjector, FrameAlignProjector, GlobalAlignProjector
            from vggt.models.vggt import VGGT
            from vggt.utils.load_fn import preprocess_normed_images
            from prismatic.util.pooling_utils import custom_pooling
        finally:
            # Remove from sys.path immediately after import
            if _sf_components_path in sys.path:
                sys.path.remove(_sf_components_path)
        
        # Initialize VGGT model
        vggt = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
            feature_only=True,
        )
        vggt.load_state_dict(torch.load(cfg.vggt_path, map_location='cpu'), strict=False)
        vggt = vggt.to(device_id)
        vggt.eval()
        for param in vggt.parameters():
            param.requires_grad = False
        
        if distributed_state.is_main_process:
            print(f"[SPATIAL FORCING] VGGT loaded from: {cfg.vggt_path}")
        
        # Initialize AlignProjector(s)
        llm_hidden_size = get_module(vla).llm_dim
        vggt_hidden_size = 1024  # VGGT feature dimension (frame or global, each 1024)
        
        if cfg.use_dual_align:
            # ========== [DUAL ALIGN] Initialize Frame and Global Projectors ==========
            frame_align_projector = FrameAlignProjector(
                llm_dim=llm_hidden_size,
                vggt_dim=vggt_hidden_size,
                align_loss_type=cfg.align_loss_type,
                use_vlm_norm=cfg.use_vlm_norm,
                num_views=cfg.num_images_in_input,
                share_projector=cfg.share_projector,
            ).to(device_id)
            
            global_align_projector = GlobalAlignProjector(
                llm_dim=llm_hidden_size,
                vggt_dim=vggt_hidden_size,
                align_loss_type=cfg.align_loss_type,
                use_vlm_norm=cfg.use_vlm_norm,
            ).to(device_id)
            
            if distributed_state.num_processes > 1:
                frame_align_projector = DDP(frame_align_projector, device_ids=[device_id])
                global_align_projector = DDP(global_align_projector, device_ids=[device_id])
            
            if distributed_state.is_main_process:
                print(f"[SPATIAL FORCING] FrameAlignProjector initialized: {llm_hidden_size} -> {vggt_hidden_size}, share_projector={cfg.share_projector}")
                print(f"[SPATIAL FORCING] GlobalAlignProjector initialized: {llm_hidden_size} -> {vggt_hidden_size}")
        else:
            # ========== [LEGACY] Initialize single concat AlignProjector ==========
            align_projector = AlignProjector(
                llm_dim=llm_hidden_size,
                vggt_dim=vggt_hidden_size,
                align_loss_type=cfg.align_loss_type,
                use_vlm_norm=cfg.use_vlm_norm,
                num_views=cfg.num_images_in_input,
                share_projector=cfg.share_projector,
            ).to(device_id)
            
            if distributed_state.num_processes > 1:
                align_projector = DDP(align_projector, device_ids=[device_id])
            
            if distributed_state.is_main_process:
                print(f"[SPATIAL FORCING] AlignProjector initialized: {llm_hidden_size} -> {2 * vggt_hidden_size}, share_projector={cfg.share_projector}")
        
        if distributed_state.is_main_process:
            print(f"[SPATIAL FORCING] Alignment loss type: {cfg.align_loss_type}")
    # ========== [END SPATIAL FORCING] ==========

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha= 2 * cfg.lora_rank,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        vla.print_trainable_parameters()

    else:
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True

    # FiLM setup
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.config_file_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
            to_bf16=True,
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        action_head = init_module(
        L1RegressionActionHead,
        "action_head",
        cfg,
        device_id,
        {
            "input_dim": vla.module.llm_dim, 
            "hidden_dim": vla.module.llm_dim, 
            "action_dim": ACTION_DIM,
            "use_pro_version": cfg.use_pro_version,
            },
        to_bf16=True,
        )

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]

    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    
    # ========== [SPATIAL FORCING] Add AlignProjector parameters to optimizer ==========
    if cfg.use_spatial_forcing:
        if cfg.use_dual_align:
            # Dual-align mode: add both frame and global projectors
            if frame_align_projector is not None:
                trainable_params += [param for param in frame_align_projector.parameters() if param.requires_grad]
            if global_align_projector is not None:
                trainable_params += [param for param in global_align_projector.parameters() if param.requires_grad]
        else:
            # Legacy mode: add single concat projector
            if align_projector is not None:
                trainable_params += [param for param in align_projector.parameters() if param.requires_grad]
    # ========== [END SPATIAL FORCING] ==========
    
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    # 1. MultiStepLR
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )
    # 2. CosineAnnealingLR
    # scheduler = CosineAnnealingLR(
    #         optimizer,
    #         T_max=cfg.num_steps_before_decay, 
    #         eta_min=0.0001,          
    #         )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    # )
    # ---

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1

    # Create training and optional validation datasets
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        use_minivlm=cfg.use_minivlm,
        use_aloha_delta=cfg.use_aloha_delta,
        )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        pad_future_actions=cfg.pad_future_actions,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            pad_future_actions=cfg.pad_future_actions,
            train=False,
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    print('Len of dataloader: ', len(dataloader))
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Start training
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # Compute training metrics and loss
            compute_diffusion_l1 = (cfg.use_l1_regression and batch_idx % cfg.diffusion_sample_freq == 0) or (cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0)
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                compute_diffusion_l1=compute_diffusion_l1,
                use_pro_version=cfg.use_pro_version,
                cfg=cfg,
                vggt=vggt if cfg.use_spatial_forcing else None,
                align_projector=align_projector if (cfg.use_spatial_forcing and not cfg.use_dual_align) else None,
                frame_align_projector=frame_align_projector if (cfg.use_spatial_forcing and cfg.use_dual_align) else None,
                global_align_projector=global_align_projector if (cfg.use_spatial_forcing and cfg.use_dual_align) else None,
                preprocess_normed_images=preprocess_normed_images if cfg.use_spatial_forcing else None,
                custom_pooling=custom_pooling if cfg.use_spatial_forcing else None,
                processor=processor,
            )

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Store recent train metrics
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)

            # Push Metrics to W&B (every wandb_log_freq gradient steps)
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)

            # [If applicable] Linearly warm up learning rate from 10% to 100% of original
            if cfg.lr_warmup_steps > 0:
                lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                # Log the learning rate
                # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                wandb.log(
                    {
                        "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                    },
                    step=log_step,
                )

            # Optimizer and LR scheduler step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()

            # Save model checkpoint: either keep latest checkpoint only or all checkpoints
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    noisy_action_projector=None,
                    action_head=action_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                    new_state_dict=RAW_STATE_DICT,
                    align_projector=align_projector if (cfg.use_spatial_forcing and not cfg.use_dual_align) else None,
                    frame_align_projector=frame_align_projector if (cfg.use_spatial_forcing and cfg.use_dual_align) else None,
                    global_align_projector=global_align_projector if (cfg.use_spatial_forcing and cfg.use_dual_align) else None,
                )

            # Test model on validation set
            if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                    vggt=vggt if cfg.use_spatial_forcing else None,
                    align_projector=align_projector if (cfg.use_spatial_forcing and not cfg.use_dual_align) else None,
                    frame_align_projector=frame_align_projector if (cfg.use_spatial_forcing and cfg.use_dual_align) else None,
                    global_align_projector=global_align_projector if (cfg.use_spatial_forcing and cfg.use_dual_align) else None,
                    preprocess_normed_images=preprocess_normed_images if cfg.use_spatial_forcing else None,
                    custom_pooling=custom_pooling if cfg.use_spatial_forcing else None,
                    processor=processor,
                )
                # Set model back to training mode after validation
                vla.train()

            # Stop training when max_steps is reached
            if log_step == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


def get_module(model):
    """
    Get the underlying module from a model, unwrapping DDP if necessary.
    
    Args:
        model: PyTorch model, possibly wrapped with DDP
        
    Returns:
        The underlying module without DDP wrapper
    """
    return model.module if hasattr(model, 'module') else model

if __name__ == "__main__":
    finetune()
