"""
grad_cam_utils.py

Grad-CAM visualization utilities for VLA models.
Supports visualization at multiple layers: SigLIP, Projector, and Qwen2 transformer blocks.

Visualization Modes:
- 'activation': Fast, uses activation magnitude (EigenCAM-like), works with inference_mode
- 'grad_cam': True Grad-CAM with gradient computation, slower but more accurate per action dimension

Supported Configurations:
- LIBERO: action_dim=7 (single arm)
- ALOHA: action_dim=14 (bimanual)
"""

import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from functools import partial
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class CAMMode(Enum):
    """CAM visualization mode."""
    ACTIVATION = "activation"  # Fast, activation-based (EigenCAM-like)
    GRAD_CAM = "grad_cam"      # True Grad-CAM with gradients


class DatasetType(Enum):
    """Dataset type for action dimension configuration."""
    LIBERO = "libero"   # 7-dim actions (single arm)
    ALOHA = "aloha"     # 14-dim actions (bimanual)
    CUSTOM = "custom"   # User-specified


@dataclass
class CAMConfig:
    """Configuration for CAM visualization."""
    enabled: bool = False
    mode: CAMMode = CAMMode.ACTIVATION
    dataset_type: DatasetType = DatasetType.LIBERO
    action_dim: int = 7
    num_views: int = 2
    output_dir: str = "./grad_cam_outputs"
    save_individual: bool = True
    save_grid: bool = True
    visualize_every_n_steps: int = 1  # Visualize every N steps (1 = every step)
    # Layer selection
    siglip_layer_indices: List[int] = None
    projector_layer_indices: List[int] = None
    qwen_layer_indices: List[int] = None
    # Action dimension selection
    action_dims_to_visualize: List[int] = None
    include_l2_norm: bool = True
    
    def __post_init__(self):
        if self.siglip_layer_indices is None:
            self.siglip_layer_indices = [-1]
        if self.projector_layer_indices is None:
            self.projector_layer_indices = [-1]
        if self.qwen_layer_indices is None:
            self.qwen_layer_indices = [0, 6, 12, 18, 23]
        if self.action_dims_to_visualize is None:
            if self.dataset_type == DatasetType.LIBERO:
                # LIBERO: 7 dims - visualize position (0,1,2) and gripper (6)
                self.action_dims_to_visualize = [0, 1, 2, 6]
            elif self.dataset_type == DatasetType.ALOHA:
                # ALOHA: 14 dims - visualize positions of both arms
                self.action_dims_to_visualize = [0, 1, 2, 6, 7, 8, 9, 13]
            else:
                self.action_dims_to_visualize = list(range(min(7, self.action_dim)))
        # Set action_dim based on dataset type
        if self.dataset_type == DatasetType.LIBERO:
            self.action_dim = 7
        elif self.dataset_type == DatasetType.ALOHA:
            self.action_dim = 14


class ActivationAndGradientCapture:
    """Captures activations and gradients from specified layers using hooks."""
    
    def __init__(self, model: nn.Module, target_layers: List[nn.Module], capture_gradients: bool = False):
        self.model = model
        self.target_layers = target_layers
        self.capture_gradients = capture_gradients
        self.activations = {}
        self.gradients = {}
        self.handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        for i, layer in enumerate(self.target_layers):
            # Forward hook for activations
            handle = layer.register_forward_hook(
                partial(self._save_activation, layer_idx=i)
            )
            self.handles.append(handle)
            
            # Backward hook for gradients (only if needed)
            if self.capture_gradients:
                handle = layer.register_full_backward_hook(
                    partial(self._save_gradient, layer_idx=i)
                )
                self.handles.append(handle)
    
    def _save_activation(self, module, input, output, layer_idx):
        if isinstance(output, tuple):
            output = output[0]
        # Keep tensor for gradient computation if needed
        if self.capture_gradients:
            self.activations[layer_idx] = output
        else:
            self.activations[layer_idx] = output.detach().clone()
    
    def _save_gradient(self, module, grad_input, grad_output, layer_idx):
        if isinstance(grad_output, tuple):
            grad_output = grad_output[0]
        if grad_output is not None:
            self.gradients[layer_idx] = grad_output.detach().clone()
    
    def clear(self):
        self.activations.clear()
        self.gradients.clear()
    
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def show_cam_on_image(
    img: np.ndarray, 
    mask: np.ndarray, 
    use_rgb: bool = True,
    alpha: float = 0.5
) -> np.ndarray:
    """Overlay CAM heatmap on image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255.0
    
    if img.max() > 1.0:
        img = img.astype(np.float32) / 255.0
    
    cam_image = alpha * heatmap + (1 - alpha) * img
    cam_image = cam_image / cam_image.max()
    return np.uint8(255 * cam_image)


class VLAGradCAMVisualizer:
    """
    Grad-CAM visualizer for VLA models.
    
    Supports two modes:
    - ACTIVATION: Fast, uses activation magnitude, compatible with inference_mode
    - GRAD_CAM: True Grad-CAM, requires gradient computation, per-action-dimension visualization
    """
    
    def __init__(
        self,
        vla_model,
        processor,
        config: CAMConfig,
        action_head=None,
        proprio_projector=None,
        image_size: int = 384,
        patch_size: int = 14,
        device: str = "cuda",
    ):
        self.vla = vla_model
        self.processor = processor
        self.config = config
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.device = device
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_per_side = image_size // patch_size  # 27 for 384/14
        self.num_patches = self.patches_per_side ** 2     # 729
        
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_model_components()
        self._step_counter = 0
    
    def _setup_model_components(self):
        """Setup references to model components."""
        vla = self.vla.module if hasattr(self.vla, 'module') else self.vla
        
        # SigLIP vision backbone
        self.siglip_blocks = None
        if hasattr(vla, 'vision_backbone') and hasattr(vla.vision_backbone, 'featurizer'):
            self.siglip_blocks = vla.vision_backbone.featurizer.blocks
        
        # Projector - handle both structures
        self.projector_layers = None
        if hasattr(vla, 'projector'):
            proj = vla.projector
            # HuggingFace model: fc1, fc2, fc3
            if hasattr(proj, 'fc3'):
                self.projector_layers = [proj.fc1, proj.fc2, proj.fc3]
            # Original model: projector.projector (Sequential)
            elif hasattr(proj, 'projector'):
                self.projector_layers = list(proj.projector)
        
        # Qwen LLM layers - handle both structures
        self.qwen_layers = None
        # HuggingFace model: language_model.model.layers
        if hasattr(vla, 'language_model') and hasattr(vla.language_model, 'model'):
            if hasattr(vla.language_model.model, 'layers'):
                self.qwen_layers = vla.language_model.model.layers
        # Original model: llm_backbone.llm.model.layers
        elif hasattr(vla, 'llm_backbone') and hasattr(vla.llm_backbone, 'llm'):
            self.qwen_layers = vla.llm_backbone.llm.model.layers
    
    def _get_target_layers(self) -> Tuple[List[nn.Module], List[str]]:
        """Get list of target layers and their names."""
        layers, names = [], []
        
        if self.siglip_blocks is not None:
            for idx in self.config.siglip_layer_indices:
                actual_idx = idx if idx >= 0 else len(self.siglip_blocks) + idx
                layers.append(self.siglip_blocks[actual_idx].norm1)
                names.append(f"siglip_b{actual_idx}")
        
        if self.projector_layers is not None:
            for idx in self.config.projector_layer_indices:
                actual_idx = idx if idx >= 0 else len(self.projector_layers) + idx
                if actual_idx < len(self.projector_layers):
                    layers.append(self.projector_layers[actual_idx])
                    names.append(f"proj_l{actual_idx}")
        
        if self.qwen_layers is not None:
            for idx in self.config.qwen_layer_indices:
                actual_idx = idx if idx >= 0 else len(self.qwen_layers) + idx
                layers.append(self.qwen_layers[actual_idx].input_layernorm)
                names.append(f"qwen_l{actual_idx}")
        
        return layers, names
    
    def _compute_activation_cam(self, activation: torch.Tensor, h: int, w: int) -> np.ndarray:
        """Compute CAM from activation using magnitude (EigenCAM-like)."""
        try:
            # Detach and clone to avoid gradient issues
            if activation.requires_grad:
                activation = activation.detach()
            
            if len(activation.shape) == 3:
                B, N, C = activation.shape
            else:
                return np.zeros((h, w))
            
            target_n = h * w
            if N > target_n:
                activation = activation[:, :target_n, :]
            elif N < target_n:
                pad = torch.zeros(B, target_n - N, C, device=activation.device, dtype=activation.dtype)
                activation = torch.cat([activation, pad], dim=1)
            
            act_spatial = activation.reshape(B, h, w, C)
            
            # Use L2 norm across channels as importance
            cam = torch.norm(act_spatial[0].float(), dim=-1).cpu().numpy()
            
            # Normalize
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros((h, w))
            
            return cam
        except Exception as e:
            print(f"Error in _compute_activation_cam: {e}")
            return np.zeros((h, w))
    
    def _compute_grad_cam(
        self, 
        activation: torch.Tensor, 
        gradient: torch.Tensor, 
        h: int, 
        w: int
    ) -> np.ndarray:
        """Compute true Grad-CAM from activation and gradient."""
        try:
            # Detach tensors to avoid gradient issues
            if activation.requires_grad:
                activation = activation.detach()
            if gradient.requires_grad:
                gradient = gradient.detach()
            
            if len(activation.shape) == 3:
                B, N, C = activation.shape
            else:
                return np.zeros((h, w))
            
            target_n = h * w
            if N > target_n:
                activation = activation[:, :target_n, :]
                gradient = gradient[:, :target_n, :]
            elif N < target_n:
                pad = torch.zeros(B, target_n - N, C, device=activation.device, dtype=activation.dtype)
                activation = torch.cat([activation, pad], dim=1)
                gradient = torch.cat([gradient, pad], dim=1)
            
            # Reshape to spatial: [B, H, W, C]
            act_spatial = activation.reshape(B, h, w, C)
            grad_spatial = gradient.reshape(B, h, w, C)
            
            # Global average pooling of gradients -> weights [B, 1, 1, C]
            weights = grad_spatial.mean(dim=(1, 2), keepdim=True)
            
            # Weighted sum
            cam = (weights * act_spatial).sum(dim=-1)  # [B, H, W]
            
            # ReLU
            cam = F.relu(cam)
            
            cam = cam[0].float().cpu().numpy()
            
            # Normalize
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = np.zeros((h, w))
            
            return cam
        except Exception as e:
            print(f"Error in _compute_grad_cam: {e}")
            return np.zeros((h, w))
    
    def _prepare_inputs(
        self,
        original_images: List[np.ndarray],
        task_label: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare model inputs from original images."""
        try:
            from PIL import Image
            
            # Convert numpy images to PIL
            pil_images = []
            for img in original_images:
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                pil_images.append(Image.fromarray(img).convert("RGB"))
            
            # Build prompt
            prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
            
            # Process primary image
            inputs = self.processor(prompt, pil_images[0]).to(self.device, dtype=torch.bfloat16)
            
            # Process additional images if any
            if len(pil_images) > 1:
                additional_inputs = [
                    self.processor(prompt, img).to(self.device, dtype=torch.bfloat16)
                    for img in pil_images[1:]
                ]
                primary_pv = inputs["pixel_values"]
                additional_pvs = [inp["pixel_values"] for inp in additional_inputs]
                inputs["pixel_values"] = torch.cat([primary_pv] + additional_pvs, dim=1)
            
            return inputs
        except Exception as e:
            print(f"Error preparing inputs for CAM: {e}")
            return None
    
    def _extract_view_activation(
        self, 
        activation: torch.Tensor, 
        view_idx: int, 
        is_qwen: bool
    ) -> torch.Tensor:
        """Extract view-specific activations."""
        if is_qwen:
            B, seq_len, C = activation.shape
            start = 1 + view_idx * self.num_patches
            end = start + self.num_patches
            if end <= seq_len:
                return activation[:, start:end, :]
            else:
                return activation[:, :self.num_patches, :]
        else:
            total = activation.shape[1]
            ppv = self.num_patches
            if total >= ppv * self.config.num_views:
                start = view_idx * ppv
                return activation[:, start:start+ppv, :]
            else:
                return activation
    
    def generate_cam(
        self,
        inputs: Optional[Dict[str, torch.Tensor]],
        original_images: List[np.ndarray],
        proprio: Optional[np.ndarray] = None,
        unnorm_key: str = None,
        step_idx: int = 0,
        episode_idx: int = 0,
        task_label: str = "",
    ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """
        Generate CAM visualizations for a single step.
        
        Args:
            inputs: Processed model inputs (if None, will be computed from original_images)
            original_images: List of original RGB images [H, W, 3] (0-255)
            proprio: Proprioception data if used
            unnorm_key: Key for action unnormalization
            step_idx: Current step index
            episode_idx: Current episode index
            task_label: Task description
            
        Returns:
            Tuple of (results_dict, action) where action is the predicted action
        """
        if not self.config.enabled:
            return {}, None
        
        # Check if we should skip this step
        self._step_counter += 1
        if self._step_counter % self.config.visualize_every_n_steps != 0:
            return {}, None
        
        results = {}
        action = None
        
        # If inputs not provided, compute from original_images
        if inputs is None:
            inputs = self._prepare_inputs(original_images, task_label)
            if inputs is None:
                return {}, None
        
        # Prepare images for overlay
        overlay_size = self.patches_per_side * 16  # 432 for 27*16
        images_overlay = []
        for img in original_images:
            img_f = img.astype(np.float32)
            if img_f.max() > 1.0:
                img_f /= 255.0
            img_r = cv2.resize(img_f, (overlay_size, overlay_size))
            images_overlay.append(img_r)
        
        # Get target layers
        target_layers, layer_names = self._get_target_layers()
        if not target_layers:
            print("Warning: No target layers found for CAM visualization")
            return results, None
        
        # Choose mode
        use_grad_cam = (self.config.mode == CAMMode.GRAD_CAM)
        
        # Setup capture
        capture = ActivationAndGradientCapture(
            self.vla, 
            target_layers, 
            capture_gradients=use_grad_cam
        )
        
        try:
            if use_grad_cam:
                # True Grad-CAM mode - need gradients
                results, action = self._generate_grad_cam_mode(
                    inputs, images_overlay, proprio, unnorm_key,
                    step_idx, episode_idx, capture, layer_names, overlay_size
                )
            else:
                # Activation mode - fast, works with inference_mode
                results, action = self._generate_activation_mode(
                    inputs, images_overlay, proprio, unnorm_key,
                    step_idx, episode_idx, capture, layer_names, overlay_size
                )
            
            # Create summary grid
            if self.config.save_grid and results:
                self.create_summary_grid(
                    results, original_images, step_idx, episode_idx
                )
        
        except Exception as e:
            print(f"Error in CAM generation: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            capture.remove_hooks()
        
        return results, action
    
    def _generate_activation_mode(
        self,
        inputs: Dict[str, torch.Tensor],
        images_overlay: List[np.ndarray],
        proprio: Optional[np.ndarray],
        unnorm_key: str,
        step_idx: int,
        episode_idx: int,
        capture: ActivationAndGradientCapture,
        layer_names: List[str],
        overlay_size: int,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Generate CAM using activation magnitude (fast mode)."""
        results = {}
        
        with torch.inference_mode():
            if self.action_head is None:
                action, _ = self.vla.predict_action(
                    **inputs, unnorm_key=unnorm_key, do_sample=False
                )
            else:
                action, _ = self.vla.predict_action(
                    **inputs,
                    unnorm_key=unnorm_key,
                    do_sample=False,
                    proprio=proprio,
                    proprio_projector=self.proprio_projector,
                    action_head=self.action_head,
                )
        
        # Process activations
        for layer_idx, layer_name in enumerate(layer_names):
            if layer_idx not in capture.activations:
                continue
            
            act = capture.activations[layer_idx]
            is_qwen = layer_name.startswith('qwen')
            
            for view_idx in range(self.config.num_views):
                view_act = self._extract_view_activation(act, view_idx, is_qwen)
                cam = self._compute_activation_cam(
                    view_act, self.patches_per_side, self.patches_per_side
                )
                cam_resized = cv2.resize(cam, (overlay_size, overlay_size))
                vis = show_cam_on_image(images_overlay[view_idx], cam_resized)
                
                key = f"{layer_name}_v{view_idx}_act"
                results[key] = vis
                
                if self.config.save_individual:
                    save_path = self.output_dir / f"ep{episode_idx:04d}_s{step_idx:04d}_{key}.png"
                    cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        return results, action
    
    def _generate_grad_cam_mode(
        self,
        inputs: Dict[str, torch.Tensor],
        images_overlay: List[np.ndarray],
        proprio: Optional[np.ndarray],
        unnorm_key: str,
        step_idx: int,
        episode_idx: int,
        capture: ActivationAndGradientCapture,
        layer_names: List[str],
        overlay_size: int,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Generate true Grad-CAM with gradients (per action dimension)."""
        results = {}
        action_result = None
        
        # Build targets: action dimensions + L2 norm
        targets = []
        target_names = []
        
        for dim in self.config.action_dims_to_visualize:
            if dim < self.config.action_dim:
                targets.append(('dim', dim))
                target_names.append(f'd{dim}')
        
        if self.config.include_l2_norm:
            targets.append(('l2', None))
            target_names.append('l2')
        
        # Need gradient computation - NOT inference_mode
        pixel_values = inputs["pixel_values"].clone().detach().requires_grad_(True)
        inputs_grad = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        inputs_grad["pixel_values"] = pixel_values
        
        for target_type, target_val in targets:
            target_name = f'd{target_val}' if target_type == 'dim' else 'l2'
            
            capture.clear()
            
            # Clear GPU memory before each iteration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with torch.enable_grad():
                if self.action_head is None:
                    action, _ = self.vla.predict_action(
                        **inputs_grad, unnorm_key=unnorm_key, do_sample=False
                    )
                else:
                    action, _ = self.vla.predict_action(
                        **inputs_grad,
                        unnorm_key=unnorm_key,
                        do_sample=False,
                        proprio=proprio,
                        proprio_projector=self.proprio_projector,
                        action_head=self.action_head,
                    )
                
                action_result = action
                
                # Convert to tensor for gradient
                if isinstance(action, np.ndarray):
                    action_tensor = torch.from_numpy(action).to(self.device).float()
                    action_tensor.requires_grad_(True)
                else:
                    action_tensor = action.float()
                
                # Compute target scalar
                if target_type == 'dim':
                    if len(action_tensor.shape) == 1:
                        target_scalar = action_tensor[target_val]
                    elif len(action_tensor.shape) == 2:
                        target_scalar = action_tensor[0, target_val]
                    else:
                        target_scalar = action_tensor.flatten()[target_val]
                else:
                    target_scalar = torch.norm(action_tensor.flatten(), p=2)
                
                # Backward pass
                self.vla.zero_grad()
                if pixel_values.grad is not None:
                    pixel_values.grad.zero_()
                
                try:
                    target_scalar.backward(retain_graph=True)
                except Exception as e:
                    print(f"Backward failed for {target_name}: {e}")
                    continue
            
            # Generate CAM for each layer and view
            for layer_idx, layer_name in enumerate(layer_names):
                if layer_idx not in capture.activations:
                    continue
                if layer_idx not in capture.gradients:
                    # Fall back to activation-based CAM
                    act = capture.activations[layer_idx]
                    is_qwen = layer_name.startswith('qwen')
                    
                    for view_idx in range(self.config.num_views):
                        view_act = self._extract_view_activation(act, view_idx, is_qwen)
                        cam = self._compute_activation_cam(
                            view_act, self.patches_per_side, self.patches_per_side
                        )
                        cam_resized = cv2.resize(cam, (overlay_size, overlay_size))
                        vis = show_cam_on_image(images_overlay[view_idx], cam_resized)
                        
                        key = f"{layer_name}_v{view_idx}_{target_name}"
                        results[key] = vis
                        
                        if self.config.save_individual:
                            save_path = self.output_dir / f"ep{episode_idx:04d}_s{step_idx:04d}_{key}.png"
                            cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    continue
                
                act = capture.activations[layer_idx]
                grad = capture.gradients[layer_idx]
                is_qwen = layer_name.startswith('qwen')
                
                for view_idx in range(self.config.num_views):
                    view_act = self._extract_view_activation(act, view_idx, is_qwen)
                    view_grad = self._extract_view_activation(grad, view_idx, is_qwen)
                    
                    cam = self._compute_grad_cam(
                        view_act, view_grad, 
                        self.patches_per_side, self.patches_per_side
                    )
                    cam_resized = cv2.resize(cam, (overlay_size, overlay_size))
                    vis = show_cam_on_image(images_overlay[view_idx], cam_resized)
                    
                    key = f"{layer_name}_v{view_idx}_{target_name}"
                    results[key] = vis
                    
                    if self.config.save_individual:
                        save_path = self.output_dir / f"ep{episode_idx:04d}_s{step_idx:04d}_{key}.png"
                        cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            
            # Clear GPU memory after each target to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results, action_result
    
    def create_summary_grid(
        self,
        results: Dict[str, np.ndarray],
        original_images: List[np.ndarray],
        step_idx: int = 0,
        episode_idx: int = 0,
    ) -> Optional[np.ndarray]:
        """Create summary grid of all visualizations."""
        if not results:
            return None
        
        first_img = next(iter(results.values()))
        img_h, img_w = first_img.shape[:2]
        
        # Parse structure
        keys = list(results.keys())
        # Extract layer names (everything before _v{idx})
        layers = sorted(set(k.rsplit('_v', 1)[0] for k in keys))
        # Extract target names
        targets = sorted(set(k.split('_')[-1] for k in keys))
        
        # Resize originals
        orig_resized = []
        for img in original_images:
            img_f = img.astype(np.float32)
            if img_f.max() > 1.0:
                img_f /= 255.0
            img_r = cv2.resize(img_f, (img_w, img_h))
            orig_resized.append((img_r * 255).astype(np.uint8))
        
        # Grid layout: columns = views * targets, rows = layers + 1 (originals)
        n_cols = self.config.num_views * len(targets)
        n_rows = len(layers) + 1
        
        if n_cols == 0 or n_rows == 0:
            return None
        
        grid = np.zeros((n_rows * img_h, n_cols * img_w, 3), dtype=np.uint8)
        
        # First row: original images (repeated for each target)
        for v in range(self.config.num_views):
            for t_idx in range(len(targets)):
                col = v * len(targets) + t_idx
                if col < n_cols and v < len(orig_resized):
                    grid[0:img_h, col*img_w:(col+1)*img_w] = orig_resized[v]
        
        # CAM rows
        for r_idx, layer in enumerate(layers, 1):
            for v in range(self.config.num_views):
                for t_idx, target in enumerate(targets):
                    col = v * len(targets) + t_idx
                    key = f"{layer}_v{v}_{target}"
                    if key in results:
                        y_start = r_idx * img_h
                        x_start = col * img_w
                        grid[y_start:y_start+img_h, x_start:x_start+img_w] = results[key]
        
        save_path = self.output_dir / f"ep{episode_idx:04d}_s{step_idx:04d}_grid.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        
        return grid
    
    def reset_step_counter(self):
        """Reset the step counter (call at episode start)."""
        self._step_counter = 0


def create_cam_config(
    enabled: bool = False,
    mode: str = "activation",
    dataset_type: str = "libero",
    output_dir: str = "./grad_cam_outputs",
    visualize_every_n_steps: int = 1,
    **kwargs
) -> CAMConfig:
    """
    Create a CAMConfig from simple arguments.
    
    Args:
        enabled: Whether to enable CAM visualization
        mode: "activation" (fast) or "grad_cam" (accurate per action dim)
        dataset_type: "libero" (7-dim) or "aloha" (14-dim)
        output_dir: Output directory for visualizations
        visualize_every_n_steps: Generate CAM every N steps
        **kwargs: Additional config options
        
    Returns:
        Configured CAMConfig instance
    """
    mode_enum = CAMMode.GRAD_CAM if mode.lower() == "grad_cam" else CAMMode.ACTIVATION
    
    if dataset_type.lower() == "aloha":
        dtype = DatasetType.ALOHA
    elif dataset_type.lower() == "libero":
        dtype = DatasetType.LIBERO
    else:
        dtype = DatasetType.CUSTOM
    
    return CAMConfig(
        enabled=enabled,
        mode=mode_enum,
        dataset_type=dtype,
        output_dir=output_dir,
        visualize_every_n_steps=visualize_every_n_steps,
        **kwargs
    )


def setup_grad_cam_visualizer(
    vla_model,
    processor,
    config: Union[CAMConfig, Dict, None] = None,
    action_head=None,
    proprio_projector=None,
    device: str = "cuda",
    # Simple config options (used if config is None)
    enabled: bool = False,
    mode: str = "activation",
    dataset_type: str = "libero",
    num_views: int = 2,
    output_dir: str = "./grad_cam_outputs",
) -> VLAGradCAMVisualizer:
    """
    Create a VLAGradCAMVisualizer with configuration.
    
    Args:
        vla_model: The VLA model
        processor: Model processor
        config: CAMConfig instance, dict, or None to use simple options
        action_head: Optional action head
        proprio_projector: Optional proprioception projector
        device: Device to use
        enabled: Enable visualization (if config is None)
        mode: "activation" or "grad_cam" (if config is None)
        dataset_type: "libero" or "aloha" (if config is None)
        num_views: Number of camera views (if config is None)
        output_dir: Output directory (if config is None)
        
    Returns:
        Configured VLAGradCAMVisualizer instance
    """
    if config is None:
        config = create_cam_config(
            enabled=enabled,
            mode=mode,
            dataset_type=dataset_type,
            output_dir=output_dir,
            num_views=num_views,
        )
    elif isinstance(config, dict):
        config = create_cam_config(**config)
    
    return VLAGradCAMVisualizer(
        vla_model=vla_model,
        processor=processor,
        config=config,
        action_head=action_head,
        proprio_projector=proprio_projector,
        device=device,
    )
