"""
CAM Visualization for VLA Models - Direct activation-based approach.
Uses pytorch-grad-cam's visualization utilities but custom activation capture.
"""

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

# Add pytorch-grad-cam to path for visualization utilities only
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pytorch-grad-cam"))
from pytorch_grad_cam.utils.image import show_cam_on_image


class CAMMethod(Enum):
    EIGENCAM = "eigencam"
    GRADCAM = "gradcam"


@dataclass
class CAMConfig:
    enabled: bool = True
    method: CAMMethod = CAMMethod.EIGENCAM
    output_dir: str = "./grad_cam_outputs"
    frames_per_episode: int = 5  # Fixed number of frames to sample per episode
    siglip_layer_indices: List[int] = field(default_factory=lambda: [-1])
    qwen_layer_indices: List[int] = field(default_factory=lambda: [5, 10, 20, 24])  # Qwen 2.5-0.5B has 24 layers
    num_views: int = 2
    image_size: int = 224  # Will be auto-detected from model if possible
    patch_size: int = 14
    save_individual: bool = True
    save_grid: bool = True
    overlay_alpha: float = 0.5
    
    @property
    def patches_per_side(self):
        return self.image_size // self.patch_size


class ActivationCapture:
    """Simple hook-based activation capture."""
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def register(self, layer, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        self.hooks.append(layer.register_forward_hook(hook))
    
    def clear(self):
        self.activations.clear()
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class VLAGradCAMVisualizer:
    """CAM visualizer using direct activation capture."""
    
    def __init__(self, vla_model, processor, config, action_head=None, proprio_projector=None):
        self.config = config
        self.processor = processor
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        vla = vla_model.module if hasattr(vla_model, 'module') else vla_model
        self.vla = vla
        self.device = next(vla_model.parameters()).device
        
        # Auto-detect image size from model if possible
        self._detect_image_size(vla)
        
        # Setup activation capture
        self.capture = ActivationCapture()
        self._setup_hooks()
        
        # Frame sampling state
        self._episode_frames = []  # Collect frames during episode
        self._current_episode_len = 0
        
        print(f"[CAM] Initialized with {len(self.capture.hooks)} hooks")
        print(f"[CAM] Image size: {self.config.image_size}, patches_per_side: {self.config.patches_per_side}")
        print(f"[CAM] Frames per episode: {self.config.frames_per_episode}")
    
    def _detect_image_size(self, vla):
        """Auto-detect image size from model configuration."""
        detected_size = None
        
        # Try to get from vision_backbone
        if hasattr(vla, 'vision_backbone'):
            vb = vla.vision_backbone
            if hasattr(vb, 'default_image_size'):
                detected_size = vb.default_image_size
            elif hasattr(vb, 'featurizer') and hasattr(vb.featurizer, 'patch_embed'):
                # Try to infer from patch_embed
                pe = vb.featurizer.patch_embed
                if hasattr(pe, 'img_size'):
                    img_size = pe.img_size
                    detected_size = img_size[0] if isinstance(img_size, (tuple, list)) else img_size
        
        # Try to get from model config
        if detected_size is None and hasattr(vla, 'config'):
            cfg = vla.config
            if hasattr(cfg, 'image_sizes') and cfg.image_sizes:
                detected_size = cfg.image_sizes[0]
        
        if detected_size and detected_size != self.config.image_size:
            print(f"[CAM] Auto-detected image size: {detected_size} (config was {self.config.image_size})")
            self.config.image_size = detected_size
    
    def _setup_hooks(self):
        """Register hooks on target layers (SigLIP and Qwen)."""
        # SigLIP vision backbone hooks
        if hasattr(self.vla, 'vision_backbone') and hasattr(self.vla.vision_backbone, 'featurizer'):
            blocks = self.vla.vision_backbone.featurizer.blocks
            for idx in self.config.siglip_layer_indices:
                actual_idx = idx if idx >= 0 else len(blocks) + idx
                if 0 <= actual_idx < len(blocks):
                    layer = blocks[actual_idx].norm1
                    self.capture.register(layer, f"siglip_b{actual_idx}")
                    print(f"[CAM] Registered hook on siglip_b{actual_idx}")
        
        # Qwen LLM backbone hooks - try multiple paths
        qwen_layers = None
        
        # Path 1: HuggingFace model structure (language_model.model.layers)
        if hasattr(self.vla, 'language_model'):
            lm = self.vla.language_model
            if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
                qwen_layers = lm.model.layers
        
        # Path 2: Original model structure (llm_backbone.llm.model.layers)
        if qwen_layers is None and hasattr(self.vla, 'llm_backbone'):
            if hasattr(self.vla.llm_backbone, 'llm'):
                llm = self.vla.llm_backbone.llm
                if hasattr(llm, 'model') and hasattr(llm.model, 'layers'):
                    qwen_layers = llm.model.layers
        
        if qwen_layers is not None:
            num_layers = len(qwen_layers)
            print(f"[CAM] Qwen has {num_layers} layers")
            for idx in self.config.qwen_layer_indices:
                # Convert 1-indexed to 0-indexed (user says layer 5 = index 4)
                actual_idx = idx - 1 if idx > 0 else idx
                if 0 <= actual_idx < num_layers:
                    # Hook on input_layernorm of decoder layer
                    layer = qwen_layers[actual_idx].input_layernorm
                    self.capture.register(layer, f"qwen_L{idx}")
                    print(f"[CAM] Registered hook on qwen_L{idx}")
    
    def _compute_cam(self, activation):
        """Compute CAM from activation using PCA (EigenCAM style).
        
        Args:
            activation: [B, N, C] where N = num_patches (should be h*w)
        Returns:
            cam: [h, w] numpy array normalized to [0, 1]
        """
        if len(activation.shape) != 3:
            return np.zeros((self.config.patches_per_side, self.config.patches_per_side))
        
        B, N, C = activation.shape
        
        # Dynamically compute spatial dimensions from token count
        side = int(np.sqrt(N))
        if side * side != N:
            # Not a perfect square, use config
            side = self.config.patches_per_side
            target_n = side * side
            if N > target_n:
                activation = activation[:, :target_n, :]
            elif N < target_n:
                pad = torch.zeros(B, target_n - N, C, device=activation.device, dtype=activation.dtype)
                activation = torch.cat([activation, pad], dim=1)
        
        # Reshape to spatial: [N, C] -> [side, side, C]
        # SigLIP uses row-first ordering (standard)
        act_spatial = activation[0].reshape(side, side, C).float().cpu().numpy()
        
        # PCA: use first principal component
        act_flat = act_spatial.reshape(-1, C)
        act_centered = act_flat - act_flat.mean(axis=0)
        
        # SVD to get first PC
        try:
            U, S, Vt = np.linalg.svd(act_centered, full_matrices=False)
            cam = U[:, 0].reshape(side, side)
            cam = np.abs(cam)  # Take absolute value
        except:
            # Fallback to L2 norm
            cam = np.linalg.norm(act_spatial, axis=-1)
        
        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros((side, side))
        
        return cam.astype(np.float32)
    
    def _extract_view(self, activation, view_idx):
        """Extract activation for a specific view."""
        # activation: [B, N, C]
        B, N, C = activation.shape
        tokens_per_view = self.config.patches_per_side ** 2
        
        start = view_idx * tokens_per_view
        end = start + tokens_per_view
        
        if end <= N:
            return activation[:, start:end, :]
        elif start < N:
            return activation[:, start:, :]
        else:
            return activation[:, :tokens_per_view, :]
    
    def collect_frame(self, pixel_values, original_images, step_idx, task_name=None):
        """Collect frame data for later CAM generation."""
        self._episode_frames.append({
            'pixel_values': pixel_values.cpu().clone(),
            'original_images': [img.copy() for img in original_images],
            'step_idx': step_idx,
            'task_name': task_name
        })
    
    def generate_episode_cam(self, total_steps, episode_idx=0):
        """Generate CAM for sampled frames at end of episode."""
        if not self._episode_frames:
            return
        
        # Sample frames uniformly
        n_frames = len(self._episode_frames)
        n_samples = min(self.config.frames_per_episode, n_frames)
        
        if n_samples >= n_frames:
            sample_indices = list(range(n_frames))
        else:
            step = n_frames / n_samples
            sample_indices = [int(i * step) for i in range(n_samples)]
        
        sampled_steps = [self._episode_frames[i]['step_idx'] for i in sample_indices]
        print(f"[CAM] Sampling {len(sample_indices)} frames from {n_frames}: steps {sampled_steps}")
        
        for idx in sample_indices:
            frame_data = self._episode_frames[idx]
            self._generate_single_cam(
                pixel_values=frame_data['pixel_values'],
                original_images=frame_data['original_images'],
                step_idx=frame_data['step_idx'],
                episode_idx=episode_idx,
                task_name=frame_data['task_name']
            )
        
        self._episode_frames.clear()
    
    def generate_cam(self, pixel_values, original_images, step_idx=0, episode_idx=0, task_name=None, **kwargs):
        """Collect frame for CAM generation (called every step)."""
        self.collect_frame(pixel_values, original_images, step_idx, task_name)
        return {}
    
    def _generate_single_cam(self, pixel_values, original_images, step_idx=0, episode_idx=0, task_name=None):
        """Generate CAM visualizations for a single frame."""
        # Create task-specific subdirectory
        if task_name:
            safe_task_name = task_name.replace(' ', '_').replace('/', '_')[:50]
            task_dir = self.output_dir / safe_task_name
            task_dir.mkdir(parents=True, exist_ok=True)
        else:
            task_dir = self.output_dir
        
        results = {}
        self.capture.clear()
        
        # Run forward pass to capture activations from both SigLIP and Qwen
        try:
            with torch.no_grad():
                if len(pixel_values.shape) == 4:
                    B, C, H, W = pixel_values.shape
                    if C == 6:
                        view0 = pixel_values[:, :3, :, :]
                        view1 = pixel_values[:, 3:6, :, :]
                        pv = torch.cat([view0, view1], dim=0)
                    else:
                        pv = pixel_values
                else:
                    pv = pixel_values.reshape(-1, *pixel_values.shape[-3:])
                
                pv = pv.to(self.device, dtype=torch.bfloat16)
                
                # Run SigLIP vision backbone
                vision_outputs = self.vla.vision_backbone.featurizer(pv)
                
                # Run Qwen LLM with vision features to capture Qwen activations
                if self.config.qwen_layer_indices and hasattr(self.vla, 'language_model'):
                    lm = self.vla.language_model
                    if hasattr(lm, 'model'):
                        # Get embedding dimension from LLM
                        embed_dim = lm.config.hidden_size  # 896 for Qwen2.5-0.5B
                        
                        # Project vision features to LLM hidden size if needed
                        if vision_outputs.shape[-1] != embed_dim:
                            # Use a linear projection (or just pad/truncate for visualization)
                            B, N, C = vision_outputs.shape
                            # Simple approach: just use first embed_dim channels or pad
                            if C > embed_dim:
                                hidden_states = vision_outputs[..., :embed_dim]
                            else:
                                pad = torch.zeros(B, N, embed_dim - C, device=vision_outputs.device, dtype=vision_outputs.dtype)
                                hidden_states = torch.cat([vision_outputs, pad], dim=-1)
                        else:
                            hidden_states = vision_outputs
                        
                        # Run through LLM layers
                        for layer in lm.model.layers:
                            layer_outputs = layer(hidden_states, attention_mask=None, position_ids=None)
                            hidden_states = layer_outputs[0]
                            
        except Exception as e:
            print(f"[CAM] Error in forward pass: {e}")
        
        # Prepare overlay images
        overlay_images = []
        for img in original_images:
            img_float = img.astype(np.float32) / 255.0 if img.max() > 1 else img.astype(np.float32)
            if img_float.shape[:2] != (self.config.image_size, self.config.image_size):
                img_float = cv2.resize(img_float, (self.config.image_size, self.config.image_size))
            overlay_images.append(img_float)
        
        # Process captured activations
        for layer_name, activation in self.capture.activations.items():
            # Debug: print activation shape
            print(f"[CAM] {layer_name} activation shape: {activation.shape}")
            
            for view_idx in range(min(self.config.num_views, len(overlay_images))):
                try:
                    # Extract view-specific activation
                    if activation.shape[0] >= self.config.num_views:
                        # Activations from batched views (each view processed separately)
                        view_act = activation[view_idx:view_idx+1]
                    else:
                        view_act = self._extract_view(activation, view_idx)
                    
                    # Compute CAM (dynamically determines spatial size from token count)
                    cam_mask = self._compute_cam(view_act)
                    # Resize to original image size
                    cam_mask = cv2.resize(cam_mask, (self.config.image_size, self.config.image_size))
                    
                    # Create visualization
                    vis = show_cam_on_image(
                        overlay_images[view_idx], 
                        cam_mask, 
                        use_rgb=True, 
                        image_weight=self.config.overlay_alpha
                    )
                    
                    key = f"{layer_name}_v{view_idx}"
                    results[key] = vis
                    
                    if self.config.save_individual:
                        # Filename format: task/frame{step_idx}_view{view_idx}.png
                        save_path = task_dir / f"frame{step_idx:04d}_{key}.png"
                        cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                        
                except Exception as e:
                    print(f"[CAM] Error processing {layer_name}_v{view_idx}: {e}")
        
        # Save grid with frame index prominently displayed
        if self.config.save_grid and results:
            self._save_grid(results, original_images, step_idx, episode_idx, task_dir)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def _save_grid(self, results, original_images, step_idx, episode_idx, save_dir=None):
        """Save grid: rows=layers (siglip, qwen_L5, qwen_L10, ...), cols=views."""
        if not results:
            return
        save_dir = save_dir or self.output_dir
        first = next(iter(results.values()))
        h, w = first.shape[:2]
        
        # Organize results by layer and view
        layers = []
        layer_results = {}
        for key in results:
            # Parse layer name from key (e.g., "siglip_b23_v0" -> "siglip_b23")
            parts = key.rsplit('_v', 1)
            layer_name = parts[0]
            if layer_name not in layer_results:
                layer_results[layer_name] = {}
                layers.append(layer_name)
            view_idx = int(parts[1]) if len(parts) > 1 else 0
            layer_results[layer_name][view_idx] = results[key]
        
        num_views = self.config.num_views
        num_layers = len(layers)
        
        # Grid layout: rows = layers, cols = original images + CAM views
        label_height = 25
        row_label_width = 80
        n_cols = num_views * 2  # original + CAM for each view
        n_rows = num_layers
        
        grid_h = label_height + n_rows * h
        grid_w = row_label_width + n_cols * w
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
        
        # Frame label
        cv2.putText(grid, f"Frame {step_idx}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Column headers
        for v in range(num_views):
            x_orig = row_label_width + v * 2 * w + w // 4
            x_cam = row_label_width + (v * 2 + 1) * w + w // 4
            cv2.putText(grid, f"View{v}", (x_orig, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(grid, f"CAM{v}", (x_cam, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Fill grid
        for row_idx, layer_name in enumerate(layers):
            y_start = label_height + row_idx * h
            
            # Row label
            short_name = layer_name.replace('siglip_b', 'SigLIP_').replace('qwen_L', 'Qwen_L')
            cv2.putText(grid, short_name, (5, y_start + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            for v in range(num_views):
                # Original image
                if v < len(original_images):
                    img = original_images[v]
                    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
                    img_uint8 = cv2.resize(img_uint8, (w, h))
                    x_start = row_label_width + v * 2 * w
                    grid[y_start:y_start+h, x_start:x_start+w] = img_uint8
                
                # CAM image
                if v in layer_results.get(layer_name, {}):
                    cam_img = layer_results[layer_name][v]
                    x_start = row_label_width + (v * 2 + 1) * w
                    grid[y_start:y_start+h, x_start:x_start+w] = cam_img
        
        cv2.imwrite(str(save_dir / f"frame{step_idx:04d}_grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    
    def cleanup(self):
        self.capture.remove_hooks()


def create_cam_config(enabled=True, method="eigencam", output_dir="./grad_cam_outputs",
                      frames_per_episode=5, num_views=2, 
                      siglip_layers=None, qwen_layers=None, **kwargs):
    """Create CAM configuration.
    
    Args:
        enabled: Enable CAM visualization
        method: CAM method (eigencam, gradcam)
        output_dir: Output directory for CAM images
        frames_per_episode: Number of frames to sample per episode
        num_views: Number of camera views
        siglip_layers: List of SigLIP layer indices (default: [-1] = last layer)
        qwen_layers: List of Qwen layer indices (default: [5, 10, 20, 24])
    """
    config = CAMConfig(
        enabled=enabled, 
        method=CAMMethod(method.lower()), 
        output_dir=output_dir,
        frames_per_episode=frames_per_episode, 
        num_views=num_views
    )
    if siglip_layers is not None:
        config.siglip_layer_indices = siglip_layers
    if qwen_layers is not None:
        config.qwen_layer_indices = qwen_layers
    return config


def setup_grad_cam_visualizer(vla_model, processor, config, action_head=None, proprio_projector=None):
    return VLAGradCAMVisualizer(vla_model, processor, config, action_head, proprio_projector)
