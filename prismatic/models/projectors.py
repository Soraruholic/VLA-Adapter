"""Implementation of additional projectors for additional inputs to the VLA models."""
import torch
import torch.nn as nn


class ProprioProjector(nn.Module):
    """
    Projects proprio state inputs into the LLM's embedding space.
    """
    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim

        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class NoisyActionProjector(nn.Module):
    """
    [Diffusion] Projects noisy action inputs into the LLM's embedding space.

    Note that since each action is tokenized into 7 tokens in OpenVLA (rather
    than having 1 token per action), each noisy action token will have dimension 1
    instead of 7.
    """
    def __init__(self, llm_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.action_token_dim = 1

        self.fc1 = nn.Linear(self.action_token_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, noisy_actions: torch.Tensor = None) -> torch.Tensor:
        # noisy_actions: (bsz, num_action_tokens=chunk_len*action_dim, 1)
        projected_features = self.fc1(noisy_actions)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class AlignProjector(nn.Module):
    """
    Calculate the alignment between LLM and VGGT embeddings.
    Projects LLM embeddings to 2*vggt_dim (2048) to align with concat(frame, global) features.
    
    Supports two modes:
    - share_projector=True: All views share a single projector (default, backward compatible)
    - share_projector=False: Each view has its own independent projector
    
    Args:
        llm_dim: LLM hidden dimension
        vggt_dim: VGGT feature dimension (1024), output will be 2*vggt_dim
        align_loss_type: Loss type for alignment ("cosine")
        use_vlm_norm: Whether to apply LayerNorm to VLM embeddings before projection
        num_views: Number of views (only used when share_projector=False)
        share_projector: Whether to share projector across views
    """
    def __init__(
            self, 
            llm_dim: int, 
            vggt_dim: int,
            align_loss_type: str = "cosine",
            use_vlm_norm: bool = False,
            num_views: int = 1,
            share_projector: bool = True,
        ) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.vggt_dim = vggt_dim
        self.align_loss_type = align_loss_type
        self.num_views = num_views
        self.share_projector = share_projector

        if share_projector:
            # Single shared projector for all views (original behavior)
            self.fc1 = nn.Linear(self.llm_dim, 2 * self.vggt_dim, bias=True)
            self.fc2 = nn.Linear(2 * self.vggt_dim, 2 * self.vggt_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.vlm_norm = nn.LayerNorm(llm_dim) if use_vlm_norm else None
        else:
            # Separate projector for each view
            self.projectors = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(llm_dim) if use_vlm_norm else nn.Identity(),
                    nn.Linear(llm_dim, 2 * vggt_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(2 * vggt_dim, 2 * vggt_dim, bias=True),
                ) for _ in range(num_views)
            ])

        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def align_dimension(self, LLM_embedding: torch.Tensor, view_idx: int = None) -> torch.Tensor:
        """
        Project LLM embeddings to VGGT feature dimension.
        
        Args:
            LLM_embedding: [B, N*P, D_llm] for shared mode, or [B, P, D_llm] for per-view mode
            view_idx: View index (only used when share_projector=False)
        """
        if self.share_projector:
            if hasattr(self, 'vlm_norm') and self.vlm_norm is not None:
                LLM_embedding = self.vlm_norm(LLM_embedding)
            projected_features = self.fc1(LLM_embedding)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            if view_idx is not None:
                # Process single view
                projected_features = self.projectors[view_idx](LLM_embedding)
            else:
                # Process all views: assume LLM_embedding shape [B, N, P, D_llm]
                B, N, P, D = LLM_embedding.shape
                projected_list = []
                for i in range(N):
                    proj_i = self.projectors[i](LLM_embedding[:, i])  # [B, P, 2*vggt_dim]
                    projected_list.append(proj_i)
                projected_features = torch.stack(projected_list, dim=1)  # [B, N, P, 2*vggt_dim]
        return projected_features
    
    def compute_align_loss_cosine(self, vision_hidden, vggt_hidden):
        """
        Compute cosine similarity loss between projected VLA tokens and VGGT features.
        
        Args:
            vision_hidden: Projected VLA tokens [B, N*P, D] or [B, N, P, D]
            vggt_hidden: VGGT features [B, N*P, D] or [B, N, P, D]
        """
        def mean_flat(x):
            return torch.mean(x, dim=list(range(1, len(x.size()))))
        
        # Flatten if needed for per-sample loss computation
        if len(vision_hidden.shape) == 4:
            B, N, P, D = vision_hidden.shape
            vision_hidden = vision_hidden.reshape(B, N * P, D)
            vggt_hidden = vggt_hidden.reshape(B, N * P, D)
        
        align_loss = 0
        bsz = vision_hidden.shape[0]
        for _vision, _vggt in zip(vision_hidden, vggt_hidden):
            _vision = torch.nn.functional.normalize(_vision, dim=-1)
            _vggt = torch.nn.functional.normalize(_vggt, dim=-1)
            align_loss += 1 - mean_flat((_vision * _vggt).sum(dim=-1))
        align_loss /= bsz
        return align_loss
    
    def forward(self, LLM_emb, target_emb):
        """
        Compute alignment loss.
        
        Args:
            LLM_emb: VLA vision tokens [B, N*P, D_llm] or [B, N, P, D_llm]
            target_emb: VGGT features [B, N*P, 2048] or [B, N, P, 2048]
        """
        if self.align_loss_type == "cosine":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                LLM_emb = self.align_dimension(LLM_emb)
            align_loss = self.compute_align_loss_cosine(LLM_emb, target_emb).mean()
            return align_loss
        else:
            raise NotImplementedError(f"Align loss type {self.align_loss_type} is not implemented.")


class FrameAlignProjector(nn.Module):
    """
    [DUAL ALIGN] Frame-level alignment projector.
    Projects LLM embeddings to vggt_dim (1024) to align with VGGT frame-level features.
    
    Frame features represent per-view spatial information (intra-frame attention).
    Each view's VLA tokens align with the corresponding view's frame features.
    
    Args:
        llm_dim: LLM hidden dimension
        vggt_dim: VGGT feature dimension (1024)
        align_loss_type: Loss type for alignment ("cosine")
        use_vlm_norm: Whether to apply LayerNorm to VLM embeddings before projection
        num_views: Number of views (only used when share_projector=False)
        share_projector: Whether to share projector across views
    """
    def __init__(
            self, 
            llm_dim: int, 
            vggt_dim: int = 1024,
            align_loss_type: str = "cosine",
            use_vlm_norm: bool = False,
            num_views: int = 1,
            share_projector: bool = True,
        ) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.vggt_dim = vggt_dim
        self.align_loss_type = align_loss_type
        self.num_views = num_views
        self.share_projector = share_projector

        if share_projector:
            # Single shared projector for all views
            self.fc1 = nn.Linear(self.llm_dim, self.vggt_dim, bias=True)
            self.fc2 = nn.Linear(self.vggt_dim, self.vggt_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.vlm_norm = nn.LayerNorm(llm_dim) if use_vlm_norm else None
        else:
            # Separate projector for each view
            self.projectors = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(llm_dim) if use_vlm_norm else nn.Identity(),
                    nn.Linear(llm_dim, vggt_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(vggt_dim, vggt_dim, bias=True),
                ) for _ in range(num_views)
            ])
        
        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def align_dimension(self, LLM_embedding: torch.Tensor, view_idx: int = None) -> torch.Tensor:
        """
        Project LLM embeddings to VGGT frame feature dimension.
        
        Args:
            LLM_embedding: [B, P, D_llm] for single view or [B, N, P, D_llm] for multi-view
            view_idx: View index (only used when share_projector=False and processing single view)
        """
        if self.share_projector:
            if hasattr(self, 'vlm_norm') and self.vlm_norm is not None:
                LLM_embedding = self.vlm_norm(LLM_embedding)
            projected_features = self.fc1(LLM_embedding)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            if view_idx is not None:
                # Process single view
                projected_features = self.projectors[view_idx](LLM_embedding)
            else:
                # Process all views: LLM_embedding shape [B, N, P, D_llm]
                B, N, P, D = LLM_embedding.shape
                projected_list = []
                for i in range(N):
                    proj_i = self.projectors[i](LLM_embedding[:, i])  # [B, P, vggt_dim]
                    projected_list.append(proj_i)
                projected_features = torch.stack(projected_list, dim=1)  # [B, N, P, vggt_dim]
        return projected_features
    
    def compute_align_loss_cosine(self, vision_hidden, vggt_hidden):
        """
        Compute cosine similarity loss between projected VLA tokens and VGGT features.
        
        Args:
            vision_hidden: Projected VLA tokens [B, N*P, D] or [B, N, P, D]
            vggt_hidden: VGGT frame features [B, N*P, D] or [B, N, P, D]
        """
        def mean_flat(x):
            return torch.mean(x, dim=list(range(1, len(x.size()))))
        
        # Flatten if needed for per-sample loss computation
        if len(vision_hidden.shape) == 4:
            B, N, P, D = vision_hidden.shape
            vision_hidden = vision_hidden.reshape(B, N * P, D)
            vggt_hidden = vggt_hidden.reshape(B, N * P, D)
        
        align_loss = 0
        bsz = vision_hidden.shape[0]
        for _vision, _vggt in zip(vision_hidden, vggt_hidden):
            _vision = torch.nn.functional.normalize(_vision, dim=-1)
            _vggt = torch.nn.functional.normalize(_vggt, dim=-1)
            align_loss += 1 - mean_flat((_vision * _vggt).sum(dim=-1))
        align_loss /= bsz
        return align_loss
    
    def forward(self, LLM_emb, target_emb):
        """
        Compute frame-level alignment loss.
        
        Args:
            LLM_emb: VLA vision tokens [B, N*P, D_llm] or [B, N, P, D_llm]
            target_emb: VGGT frame features [B, N*P, 1024] or [B, N, P, 1024]
        """
        if self.align_loss_type == "cosine":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                LLM_emb = self.align_dimension(LLM_emb)
            align_loss = self.compute_align_loss_cosine(LLM_emb, target_emb).mean()
            return align_loss
        else:
            raise NotImplementedError(f"Align loss type {self.align_loss_type} is not implemented.")


class GlobalAlignProjector(nn.Module):
    """
    [DUAL ALIGN] Global-level alignment projector.
    Projects LLM embeddings to vggt_dim (1024) to align with VGGT global-level features.
    
    Global features represent cross-view aggregated information (inter-frame attention).
    All views share the same global projector since global features encode cross-view relationships.
    
    Args:
        llm_dim: LLM hidden dimension
        vggt_dim: VGGT feature dimension (1024)
        align_loss_type: Loss type for alignment ("cosine")
        use_vlm_norm: Whether to apply LayerNorm to VLM embeddings before projection
    """
    def __init__(
            self, 
            llm_dim: int, 
            vggt_dim: int = 1024,
            align_loss_type: str = "cosine",
            use_vlm_norm: bool = False,
        ) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.vggt_dim = vggt_dim
        self.align_loss_type = align_loss_type

        self.fc1 = nn.Linear(self.llm_dim, self.vggt_dim, bias=True)
        self.fc2 = nn.Linear(self.vggt_dim, self.vggt_dim, bias=True)
        self.act_fn1 = nn.GELU()
        
        self.vlm_norm = nn.LayerNorm(llm_dim) if use_vlm_norm else None

        self.initialize_weights()
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def align_dimension(self, LLM_embedding: torch.Tensor) -> torch.Tensor:
        """
        Project LLM embeddings to VGGT global feature dimension.
        
        Args:
            LLM_embedding: [B, N*P, D_llm] - all views' tokens concatenated
        """
        if self.vlm_norm is not None:
            LLM_embedding = self.vlm_norm(LLM_embedding)
        projected_features = self.fc1(LLM_embedding)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features
    
    def compute_align_loss_cosine(self, vision_hidden, vggt_hidden):
        """
        Compute cosine similarity loss between projected VLA tokens and VGGT global features.
        
        Args:
            vision_hidden: Projected VLA tokens [B, N*P, D]
            vggt_hidden: VGGT global features [B, N*P, D]
        """
        def mean_flat(x):
            return torch.mean(x, dim=list(range(1, len(x.size()))))
        
        # Flatten if needed
        if len(vision_hidden.shape) == 4:
            B, N, P, D = vision_hidden.shape
            vision_hidden = vision_hidden.reshape(B, N * P, D)
            vggt_hidden = vggt_hidden.reshape(B, N * P, D)
        
        align_loss = 0
        bsz = vision_hidden.shape[0]
        for _vision, _vggt in zip(vision_hidden, vggt_hidden):
            _vision = torch.nn.functional.normalize(_vision, dim=-1)
            _vggt = torch.nn.functional.normalize(_vggt, dim=-1)
            align_loss += 1 - mean_flat((_vision * _vggt).sum(dim=-1))
        align_loss /= bsz
        return align_loss
    
    def forward(self, LLM_emb, target_emb):
        """
        Compute global-level alignment loss.
        
        Args:
            LLM_emb: VLA vision tokens [B, N*P, D_llm] - all views concatenated
            target_emb: VGGT global features [B, N*P, 1024] - global attention output
        """
        if self.align_loss_type == "cosine":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                LLM_emb = self.align_dimension(LLM_emb)
            align_loss = self.compute_align_loss_cosine(LLM_emb, target_emb).mean()
            return align_loss
        else:
            raise NotImplementedError(f"Align loss type {self.align_loss_type} is not implemented.")

