"""
models_mae_pamae.py — Physics-Aware Masked Autoencoder for SOC estimation.
 
Extends SatMAE's MaskedAutoencoderViT with a Pedological Coherence Loss
derived from Jenny's CLORPT model (1941). The physics constraint operates
on two soil-forming factors:
 
    Relief  → slope raster  → penalizes dark (high-vegetation/SOC) 
                               reconstructions on steep terrain
    Climate → precipitation → penalizes dark reconstructions in 
                               dry flat regions
 
The total pre-training loss becomes:
    L_total = L_reconstruction + λ · L_coherence
 
where L_coherence is computed only on masked patches (the same patches
the reconstruction loss operates on), ensuring the physics constraint
guides the model's generative predictions rather than memorized inputs.
 
The Coherence Loss is fully differentiable, requires no SOC labels, and
operates during the unsupervised pre-training phase, this is the core
innovation of PAMAE-ViT, bridging the gap between Self-Supervised
Learning (data-efficient) and Physics-Informed ML (scientifically
consistent).
 
"""

from functools import partial

import torch
import torch.nn as nn

from time.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

# Import base classes
from models_mae import MaskedAutoencoderViT


class PedologicalCoherenceLoss(nn.Module):
    """
    Computes the physics-informed coherence loss based on Jenny's CLORPT model for MAE pre-training.

    This module encodes two pedological constraints from the CLORPT model:
 
    1. RELIEF CONSTRAINT (slope-based):
       Steep slopes → high erosion → low SOC accumulation → sparse vegetation
       Therefore: penalize "dark" (high-absorption) reconstructions where
       the slope is steep.
 
       Physical reasoning: On steep terrain (>~30°), gravitational erosion
       removes topsoil faster than organic matter can accumulate. Vegetation
       is sparse due to thin soil and water runoff. Spectral signatures
       should therefore be brighter (lower absorption) on steep slopes.
 
    2. CLIMATE CONSTRAINT (precipitation-based):
       Low rainfall + flat terrain → low biomass input → low SOC
       Therefore: penalize "dark" reconstructions in dry, flat areas.
 
       Physical reasoning: On flat terrain where erosion isn't dominant,
       SOC is primarily driven by organic matter input from vegetation.
       In arid regions (<~750mm/yr), biomass production is water-limited,
       so spectral signatures should not indicate dense vegetation.
 
    Parameters
    ----------
    slope_threshold : float
        Normalized slope above which terrain is considered "steep" (default 0.33,
        corresponding to ~30° since slope is normalized by dividing by 90°).
    precip_threshold : float
        Normalized precipitation below which climate is considered "dry"
        (default 0.3, corresponding to ~750mm/yr with PRECIP_MAX=2500).
    darkness_offset : float
        Baseline subtracted from darkness proxy before penalizing (default 0.0).
        Controls how "dark" a patch must be before the penalty activates.
    
    """

    def __init__(self, slope_threshold=0.33, precip_threshold=0.3, darkness_offset=0.0):
        super().__init__()
        self.slope_threshold = slope_threshold
        self.precip_threshold = precip_threshold
        self.darkness_offset = darkness_offset

        def forward(
                self,
                pred_patches: torch.Tensor,
                slope_patches: torch.Tensor,
                precip: torch.Tensor,
                mask: torch.Tensor
        ) -> torch.Tensor:
            """
            Compute the coherence penalty
            
            Parameters
            ----------
            red_patches : (N, L, C*p*p)
                Reconstructed patches from the MAE decoder (z-score normalized space).
            slope_patches : (N, L)
                Mean normalized slope per patch position, in [0, 1].
            precip : (N,)
                Normalized precipitation per sample, in [0, 1].
            mask : (N, L)
                Binary mask where 1 = masked (reconstructed), 0 = visible.
    
            Returns
            -------
            coherence_loss : scalar tensor
                Mean physics penalty across all masked patches in the batch.
            
            """

            # ─── Darkness proxy ──────────────────────────────────────────
            # In z-score normalized space, more negative values = lower
            # reflectance = darker = more vegetation/organic matter.
            # We compute: darkness = -mean(predicted_reflectance_per_patch)
            # So high darkness = low reflectance = dense vegetation.
            #
            # Note: after z-score normalization, values centered ~0.
            # Negative mean → below-average reflectance → darker than typical.

            patch_mean_reflectance = pred_patches.mean(dim=-1)  # (N, L)
            darkness = -patch_mean_reflectance  # (N, L), higher = darker

            # shift darkness by offset to control penalty activation threshold
            darkness = torch.relu(darkness - self.darkness_offset) # (N, L)

            # ─── Relief penalty ─────────────────────────────────────────
            # Penalize dark reconstructions on steep slopes.
            # steep_mask: 1 where slope exceeds threshold, 0 otherwise
            steep_mask = (slope_patches > self.slope_threshold).float()  # (N, L)

            # Penalty scales with both how dark the prediction is and how steep
            # the slope is (continuous, not just binary threshold)
            relief_penalty = darkness * steep_mask * slope_patches # (N, L), nonzero only on steep patches


            # ─── Climate penalty ────────────────────────────────────────
            # Penalize dark reconstructions in dry, flat areas.
            # dry_mask: 1 where precipitation is below threshold, 0 otherwise
            dry_mask = (precip < self.precip_threshold).float() # (N,)
            flat_mask = 1 - steep_mask  # (N, L)

            # Expand precip and dry_mask to patch dimension
            # dryness_score: how far below the threshold (continuous)

            dryness_score = torch.relu(self.precip_threshold - precip).unsqueeze(1)  # (N, 1)

            climate_penalty = darkness * dry_mask.unsqueeze(1) * flat_mask * dryness_score


            # ─── Total coherence penalty ───────────────────────────────
            # Sum relief and climate penalties
            # Only penalize masked(reconstructed) patches to guide the generative predictions
            total_penalty = relief_penalty + climate_penalty  # (N, L)
            total_penalty_masked = total_penalty * mask  # (N, L), zero out visible patches



            # Mean over all masked patches in the batch
            num_masked_patches = mask.sum()

            if num_masked_patches > 0:
                coherence_loss = total_penalty_masked.sum() / num_masked_patches
            else:
                coherence_loss = torch.tensor(0.0, device=pred_patches.device)

            
            return coherence_loss


class PAMAEViT(MaskedAutoencoderViT):
    """
    Physics-Aware Masked Autoencoder Vision Transformer.
 
    Extends SatMAE's vanilla MaskedAutoencoderViT with:
    1. A patchify_slope() method for slope rasters
    2. A PedologicalCoherenceLoss module
    3. Modified forward() and forward_loss() to compute both
       reconstruction and physics losses
 
    The encoder and decoder architecture is unchanged — only the
    loss computation is extended.
 
    Parameters
    ----------
    lambda_physics : float
        Weight for the coherence loss (default 0.1).
        L_total = L_recon + lambda_physics * L_coherence
    slope_threshold : float
        Normalized slope threshold for "steep" terrain (default 0.33).
    precip_threshold : float
        Normalized precipitation threshold for "dry" climate (default 0.3).
    **kwargs
        All arguments passed to MaskedAutoencoderViT (img_size, patch_size,
        in_chans, embed_dim, depth, etc.)
    """

    def __init__(self, lambda_physics=0.1, slope_threshold=0.33, precip_threshold=0.3, **kwargs):
        super().__init__(**kwargs)

        self.lambda_physics = lambda_physics

        self.coherence_loss_fn = PedologicalCoherenceLoss(
            slope_threshold=slope_threshold,
            precip_threshold=precip_threshold
        )

    def patchify_slope(self, slope: torch.Tensor) -> torch.Tensor:
        """
        Convert slope raster to per-patch mean slope values.

        Parameters
        ----------
        slope : (N, 1, H, W)
            Normalized slope map in [0, 1] (divided by 90°).

        Returns
        -------
        slope_per_patch : (N, L)
            Mean slope value for each patch position.
            L = (H // patch_size) ** 2 = 144 for 96×96 with patch_size=8.
        """

        p = self.patch_embed.patch_size[0]  # assuming square patches

        # Use the existing patchify method with c=1
        slope_patches = self.patchify(slope, p, 1)

        # Mean slope per patch: (N, L)
        slope_per_patch = slope_patches.mean(dim=-1)

        return slope_per_patch
    
    def forward_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        slope: torch.Tensor = None,
        precip: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute combined reconstruction and coherence loss.

        Parameters
        ----------
        imgs : (N, C, H, W)
            Original multi-spectral images.
        pred : (N, L, C*p*p)
            MAE decoder predictions for all patches.Reconstructed patches from the decoder.
        mask : (N, L)
            Binary mask where 1 = masked (reconstructed), 0 = visible.
        slope : (N, 1, H, W)
            Normalized slope maps in [0, 1]. If None, coherence loss = 0.
        precip : (N,)
            Normalized precipitation  If None, coherence loss = 0.

        Returns
        -------
        loss_total : scalar
            L_recon + lambda * L_coherence
        loss_recon : scalar
            Standard MAE reconstruction MSE on masked patches.
        loss_physics : scalar
            Pedological coherence penalty on masked patches.
        """
        
        # ─── Standard reconstruction loss (unchanged from SatMAE) ────
        target = self.patchify(imgs, self.patch_embed.patch_size[0], self.in_c)
 
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
 
        loss_recon = (pred - target) ** 2
        loss_recon = loss_recon.mean(dim=-1)  # (N, L), mean loss per patch
        loss_recon = (loss_recon * mask).sum() / mask.sum()  # mean on masked
 
        # ─── Coherence loss ──────────────────────────────────────────
        if slope is not None and precip is not None:
            # Convert slope raster to per-patch mean slope values
            slope_patches = self.patchify_slope(slope)  # (N, L)

            # Compute physics penalty
            loss_physics = self.coherence_loss_fn(
                pred_patches=pred,
                slope_patches=slope_patches,
                precip=precip,
                mask=mask
            )
        else:
            loss_physics = torch.tensor(0.0, device=imgs.device)

        # ─── Combine ────────────────────────────────────────────────
        loss_total = loss_recon + self.lambda_physics * loss_physics

        return loss_total, loss_recon, loss_physics
    
    def forward(
        self,
        imgs: torch.Tensor,
        slope: torch.Tensor = None,
        precip: torch.Tensor = None,
        mask_ratio: float = 0.75,
    ) -> torch.Tensor:
        """
        Full forward pass: encode → decode → loss.
 
        Parameters
        ----------
        imgs : (N, C, H, W)
            Multi-spectral satellite images.
        slope : (N, 1, H, W) or None
            Normalized slope raster.
        precip : (N,) or None
            Normalized precipitation scalar.
        mask_ratio : float
            Fraction of patches to mask (default 0.75).
 
        Returns
        -------
        loss_total : scalar
            Combined loss.
        loss_recon : scalar
            Reconstruction loss (for logging).
        loss_physics : scalar
            Physics loss (for logging).
        pred : (N, L, C*p*p)
            Predicted patches.
        mask : (N, L)
            Binary mask.
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore) 

        loss_total, loss_recon, loss_physics = self.forward_loss(
            imgs=imgs,
            pred=pred,
            mask=mask,
            slope=slope,
            precip=precip)

        return loss_total, loss_recon, loss_physics, pred, mask
    

# ──────────────────────────────────────────────────────────────
# Model factory functions (drop-in replacements for SatMAE's)
# ──────────────────────────────────────────────────────────────
def pamae_vit_base_patch16(**kwargs):
    """PAMAE-ViT Base: 12 encoder blocks, 768-dim, 8 decoder blocks, 512-dim."""
    model = PAMAEViT(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
 
 
def pamae_vit_large_patch16(**kwargs):
    """PAMAE-ViT Large: 24 encoder blocks, 1024-dim, 8 decoder blocks, 512-dim."""
    model = PAMAEViT(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

