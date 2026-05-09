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

    
    """