"""
ViT encoder with regression head for SOC estimation.
 
Phase 3 of PAMAE-ViT: Fine-tuning for Soil Organic Carbon regression.
 
This module builds an encoder-only Vision Transformer for regression by:
1. Constructing a ViT-Base encoder (same architecture as pre-training)
2. Loading pre-trained encoder weights from checkpoint-100.pth (decoder discarded)
3. Attaching a regression head: LayerNorm → Linear → GELU → Dropout → Linear(1)
4. Returning a scalar SOC prediction per sample
 
The encoder architecture mirrors the PAMAEViT encoder exactly:
- PatchEmbed: Conv2d(11, 768, 8×8)
- 12 Transformer blocks (768-dim, 12 heads)
- LayerNorm
- CLS token + sinusoidal positional embedding (145 tokens = 1 CLS + 144 patches)
 
Key design decisions:
- Global average pooling over patch tokens (excludes CLS) → more stable than
  CLS-only for regression with small datasets (Zhai et al., 2022)
- Regression head has one hidden layer (768→256→1), not just a linear probe,
  because SOC is a complex function of spectral features
- Drop path (stochastic depth) added during fine-tuning for regularization

"""

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

class ViTForSOCRegression(nn.Module):
    """
    Vision Transformer for SOC regression.
 
    Architecture
    ------------
    Input:  (B, 11, 96, 96) — z-score normalized Sentinel-2 bands
    Encoder: ViT-Base (12 blocks, 768-dim, 12 heads, patch_size=8)
    Pooling: Global average pool over 144 patch tokens (excl. CLS)
    Head:    LayerNorm(768) → Linear(768, 256) → GELU → Dropout → Linear(256, 1)
    Output:  (B,) — predicted log1p(SOC) per sample
 
    Parameters
    ----------
    img_size : int
        Input image size (default: 96).
    patch_size : int
        Patch size (default: 8). Yields (96/8)² = 144 patches.
    in_chans : int
        Number of input channels (default: 11 for Sentinel-2 bands).
    embed_dim : int
        Embedding dimension (default: 768 for ViT-Base).
    depth : int
        Number of transformer blocks (default: 12 for ViT-Base).
    num_heads : int
        Number of attention heads (default: 12 for ViT-Base).
    mlp_ratio : float
        MLP hidden dim ratio (default: 4.0).
    drop_path_rate : float
        Stochastic depth rate (default: 0.2).
    head_dropout : float
        Dropout rate in the regression head (default: 0.1).
    """

    def __init__(self, img_size=96, patch_size=8, in_chans=11, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., drop_path_rate=0.2, head_dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim

        # ── Encoder (mirrors PAMAEViT encoder exactly) ────
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # CLS token and positional embedding
        num_patches = self.patch_embed.num_patches # (96/8)^2 = 144
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # +1 for CLS token

        # Transformer blocks with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # ── Global average pooling norm ──────────────────────────────
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize positional embeddings, CLS token, and linear layers."""
        # Sinusoidal positional embedding (frozen, not learned)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # CLS token
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # Linear layers
        self.apply(self._init_linear)


    @staticmethod
    def _init_linear(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def no_weight_decay(self):
        """Parameters excluded from weight decay (used by lrd.param_groups_lrd)."""
        return {"pos_embed", "cls_token"}
    
    def forward(self, x):
        """
        Forward pass.
 
        Parameters
        ----------
        x : (B, 11, 96, 96)
            Z-score normalized Sentinel-2 image patches.
 
        Returns
        -------
        soc_pred : (B,)
            Predicted log1p(SOC) values.
        """
        B = x.shape[0]
 
        # Patch embedding: (B, 11, 96, 96) → (B, 144, 768)
        x = self.patch_embed(x)
 
        # Prepend CLS token: (B, 145, 768)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
 
        # Add positional embedding
        x = x + self.pos_embed
 
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
 
        # Global average pool over patch tokens (exclude CLS at index 0)
        x = x[:, 1:, :].mean(dim=1)  # (B, 768)
 
        # LayerNorm after pooling
        x = self.fc_norm(x)
 
        # Regression head: (B, 768) → (B, 1) → (B,)
        soc_pred = self.head(x).squeeze(-1)
 
        return soc_pred

# ──────────────────────────────────────────────────────────────
# Model factory function
# ──────────────────────────────────────────────────────────────
def vit_base_patch8_regression(**kwargs):
    """ViT-Base for SOC regression: 12 blocks, 768-dim, patch_size=8, 96×96 input."""
    model = ViTForSOCRegression(
        img_size=96,
        patch_size=8,
        in_chans=11,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        **kwargs,
    )
    return model


# ──────────────────────────────────────────────────────────────
# Weight loading utility
# ──────────────────────────────────────────────────────────────
def load_pretrained_encoder(model, checkpoint_path, device="cpu"):
    """
    Load encoder weights from a PAMAEViT pre-training checkpoint.
 
    The checkpoint contains the full MAE model (encoder + decoder +
    coherence loss module). This function extracts only encoder-relevant
    weights and loads them into the ViTForSOCRegression model.
 
    Specifically loaded:
    - patch_embed.proj.weight, patch_embed.proj.bias
    - cls_token
    - pos_embed
    - blocks.{0-11}.{norm1, attn, norm2, mlp}.*
    - norm.weight, norm.bias
 
    Specifically skipped:
    - decoder_embed, decoder_blocks, decoder_norm, decoder_pred
    - decoder_pos_embed, mask_token
    - coherence_loss_fn.*
 
    Parameters
    ----------
    model : ViTForSOCRegression
        Target model to load weights into.
    checkpoint_path : str
        Path to checkpoint-100.pth (or any PAMAEViT checkpoint).
    device : str
        Device to load the checkpoint onto.
 
    Returns
    -------
    msg : NamedTuple
        Output of load_state_dict with missing_keys and unexpected_keys.
    """
    print(f"Loading pre-trained checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_model = checkpoint["model"]
 
    # ── Filter: keep only encoder keys ──────────────────────────
    encoder_prefixes = (
        "patch_embed.",
        "cls_token",
        "pos_embed",
        "blocks.",
        "norm.",
    )
    decoder_prefixes = (
        "decoder_",
        "mask_token",
        "coherence_loss_fn.",
    )
 
    encoder_state = {}
    skipped_keys = []
 
    for k, v in checkpoint_model.items():
        # Skip anything that starts with a decoder prefix
        if any(k.startswith(dp) for dp in decoder_prefixes):
            skipped_keys.append(k)
            continue
        # Keep anything that starts with an encoder prefix
        if any(k.startswith(ep) for ep in encoder_prefixes):
            encoder_state[k] = v
        else:
            skipped_keys.append(k)
 
    print(f"  Encoder keys loaded: {len(encoder_state)}")
    print(f"  Decoder/other keys skipped: {len(skipped_keys)}")
 
    # ── Load into model (strict=False allows missing head weights) ──
    msg = model.load_state_dict(encoder_state, strict=False)
 
    # Expected missing keys: fc_norm.* and head.*
    print(f"  Missing keys (expected — new head): {msg.missing_keys}")
    if msg.unexpected_keys:
        print(f"  Unexpected keys (should be empty): {msg.unexpected_keys}")
 
    return msg
 
        