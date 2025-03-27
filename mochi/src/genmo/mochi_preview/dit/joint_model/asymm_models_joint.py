import os
from typing import Dict, List, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention import sdpa_kernel

import genmo.mochi_preview.dit.joint_model.context_parallel as cp
from genmo.lib.attn_imports import flash_varlen_attn, sage_attn, sdpa_attn_ctx
from genmo.mochi_preview.dit.joint_model.layers import (
    FeedForward,
    PatchEmbed,
    RMSNorm,
    TimestepEmbedder,
)
from genmo.mochi_preview.dit.joint_model.lora import LoraLinear
from genmo.mochi_preview.dit.joint_model.mod_rmsnorm import modulated_rmsnorm
from genmo.mochi_preview.dit.joint_model.residual_tanh_gated_rmsnorm import (
    residual_tanh_gated_rmsnorm,
)
from genmo.mochi_preview.dit.joint_model.rope_mixed import (
    compute_mixed_rotation,
    create_position_matrix,
)
from genmo.mochi_preview.dit.joint_model.temporal_rope import apply_rotary_emb_qk_real
from genmo.mochi_preview.dit.joint_model.utils import (
    AttentionPool,
    modulate,
    pad_and_split_xy,
)

from genmo.mochi_preview.dit.joint_model.audio_adapter import AudioProjector, AudioCrossAttention

COMPILE_FINAL_LAYER = os.environ.get("COMPILE_DIT") == "1"
COMPILE_MMDIT_BLOCK = os.environ.get("COMPILE_DIT") == "1"


def ck(fn, *args, enabled=True, **kwargs) -> torch.Tensor:
    if enabled:
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs, use_reentrant=False)

    return fn(*args, **kwargs)


class AsymmetricAttention(nn.Module):
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        update_y: bool = True,
        out_bias: bool = True,
        attention_mode: str = "flash",
        softmax_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
        # Disable LoRA by default ...
        qkv_proj_lora_rank: int = 0,
        qkv_proj_lora_alpha: int = 0,
        qkv_proj_lora_dropout: float = 0.0,
        out_proj_lora_rank: int = 0,
        out_proj_lora_alpha: int = 0,
        out_proj_lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.attention_mode = attention_mode
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_heads = num_heads
        self.head_dim = dim_x // num_heads
        self.update_y = update_y
        self.softmax_scale = softmax_scale
        if dim_x % num_heads != 0:
            raise ValueError(f"dim_x={dim_x} should be divisible by num_heads={num_heads}")

        # Input layers.
        self.qkv_bias = qkv_bias
        qkv_lora_kwargs = dict(
            bias=qkv_bias,
            device=device,
            r=qkv_proj_lora_rank,
            lora_alpha=qkv_proj_lora_alpha,
            lora_dropout=qkv_proj_lora_dropout,
        )
        self.qkv_x = LoraLinear(dim_x, 3 * dim_x, **qkv_lora_kwargs)
        # Project text features to match visual features (dim_y -> dim_x)
        self.qkv_y = LoraLinear(dim_y, 3 * dim_x, **qkv_lora_kwargs)

        # Query and key normalization for stability.
        assert qk_norm
        self.q_norm_x = RMSNorm(self.head_dim, device=device)
        self.k_norm_x = RMSNorm(self.head_dim, device=device)
        self.q_norm_y = RMSNorm(self.head_dim, device=device)
        self.k_norm_y = RMSNorm(self.head_dim, device=device)

        # Output layers. y features go back down from dim_x -> dim_y.
        proj_lora_kwargs = dict(
            bias=out_bias,
            device=device,
            r=out_proj_lora_rank,
            lora_alpha=out_proj_lora_alpha,
            lora_dropout=out_proj_lora_dropout,
        )
        self.proj_x = LoraLinear(dim_x, dim_x, **proj_lora_kwargs)
        self.proj_y = LoraLinear(dim_x, dim_y, **proj_lora_kwargs) if update_y else nn.Identity()

    def run_qkv_y(self, y):
        cp_rank, cp_size = cp.get_cp_rank_size()
        local_heads = self.num_heads // cp_size

        if cp.is_cp_active():
            # Only predict local heads.
            assert not self.qkv_bias
            W_qkv_y = self.qkv_y.weight.view(3, self.num_heads, self.head_dim, self.dim_y)
            W_qkv_y = W_qkv_y.narrow(1, cp_rank * local_heads, local_heads)
            W_qkv_y = W_qkv_y.reshape(3 * local_heads * self.head_dim, self.dim_y)
            qkv_y = F.linear(y, W_qkv_y, None)  # (B, L, 3 * local_h * head_dim)
        else:
            qkv_y = self.qkv_y(y)  # (B, L, 3 * dim)

        qkv_y = qkv_y.view(qkv_y.size(0), qkv_y.size(1), 3, local_heads, self.head_dim)
        q_y, k_y, v_y = qkv_y.unbind(2)

        q_y = self.q_norm_y(q_y)
        k_y = self.k_norm_y(k_y)
        return q_y, k_y, v_y

    def prepare_qkv(
        self,
        x: torch.Tensor,  # (B, M, dim_x)
        y: torch.Tensor,  # (B, L, dim_y)
        *,
        scale_x: torch.Tensor,
        scale_y: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        valid_token_indices: torch.Tensor,
        max_seqlen_in_batch: int,
    ):
        # Process visual features
        x = modulated_rmsnorm(x, scale_x)  # (B, M, dim_x) where M = N / cp_group_size
        qkv_x = self.qkv_x(x)  # (B, M, 3 * dim_x)
        assert qkv_x.dtype == torch.bfloat16

        qkv_x = cp.all_to_all_collect_tokens(qkv_x, self.num_heads)  # (3, B, N, local_h, head_dim)

        # Split qkv_x into q, k, v
        q_x, k_x, v_x = qkv_x.unbind(0)  # (B, N, local_h, head_dim)
        q_x = self.q_norm_x(q_x)
        q_x = apply_rotary_emb_qk_real(q_x, rope_cos, rope_sin)
        k_x = self.k_norm_x(k_x)
        k_x = apply_rotary_emb_qk_real(k_x, rope_cos, rope_sin)

        # Concatenate streams
        B, N, num_heads, head_dim = q_x.size()
        D = num_heads * head_dim

        # Process text features
        if B == 1:
            text_seqlen = max_seqlen_in_batch - N
            if text_seqlen > 0:
                y = y[:, :text_seqlen]  # Remove padding tokens.
                y = modulated_rmsnorm(y, scale_y)  # (B, L, dim_y)
                q_y, k_y, v_y = self.run_qkv_y(y)  # (B, L, local_heads, head_dim)

                q = torch.cat([q_x, q_y], dim=1)
                k = torch.cat([k_x, k_y], dim=1)
                v = torch.cat([v_x, v_y], dim=1)
            else:
                q, k, v = q_x, k_x, v_x
        else:
            y = modulated_rmsnorm(y, scale_y)  # (B, L, dim_y)
            q_y, k_y, v_y = self.run_qkv_y(y)  # (B, L, local_heads, head_dim)

            indices = valid_token_indices[:, None].expand(-1, D)
            q = torch.cat([q_x, q_y], dim=1).view(-1, D).gather(0, indices)  # (total, D)
            k = torch.cat([k_x, k_y], dim=1).view(-1, D).gather(0, indices)  # (total, D)
            v = torch.cat([v_x, v_y], dim=1).view(-1, D).gather(0, indices)  # (total, D)

        q = q.view(-1, num_heads, head_dim)
        k = k.view(-1, num_heads, head_dim)
        v = v.view(-1, num_heads, head_dim)
        return q, k, v

    @torch.autocast("cuda", enabled=False)
    def flash_attention(self, q, k, v, cu_seqlens, max_seqlen_in_batch, total, local_dim):
        out: torch.Tensor = flash_varlen_attn(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen_in_batch,
            max_seqlen_k=max_seqlen_in_batch,
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
        )  # (total, local_heads, head_dim)
        return out.view(total, local_dim)

    def sdpa_attention(self, q, k, v):
        with sdpa_attn_ctx(training=self.training):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            return out

    @torch.autocast("cuda", enabled=False)
    def sage_attention(self, q, k, v):
        return sage_attn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

    def run_attention(
        self,
        q: torch.Tensor,  # (total <= B * (N + L), num_heads, head_dim)
        k: torch.Tensor,  # (total <= B * (N + L), num_heads, head_dim)
        v: torch.Tensor,  # (total <= B * (N + L), num_heads, head_dim)
        *,
        B: int,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen_in_batch: Optional[int] = None,
    ):
        _, cp_size = cp.get_cp_rank_size()
        assert self.num_heads % cp_size == 0
        local_heads = self.num_heads // cp_size
        local_dim = local_heads * self.head_dim

        # Check shapes
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        total = q.size(0)
        assert k.size(0) == total and v.size(0) == total

        if self.attention_mode == "flash":
            out = self.flash_attention(
                q, k, v, cu_seqlens, max_seqlen_in_batch, total, local_dim)  # (total, local_dim)
        else:
            assert B == 1, \
                f"Non-flash attention mode {self.attention_mode} only supports batch size 1, got {B}"

            q = rearrange(q, "(b s) h d -> b h s d", b=B)
            k = rearrange(k, "(b s) h d -> b h s d", b=B)
            v = rearrange(v, "(b s) h d -> b h s d", b=B)

            if self.attention_mode == "sdpa":
                out = self.sdpa_attention(q, k, v)  # (B, local_heads, seq_len, head_dim)
            elif self.attention_mode == "sage":
                out = self.sage_attention(q, k, v)  # (B, local_heads, seq_len, head_dim)
            else:
                raise ValueError(f"Unknown attention mode: {self.attention_mode}")

            out = rearrange(out, "b h s d -> (b s) (h d)")

        return out

    def post_attention(
        self,
        out: torch.Tensor,
        B: int,
        M: int,
        L: int,
        dtype: torch.dtype,
        valid_token_indices: torch.Tensor,
    ):
        """
        Args:
            out: (total <= B * (N + L), local_dim)
            valid_token_indices: (total <= B * (N + L),)
            B: Batch size
            M: Number of visual tokens per context parallel rank
            L: Number of text tokens
            dtype: Data type of the input and output tensors

        Returns:
            x: (B, N, dim_x) tensor of visual tokens where N = M * cp_size
            y: (B, L, dim_y) tensor of text token features
        """
        _, cp_size = cp.get_cp_rank_size()
        local_heads = self.num_heads // cp_size
        local_dim = local_heads * self.head_dim
        N = M * cp_size

        # Split sequence into visual and text tokens, adding back padding.
        if B == 1:
            out = out.view(B, -1, local_dim)
            if out.size(1) > N:
                x, y = torch.tensor_split(out, (N,), dim=1)  # (B, N, local_dim), (B, <= L, local_dim)
                y = F.pad(y, (0, 0, 0, L - y.size(1)))  # (B, L, local_dim)
            else:
                # Empty prompt.
                x, y = out, out.new_zeros(B, L, local_dim)
        else:
            x, y = pad_and_split_xy(out, valid_token_indices, B, N, L, dtype)
        assert x.size() == (B, N, local_dim)
        assert y.size() == (B, L, local_dim)

        # Communicate across context parallel ranks.
        x = x.view(B, N, local_heads, self.head_dim)
        x = cp.all_to_all_collect_heads(x)  # (B, M, dim_x = num_heads * head_dim)
        if cp.is_cp_active():
            y = cp.all_gather(y)  # (cp_size * B, L, local_heads * head_dim)
            y = rearrange(y, "(G B) L D -> B L (G D)", G=cp_size, D=local_dim)  # (B, L, dim_x)

        x = self.proj_x(x)
        y = self.proj_y(y)
        return x, y

    def forward(
        self,
        x: torch.Tensor,  # (B, M, dim_x)
        y: torch.Tensor,  # (B, L, dim_y)
        *,
        scale_x: torch.Tensor,  # (B, dim_x), modulation for pre-RMSNorm.
        scale_y: torch.Tensor,  # (B, dim_y), modulation for pre-RMSNorm.
        packed_indices: Dict[str, torch.Tensor] = None,
        checkpoint_qkv: bool = False,
        checkpoint_post_attn: bool = False,
        **rope_rotation,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of asymmetric multi-modal attention.

        Args:
            x: (B, M, dim_x) tensor of visual tokens
            y: (B, L, dim_y) tensor of text token features
            packed_indices: Dict with keys for Flash Attention
            num_frames: Number of frames in the video. N = num_frames * num_spatial_tokens

        Returns:
            x: (B, M, dim_x) tensor of visual tokens after multi-modal attention
            y: (B, L, dim_y) tensor of text token features after multi-modal attention
        """
        B, L, _ = y.shape
        _, M, _ = x.shape

        # Predict a packed QKV tensor from visual and text features.
        q, k, v = ck(self.prepare_qkv,
            x=x,
            y=y,
            scale_x=scale_x,
            scale_y=scale_y,
            rope_cos=rope_rotation.get("rope_cos"),
            rope_sin=rope_rotation.get("rope_sin"),
            valid_token_indices=packed_indices["valid_token_indices_kv"],
            max_seqlen_in_batch=packed_indices["max_seqlen_in_batch_kv"],
            enabled=checkpoint_qkv,
        )  # (total <= B * (N + L), 3, local_heads, head_dim)

        # Self-attention is expensive, so don't checkpoint it.
        out = self.run_attention(
            q, k, v, B=B,
            cu_seqlens=packed_indices["cu_seqlens_kv"],
            max_seqlen_in_batch=packed_indices["max_seqlen_in_batch_kv"],
        )

        x, y = ck(self.post_attention,
            out,
            B=B, M=M, L=L,
            dtype=v.dtype,
            valid_token_indices=packed_indices["valid_token_indices_kv"],
            enabled=checkpoint_post_attn,
        )

        return x, y


@torch.compile(disable=not COMPILE_MMDIT_BLOCK)
class AsymmetricJointBlock(nn.Module):
    def __init__(
        self,
        hidden_size_x: int,
        hidden_size_y: int,
        num_heads: int,
        *,
        mlp_ratio_x: float = 8.0,  # Ratio of hidden size to d_model for MLP for visual tokens.
        mlp_ratio_y: float = 4.0,  # Ratio of hidden size to d_model for MLP for text tokens.
        update_y: bool = True,  # Whether to update text tokens in this block.
        device: Optional[torch.device] = None,
        **block_kwargs,
    ):
        super().__init__()
        self.update_y = update_y
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.mod_x = nn.Linear(hidden_size_x, 4 * hidden_size_x, device=device)
        if self.update_y:
            self.mod_y = nn.Linear(hidden_size_x, 4 * hidden_size_y, device=device)
        else:
            self.mod_y = nn.Linear(hidden_size_x, hidden_size_y, device=device)

        # Self-attention:
        self.attn = AsymmetricAttention(
            hidden_size_x,
            hidden_size_y,
            num_heads=num_heads,
            update_y=update_y,
            device=device,
            **block_kwargs,
        )

        # MLP.
        mlp_hidden_dim_x = int(hidden_size_x * mlp_ratio_x)
        assert mlp_hidden_dim_x == int(1536 * 8)
        self.mlp_x = FeedForward(
            in_features=hidden_size_x,
            hidden_size=mlp_hidden_dim_x,
            multiple_of=256,
            ffn_dim_multiplier=None,
            device=device,
        )

        # MLP for text not needed in last block.
        if self.update_y:
            mlp_hidden_dim_y = int(hidden_size_y * mlp_ratio_y)
            self.mlp_y = FeedForward(
                in_features=hidden_size_y,
                hidden_size=mlp_hidden_dim_y,
                multiple_of=256,
                ffn_dim_multiplier=None,
                device=device,
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        y: torch.Tensor,
        perturb_mode: str,
        # TODO: These could probably just go into attn_kwargs
        checkpoint_ff: bool = False,
        checkpoint_qkv: bool = False,
        checkpoint_post_attn: bool = False,
        **attn_kwargs,
    ):
        """Forward pass of a block.

        Args:
            x: (B, N, dim) tensor of visual tokens
            c: (B, dim) tensor of conditioned features
            y: (B, L, dim) tensor of text tokens
            num_frames: Number of frames in the video. N = num_frames * num_spatial_tokens

        Returns:
            x: (B, N, dim) tensor of visual tokens after block
            y: (B, L, dim) tensor of text tokens after block
        """
        if perturb_mode == "STG-R":
            return x, y
        N = x.size(1)

        c = F.silu(c)
        mod_x = self.mod_x(c)
        scale_msa_x, gate_msa_x, scale_mlp_x, gate_mlp_x = mod_x.chunk(4, dim=1)
        mod_y = self.mod_y(c)

        if self.update_y:
            scale_msa_y, gate_msa_y, scale_mlp_y, gate_mlp_y = mod_y.chunk(4, dim=1)
        else:
            scale_msa_y = mod_y

        # Self-attention block.
        x_attn, y_attn = self.attn(
            x,
            y,
            scale_x=scale_msa_x,
            scale_y=scale_msa_y,
            checkpoint_qkv=checkpoint_qkv,
            checkpoint_post_attn=checkpoint_post_attn,
            **attn_kwargs,
        )

        assert x_attn.size(1) == N
        x = residual_tanh_gated_rmsnorm(x, x_attn, gate_msa_x)

        if self.update_y:
            y = residual_tanh_gated_rmsnorm(y, y_attn, gate_msa_y)

        # MLP block.
        x = ck(self.ff_block_x, x, scale_mlp_x, gate_mlp_x, enabled=checkpoint_ff)
        if self.update_y:
            y = ck(self.ff_block_y, y, scale_mlp_y, gate_mlp_y, enabled=checkpoint_ff)  # type: ignore
        return x, y

    def ff_block_x(self, x, scale_x, gate_x):
        x_mod = modulated_rmsnorm(x, scale_x)
        x_res = self.mlp_x(x_mod)
        x = residual_tanh_gated_rmsnorm(x, x_res, gate_x)  # Sandwich norm
        return x

    def ff_block_y(self, y, scale_y, gate_y):
        y_mod = modulated_rmsnorm(y, scale_y)
        y_res = self.mlp_y(y_mod)
        y = residual_tanh_gated_rmsnorm(y, y_res, gate_y)  # Sandwich norm
        return y


@torch.compile(disable=not COMPILE_FINAL_LAYER)
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, device=device)
        self.mod = nn.Linear(hidden_size, 2 * hidden_size, device=device)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, device=device)

    def forward(self, x, c):
        c = F.silu(c)
        shift, scale = self.mod(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class AsymmDiTJoint(nn.Module):
    """
    Diffusion model with a Transformer backbone.

    Ingests text embeddings instead of a label.
    """

    def __init__(
        self,
        *,
        patch_size=2,
        in_channels=4,
        hidden_size_x=1152,
        hidden_size_y=1152,
        depth=48,
        num_heads=16,
        mlp_ratio_x=8.0,
        mlp_ratio_y=4.0,
        t5_feat_dim: int = 4096,
        t5_token_length: int = 256,
        patch_embed_bias: bool = True,
        timestep_mlp_bias: bool = True,
        timestep_scale: Optional[float] = None,
        use_extended_posenc: bool = False,
        rope_theta: float = 10000.0,
        device: Optional[torch.device] = None,
        audio_mode: str = None,
        audio_cross_attn_layers: List[int] = [6, 7, 8, 9, 10, 21, 34, 35, 36, 38, 39, 43, 44, 45, 46, 47],
        **block_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.head_dim = hidden_size_x // num_heads  # Head dimension and count is determined by visual.
        self.use_extended_posenc = use_extended_posenc
        self.t5_token_length = t5_token_length
        self.t5_feat_dim = t5_feat_dim
        self.rope_theta = rope_theta  # Scaling factor for frequency computation for temporal RoPE.

        self.x_embedder = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size_x,
            bias=patch_embed_bias,
            device=device,
        )
        # Conditionings
        # Timestep
        self.t_embedder = TimestepEmbedder(hidden_size_x, bias=timestep_mlp_bias, timestep_scale=timestep_scale)

        # Caption Pooling (T5)
        self.t5_y_embedder = AttentionPool(t5_feat_dim, num_heads=8, output_dim=hidden_size_x, device=device)

        # Dense Embedding Projection (T5)
        self.t5_yproj = nn.Linear(t5_feat_dim, hidden_size_y, bias=True, device=device)

        # Initialize pos_frequencies as an empty parameter.
        self.pos_frequencies = nn.Parameter(torch.empty(3, self.num_heads, self.head_dim // 2, device=device))

        # for depth 48:
        #  b =  0: AsymmetricJointBlock, update_y=True
        #  b =  1: AsymmetricJointBlock, update_y=True
        #  ...
        #  b = 46: AsymmetricJointBlock, update_y=True
        #  b = 47: AsymmetricJointBlock, update_y=False. No need to update text features.
        blocks = []
        for b in range(depth):
            # Joint multi-modal block
            update_y = b < depth - 1
            block = AsymmetricJointBlock(
                hidden_size_x,
                hidden_size_y,
                num_heads,
                mlp_ratio_x=mlp_ratio_x,
                mlp_ratio_y=mlp_ratio_y,
                update_y=update_y,
                device=device,
                **block_kwargs,
            )

            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.audio_mode = audio_mode
        self.enable_audio = audio_mode is not None
        if self.enable_audio:
            self.audio_cross_attn_blocks = None
            if self.audio_mode == 'cross_attn':
                self.audio_cross_attn_layers = audio_cross_attn_layers
                self.hidden_size_audio = 768
                self.audio_projection = AudioProjector(
                    self.hidden_size_audio,
                    hidden_size_x,
                    hidden_size_x,
                    zero_init=False
                )
                audio_cross_attn_blocks = []
                for i in range(len(self.blocks)):
                    if i in self.audio_cross_attn_layers:
                        audio_ca = AudioCrossAttention(
                            hidden_size_x,
                        )
                        audio_cross_attn_blocks.append(audio_ca)
                print("Num of cross attn layers:", len(audio_cross_attn_blocks))
                self.audio_cross_attn_blocks = nn.ModuleList(audio_cross_attn_blocks)
            else:
                raise NotImplementedError(f"Audio conditioning mode {self.audio_mode} is not implemented")

        self.final_layer = FinalLayer(hidden_size_x, patch_size, self.out_channels, device=device)

    def embed_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C=12, T, H, W) tensor of visual tokens

        Returns:
            x: (B, C=3072, N) tensor of visual tokens with positional embedding.
        """
        return self.x_embedder(x)  # Convert BcTHW to BCN

    @torch.compile(disable=not COMPILE_MMDIT_BLOCK)
    def prepare(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        t5_feat: torch.Tensor,
        t5_mask: torch.Tensor,
        audio_feat: torch.Tensor,
    ):
        """Prepare input and conditioning embeddings."""

        # Visual patch embeddings with positional encoding.
        T, H, W = x.shape[-3:]
        pH, pW = H // self.patch_size, W // self.patch_size
        x = self.embed_x(x)  # (B, N, D), where N = T * H * W / patch_size ** 2
        assert x.ndim == 3
        B = x.size(0)

        # Construct position array of size [N, 3].
        # pos[:, 0] is the frame index for each location,
        # pos[:, 1] is the row index for each location, and
        # pos[:, 2] is the column index for each location.
        N = T * pH * pW
        assert x.size(1) == N
        pos = create_position_matrix(T, pH=pH, pW=pW, device=x.device, dtype=torch.float32)  # (N, 3)
        rope_cos, rope_sin = compute_mixed_rotation(
            freqs=self.pos_frequencies, pos=pos
        )  # Each are (N, num_heads, dim // 2)

        if self.enable_audio:
            audio_feat = self.audio_projection(audio_feat, T)
        else:
            audio_feat = None

        # Global vector embedding for conditionings.
        c_t = self.t_embedder(1 - sigma)  # (B, D)

        # Pool T5 tokens using attention pooler
        # Note y_feat[1] contains T5 token features.
        assert (
            t5_feat.size(1) == self.t5_token_length
        ), f"Expected L={self.t5_token_length}, got {t5_feat.shape} for y_feat."
        t5_y_pool = self.t5_y_embedder(t5_feat, t5_mask)  # (B, D)
        assert t5_y_pool.size(0) == B, f"Expected B={B}, got {t5_y_pool.shape} for t5_y_pool."

        c = c_t + t5_y_pool

        y_feat = self.t5_yproj(t5_feat)  # (B, L, t5_feat_dim) --> (B, L, D)

        return x, c, y_feat, audio_feat, rope_cos, rope_sin

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        y_feat: List[torch.Tensor],
        y_mask: List[torch.Tensor],
        packed_indices: Dict[str, torch.Tensor] = None,
        rope_cos: torch.Tensor = None,
        rope_sin: torch.Tensor = None,
        num_ff_checkpoint: int = 0,
        num_qkv_checkpoint: int = 0,
        num_post_attn_checkpoint: int = 0,
        audio_feat: torch.Tensor = None,
        stg_block_idx: List[int] = None,
    ):
        """Forward pass of DiT.

        Args:
            x: (B, C, T, H, W) tensor of spatial inputs (images or latent representations of images)
            sigma: (B,) tensor of noise standard deviations
            y_feat: List((B, L, y_feat_dim) tensor of caption token features. For SDXL text encoders: L=77, y_feat_dim=2048)
            y_mask: List((B, L) boolean tensor indicating which tokens are not padding)
            packed_indices: Dict with keys for Flash Attention. Result of compute_packed_indices.
        """
        _, _, T, H, W = x.shape

        is_perturbed = stg_block_idx is not None

        if self.pos_frequencies.dtype != torch.float32:
            warnings.warn(f"pos_frequencies dtype {self.pos_frequencies.dtype} != torch.float32")

        # Use EFFICIENT_ATTENTION backend for T5 pooling, since we have a mask.
        # Have to call sdpa_kernel outside of a torch.compile region.
        with sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
            x, c, y_feat, audio_feat, rope_cos, rope_sin = self.prepare(x, sigma, y_feat[0], y_mask[0], audio_feat)
        del y_mask

        cp_rank, cp_size = cp.get_cp_rank_size()
        N = x.size(1)
        M = N // cp_size
        assert N % cp_size == 0, f"Visual sequence length ({x.shape[1]}) must be divisible by cp_size ({cp_size})."

        if cp_size > 1:
            x = x.narrow(1, cp_rank * M, M)

            assert self.num_heads % cp_size == 0
            local_heads = self.num_heads // cp_size
            rope_cos = rope_cos.narrow(1, cp_rank * local_heads, local_heads)
            rope_sin = rope_sin.narrow(1, cp_rank * local_heads, local_heads)

        if is_perturbed:
            if isinstance(stg_block_idx, list):
                perturb_mode = "STG-R"
            else:
                raise TypeError("stg_block_idx must be a list")
        else:
            perturb_mode = "None"

        for i, block in enumerate(self.blocks):
            
            if is_perturbed and i in stg_block_idx and perturb_mode == "STG-R":

                x, y_feat = block(
                    x,
                    c,
                    y_feat,
                    perturb_mode=perturb_mode,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    packed_indices=packed_indices,
                    checkpoint_ff=i < num_ff_checkpoint,
                    checkpoint_qkv=i < num_qkv_checkpoint,
                    checkpoint_post_attn=i < num_post_attn_checkpoint,
                )  # (B, M, D), (B, L, D)
            
            else:

                x, y_feat = block(
                    x,
                    c,
                    y_feat,
                    perturb_mode="None",
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    packed_indices=packed_indices,
                    checkpoint_ff=i < num_ff_checkpoint,
                    checkpoint_qkv=i < num_qkv_checkpoint,
                    checkpoint_post_attn=i < num_post_attn_checkpoint,
                )  # (B, M, D), (B, L, D)

            if self.enable_audio:
                if self.audio_mode == "cross_attn":
                    if i in self.audio_cross_attn_layers:
                        x = self.audio_cross_attn_blocks[self.audio_cross_attn_layers.index(i)](x, audio_feat)

        del y_feat  # Final layers don't use dense text features.

        x = self.final_layer(x, c)  # (B, M, patch_size ** 2 * out_channels)

        patch = x.size(2)
        x = cp.all_gather(x)
        x = rearrange(x, "(G B) M P -> B (G M) P", G=cp_size, P=patch)
        x = rearrange(
            x,
            "B (T hp wp) (p1 p2 c) -> B c T (hp p1) (wp p2)",
            T=T,
            hp=H // self.patch_size,
            wp=W // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels,
        )

        return x
