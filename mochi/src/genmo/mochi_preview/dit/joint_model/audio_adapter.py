from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from genmo.lib.attn_imports import sdpa_attn_ctx


class AudioProjector(torch.nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        zero_init: bool = False,
    ):
        super().__init__()

        if hidden_dim == -1:
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(input_dim, output_dim, bias=bias, device=device),
            )
            if zero_init:
                torch.nn.init.zeros_(self.projection[0].weight)
                torch.nn.init.zeros_(self.projection[0].bias)
        else:
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim, bias=bias, device=device),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_dim, output_dim, bias=bias, device=device),
            )
            if zero_init:
                torch.nn.init.zeros_(self.projection[2].weight)
                torch.nn.init.zeros_(self.projection[2].bias)

    def forward(self, audio_embed: torch.Tensor, num_frames: int) -> torch.Tensor:
        B, T_audio, D_audio = audio_embed.shape
        
        projected = self.projection(audio_embed)
        
        if T_audio != num_frames:
            projected = F.interpolate(
                projected.transpose(1, 2),
                size=num_frames,
                mode='nearest',
            ).transpose(1, 2)
            
        return projected
    
    def get_state_dict(self):
        return {
            f"audio_projection.{key}": value 
            for key, value in self.state_dict().items()
        }


class AudioCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        device: torch.device = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias, device=device)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias, device=device)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias, device=device)
        
        self.to_out = nn.Linear(dim, dim, bias=qkv_bias, device=device)
        nn.init.zeros_(self.to_out.weight)
        if qkv_bias:
            nn.init.zeros_(self.to_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        audio_feat: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape
        B_a, T, D_a = audio_feat.shape
        assert D == D_a, f"Dimension mismatch: {D} != {D_a}"
        assert B == B_a or B_a == 1, f"Batch size mismatch: {B} != {B_a}"
        
        q = self.to_q(x)
        k = self.to_k(audio_feat)
        v = self.to_v(audio_feat)
        
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B_a, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B_a, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        with sdpa_attn_ctx(training=self.training):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
        del q, k, v, audio_feat

        out = out.transpose(1, 2).reshape(B, N, D)

        out = self.to_out(out)
        
        return x + out

    def get_state_dict(self):
        return {
            f"audio_cross_attn.{key}": value 
            for key, value in self.state_dict().items()
        }


def mark_only_audio_as_trainable(model: nn.Module, bias: str = "none") -> None:
    assert bias == "none", f"Only bias='none' is supported"
    for n, p in model.named_parameters():
        if "audio_" not in n:
            p.requires_grad = False


def mark_audio_and_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    assert bias == "none", f"Only bias='none' is supported"
    for n, p in model.named_parameters():
        if "lora_" not in n and "audio_" not in n:
            p.requires_grad = False