from __future__ import annotations

import logging
import math
import struct
import subprocess
import threading
import zlib
from pathlib import Path

import httpx
import numpy as np
import parselmouth
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import remove_weight_norm, weight_norm
from transformers import HubertModel

from ..media.ffmpeg import ffmpeg_bin

log = logging.getLogger(__name__)

ADAM_RVC_ZIP_URL = "https://huggingface.co/Coolwowsocoolwow/Adam_Elevenlabs/resolve/main/added_Adam_Elevenlabs_v2.zip"
ADAM_PTH_ZIP_OFFSET = 246269010
CONTENTVEC_REPO = "lengyue233/content-vec-best"
DEFAULT_TARGET_F0_HZ = 141.2  # Measured median F0 from bao.wav


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def _sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(length.max().item())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def _convert_pad_shape(pad_shape: list[list[int]]) -> list[int]:
    l = pad_shape[::-1]
    return [item for sublist in l for item in sublist]


@torch.jit.script
def _fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: torch.Tensor
) -> torch.Tensor:
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act


class _LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class _MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int | None = None,
        heads_share: bool = True,
    ):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.heads_share = heads_share
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x = self._attention(q, k, v, mask=attn_mask)
        return self.conv_o(x)

    def _attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, d, t_s = key.size()
        t_t = query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = torch.matmul(query / math.sqrt(self.k_channels), key_relative_embeddings.unsqueeze(0).transpose(-2, -1))
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + torch.matmul(relative_weights, value_relative_embeddings.unsqueeze(0))
        return output.transpose(2, 3).contiguous().view(b, d, t_t)

    def _get_relative_embeddings(self, relative_embeddings: torch.Tensor, length: int) -> torch.Tensor:
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded = F.pad(relative_embeddings, _convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        else:
            padded = relative_embeddings
        return padded[:, slice_start_position:slice_end_position]

    def _relative_position_to_absolute_position(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = F.pad(x, _convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, _convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1 :]
        return x_final

    def _absolute_position_to_relative_position(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = F.pad(x, _convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        x_flat = F.pad(x_flat, _convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final


class _FFN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, filter_channels: int, kernel_size: int, p_dropout: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(self._same_pad(x * x_mask))
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self._same_pad(x * x_mask))
        return x * x_mask

    def _same_pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        return F.pad(x, _convert_pad_shape([[0, 0], [0, 0], [pad_l, pad_r]]))


class _Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(
                _MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size)
            )
            self.norm_layers_1.append(_LayerNorm(hidden_channels))
            self.ffn_layers.append(_FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(_LayerNorm(hidden_channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        return x * x_mask


class _TextEncoder768(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.emb_phone = nn.Linear(768, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.emb_pitch = nn.Embedding(256, hidden_channels)
        self.encoder = _Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone: torch.Tensor, pitch: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(x.shape[-1])
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(_sequence_mask(lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class _WN(nn.Module):
    def __init__(self, hidden_channels: int, kernel_size: int, dilation_rate: int, n_layers: int, gin_channels: int = 0, p_dropout: float = 0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)
        if gin_channels != 0:
            self.cond_layer = weight_norm(nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1), name="weight")
        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            self.in_layers.append(
                weight_norm(nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding), name="weight")
            )
            res_skip_channels = hidden_channels if i == n_layers - 1 else 2 * hidden_channels
            self.res_skip_layers.append(weight_norm(nn.Conv1d(hidden_channels, res_skip_channels, 1), name="weight"))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        g = self.cond_layer(g) if g is not None else None
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)
            acts = _fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self) -> None:
        if hasattr(self, "cond_layer"):
            remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            remove_weight_norm(l)
        for l in self.res_skip_layers:
            remove_weight_norm(l)


class _Flip(nn.Module):
    def forward(self, x: torch.Tensor, *args, reverse: bool = False, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = torch.flip(x, [1])
        if not reverse:
            return x, torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        return x, None


class _ResidualCouplingLayer(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int, dilation_rate: int, n_layers: int, p_dropout: float = 0, gin_channels: int = 0):
        super().__init__()
        self.half_channels = channels // 2
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = _WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        m = self.post(h) * x_mask
        x1 = (x1 + m) * x_mask if not reverse else (x1 - m) * x_mask
        return torch.cat([x0, x1], 1), None

    def remove_weight_norm(self) -> None:
        self.enc.remove_weight_norm()


class _ResidualCouplingBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int, dilation_rate: int, n_layers: int, n_flows: int = 4, gin_channels: int = 0):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(_ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels))
            self.flows.append(_Flip())

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None, reverse: bool = False) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self) -> None:
        for i in range(0, len(self.flows), 2):
            self.flows[i].remove_weight_norm()


class _ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int, int, int] = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=_get_padding(kernel_size, dilation[0]))),
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=_get_padding(kernel_size, dilation[1]))),
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=_get_padding(kernel_size, dilation[2]))),
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=_get_padding(kernel_size, 1))),
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=_get_padding(kernel_size, 1))),
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=_get_padding(kernel_size, 1))),
            ]
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        return x * x_mask if x_mask is not None else x

    def remove_weight_norm(self) -> None:
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class _SineGen(nn.Module):
    def __init__(self, samp_rate: int, harmonic_num: int = 0, sine_amp: float = 0.1, noise_std: float = 0.003, voiced_threshold: float = 0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(f0) * (f0 > self.voiced_threshold)

    def forward(self, f0: torch.Tensor, upp: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            f0 = f0[:, None].transpose(1, 2)
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in range(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
            rad_values = (f0_buf / float(self.sampling_rate)) % 1
            rand_ini = torch.rand(f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device)
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
            tmp_over_one = torch.cumsum(rad_values, 1)
            tmp_over_one *= upp
            tmp_over_one = F.interpolate(tmp_over_one.transpose(2, 1), scale_factor=float(upp), mode="linear", align_corners=True).transpose(2, 1)
            rad_values = F.interpolate(rad_values.transpose(2, 1), scale_factor=float(upp), mode="nearest").transpose(2, 1)
            tmp_over_one %= 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
            sine_waves = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * math.pi)
            sine_waves = sine_waves * self.sine_amp
            uv = self._f02uv(f0)
            uv = F.interpolate(uv.transpose(2, 1), scale_factor=float(upp), mode="nearest").transpose(2, 1)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class _SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate: int, harmonic_num: int = 0, sine_amp: float = 0.1, add_noise_std: float = 0.003, voiced_threshod: float = 0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = _SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, upp: int = 1) -> tuple[torch.Tensor, None, None]:
        sine_wavs, _uv, _ = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


class _GeneratorNSF(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int,
        sr: int,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.f0_upsamp = nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = _SourceModuleHnNSF(sampling_rate=sr, harmonic_num=0)
        self.noise_convs = nn.ModuleList()
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(weight_norm(nn.ConvTranspose1d(upsample_initial_channel // (2**i), c_cur, k, u, padding=(k - u) // 2)))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
            else:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(_ResBlock1(ch, k, tuple(d)))
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
        self.upp = math.prod(upsample_rates)

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self) -> None:
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class _SynthesizerTrnMs768NSFsid(nn.Module):
    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
    ):
        super().__init__()
        self.enc_p = _TextEncoder768(inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout))
        self.dec = _GeneratorNSF(
            inter_channels,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
        )
        self.flow = _ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def remove_weight_norm(self) -> None:
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()

    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        nsff0: torch.Tensor,
        sid: torch.Tensor,
        noise_scale: float = 0.66666,
    ) -> torch.Tensor:
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, g=g)
        return o


class _HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def _mr_stft_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multi-resolution log-STFT magnitude + spectral convergence loss."""
    loss = torch.tensor(0.0, device=y.device)
    for n_fft in (512, 1024, 2048):
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=y.device)
        s_hat = torch.stft(y_hat, n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True).abs() + 1e-6
        s_ref = torch.stft(y, n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True).abs() + 1e-6
        sc = torch.norm(s_ref - s_hat, p="fro") / (torch.norm(s_ref, p="fro") + 1e-6)
        mag = F.l1_loss(torch.log(s_hat), torch.log(s_ref))
        loss = loss + sc + mag
    return loss


class AdamRVCConverter:
    """Singleton RVC v2 voice converter adapted on assets/voices/bao.wav."""

    _lock = threading.Lock()
    _instance: AdamRVCConverter | None = None

    def __init__(self, pth_path: Path):
        log.info("Loading ContentVec (%s) and RVC Adam (%s)...", CONTENTVEC_REPO, pth_path)
        self.hubert = _HubertModelWithFinalProj.from_pretrained(CONTENTVEC_REPO)
        self.hubert.eval()

        cpt = torch.load(pth_path, map_location="cpu")
        cfg = cpt["config"]
        self.sr = int(cpt.get("sr", 32000))
        self.net_g = _SynthesizerTrnMs768NSFsid(*cfg)
        weights_f32 = {k: v.float() for k, v in cpt["weight"].items()}
        self.net_g.load_state_dict(weights_f32, strict=True)
        self.net_g.eval()
        self.net_g.remove_weight_norm()

        self.target_f0_hz: float = DEFAULT_TARGET_F0_HZ
        self.index_bank: torch.Tensor | None = None  # [N, 768] L2-normalized ContentVec vectors from bao.wav
        self.index_raw: torch.Tensor | None = None  # [N, 768] raw ContentVec vectors from bao.wav
        self.eq_curve: np.ndarray | None = None  # [1025] frequency-domain spectral transfer curve from bao.wav

        self._load_or_learn_bao()

    @classmethod
    def get(cls) -> AdamRVCConverter:
        with cls._lock:
            if cls._instance is None:
                pth = cls._ensure_pth()
                cls._instance = cls(pth)
            return cls._instance

    @staticmethod
    def _models_dir() -> Path:
        d = Path(__file__).resolve().parents[2] / "assets" / "models" / "adam_rvc"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def _ensure_pth(cls) -> Path:
        pth_path = cls._models_dir() / "Adam_Elevenlabs.pth"
        if pth_path.exists() and pth_path.stat().st_size > 40_000_000:
            return pth_path
        log.info("Downloading Adam_Elevenlabs.pth from HuggingFace...")
        h = httpx.head(ADAM_RVC_ZIP_URL, follow_redirects=True, timeout=30.0)
        size = int(h.headers["content-length"])
        r = httpx.get(
            ADAM_RVC_ZIP_URL,
            headers={"Range": f"bytes={ADAM_PTH_ZIP_OFFSET}-{size - 1}"},
            follow_redirects=True,
            timeout=120.0,
        )
        r.raise_for_status()
        buf = r.content
        fname_len, extra_len = struct.unpack_from("<HH", buf, 26)
        data_start = 30 + fname_len + extra_len
        d = zlib.decompressobj(-zlib.MAX_WBITS)
        raw_pth = d.decompress(buf[data_start:]) + d.flush()
        pth_path.write_bytes(raw_pth)
        return pth_path

    def _load_or_learn_bao(self) -> None:
        bao_wav = Path(__file__).resolve().parents[2] / "assets" / "voices" / "bao.wav"
        learned_pt = self._models_dir() / "bao_learned.pt"
        if learned_pt.exists():
            data = torch.load(learned_pt, map_location="cpu")
            if not bao_wav.exists() or data.get("bao_mtime") == int(bao_wav.stat().st_mtime):
                self.net_g.load_state_dict(data["net_g_state"], strict=True)
                self.net_g.eval()
                self.target_f0_hz = float(data["target_f0_hz"])
                self.index_raw = data["index_raw"].float()
                self.index_bank = F.normalize(self.index_raw, p=2, dim=-1)
                self.eq_curve = data["eq_curve"].numpy()
                log.info("Loaded learned bao.wav RVC profile (%d index frames, F0=%.1fHz)", len(self.index_raw), self.target_f0_hz)
                return

        if bao_wav.exists():
            self.learn_from_reference(bao_wav, learned_pt)

    def learn_from_reference(self, ref_wav: Path, save_pt: Path, steps: int = 36) -> None:
        """Learn speaker features, fine-tune RVC generator & spectral envelope from bao.wav."""
        log.info("Learning RVC voice profile from %s...", ref_wav)
        raw16 = subprocess.check_output(
            [ffmpeg_bin(), "-v", "error", "-i", str(ref_wav), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
        )
        x16 = np.frombuffer(raw16, dtype=np.int16).astype(np.float32) / 32768.0
        raw32 = subprocess.check_output(
            [ffmpeg_bin(), "-v", "error", "-i", str(ref_wav), "-ac", "1", "-ar", "32000", "-f", "s16le", "-"]
        )
        x32 = np.frombuffer(raw32, dtype=np.int16).astype(np.float32) / 32768.0

        # Extract ContentVec features in 15s chunks to keep memory small
        feats_list = []
        chunk_16k = 16000 * 15
        with torch.no_grad():
            for start in range(0, len(x16), chunk_16k):
                seg = x16[start : start + chunk_16k]
                if len(seg) < 1600:
                    break
                f = self.hubert(torch.from_numpy(seg).unsqueeze(0))["last_hidden_state"]
                f = F.interpolate(f.permute(0, 2, 1), scale_factor=2.0, mode="nearest").permute(0, 2, 1)
                feats_list.append(f[0])
        all_feats = torch.cat(feats_list, dim=0)

        p_len = min(len(x16) // 160, len( x32 ) // 320, all_feats.shape[0])
        all_feats = all_feats[:p_len]
        x32 = x32[: p_len * 320]

        # Extract exact F0 from bao.wav
        snd = parselmouth.Sound(x16[: p_len * 160], sampling_frequency=16000)
        f0 = snd.to_pitch_ac(time_step=0.01, voicing_threshold=0.55, pitch_floor=50.0, pitch_ceiling=500.0).selected_array["frequency"]
        if len(f0) < p_len:
            f0 = np.pad(f0, (0, p_len - len(f0)), mode="constant")
        else:
            f0 = f0[:p_len]

        voiced_mask = f0 > 0
        self.target_f0_hz = float(np.median(f0[voiced_mask])) if np.any(voiced_mask) else DEFAULT_TARGET_F0_HZ

        # Build feature retrieval index from voiced frames of bao.wav
        voiced_idx = np.where(voiced_mask)[0]
        self.index_raw = all_feats[voiced_idx].detach().clone()
        self.index_bank = F.normalize(self.index_raw, p=2, dim=-1)

        pitch, pitchf = self._f0_to_tensors(f0)

        # Fine-tune speaker embedding, conditioning, flow & decoder layers on bao.wav segments
        trainable_params = []
        for name, param in self.net_g.named_parameters():
            if any(k in name for k in ("emb_g", "dec.cond", "dec.conv_pre", "dec.conv_post", "enc_p.proj", "flow")):
                param.requires_grad_(True)
                trainable_params.append(param)
            else:
                param.requires_grad_(False)

        opt = torch.optim.AdamW(trainable_params, lr=2e-4, weight_decay=1e-4)
        seg_frames = 180  # 1.8s windows (180 frames = 57,600 samples at 32kHz)
        valid_starts = [i for i in range(0, p_len - seg_frames, 40) if np.mean(voiced_mask[i : i + seg_frames]) > 0.6]
        if not valid_starts:
            valid_starts = [0]

        self.net_g.train()
        rng = np.random.default_rng(42)
        for step in range(steps):
            s = int(rng.choice(valid_starts))
            e = s + seg_frames
            f_seg = all_feats[s:e].unsqueeze(0)
            p_seg = pitch[:, s:e]
            pf_seg = pitchf[:, s:e]
            y_ref = torch.from_numpy(x32[s * 320 : e * 320]).unsqueeze(0)
            sid = torch.LongTensor([0])
            plen = torch.LongTensor([seg_frames])

            opt.zero_grad()
            y_hat = self.net_g.infer(f_seg, plen, p_seg, pf_seg, sid, noise_scale=0.2)[0]
            loss = _mr_stft_loss(y_hat, y_ref)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            if (step + 1) % 12 == 0 or step == 0:
                log.info("RVC bao.wav adaptation step %d/%d: stft_loss=%.4f", step + 1, steps, float(loss.item()))

        for param in self.net_g.parameters():
            param.requires_grad_(False)
        self.net_g.eval()

        # Compute spectral matching curve (LTAS) across voiced segments so timbre/EQ matches bao.wav 1:1
        n_fft = 2048
        win = np.hanning(n_fft).astype(np.float32)
        ref_spec = np.zeros(n_fft // 2 + 1, dtype=np.float64)
        gen_spec = np.zeros(n_fft // 2 + 1, dtype=np.float64)
        with torch.no_grad():
            for s in valid_starts[:12]:
                e = s + seg_frames
                f_seg = all_feats[s:e].unsqueeze(0)
                p_seg = pitch[:, s:e]
                pf_seg = pitchf[:, s:e]
                y_gen = self.net_g.infer(f_seg, torch.LongTensor([seg_frames]), p_seg, pf_seg, torch.LongTensor([0]), noise_scale=0.0)[0, 0].cpu().numpy()
                y_true = x32[s * 320 : e * 320]
                for pos in range(0, len(y_true) - n_fft, n_fft // 2):
                    if np.sqrt(np.mean(y_true[pos : pos + n_fft] ** 2)) < 0.02:
                        continue
                    ref_spec += np.abs(np.fft.rfft(y_true[pos : pos + n_fft] * win))
                    gen_spec += np.abs(np.fft.rfft(y_gen[pos : pos + n_fft] * win))

        ratio = (ref_spec + 1e-4) / (gen_spec + 1e-4)
        # Smooth in log domain and clamp to +/- 6dB (0.5x to 2.0x) to avoid ringing
        log_ratio = np.log(np.clip(ratio, 0.25, 4.0))
        kernel = np.ones(15) / 15.0
        log_ratio_smooth = np.convolve(np.pad(log_ratio, (7, 7), mode="edge"), kernel, mode="valid")
        # Blend 75% spectral transfer within 70Hz..12000Hz
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sr)
        weight = np.where((freqs >= 70.0) & (freqs <= 12000.0), 0.75, 0.0)
        eq_curve = np.exp(np.clip(log_ratio_smooth * weight, -0.7, 0.7)).astype(np.float32)
        self.eq_curve = eq_curve

        torch.save(
            {
                "bao_mtime": int(ref_wav.stat().st_mtime),
                "net_g_state": self.net_g.state_dict(),
                "target_f0_hz": self.target_f0_hz,
                "index_raw": self.index_raw.half(),
                "eq_curve": torch.from_numpy(self.eq_curve),
            },
            save_pt,
        )
        log.info("Saved learned bao.wav RVC model to %s", save_pt)

    def _retrieve_index(self, feats: torch.Tensor, index_rate: float = 0.65) -> torch.Tensor:
        """Blend ContentVec features with top-8 nearest neighbors from bao.wav feature bank."""
        if self.index_bank is None or self.index_raw is None or index_rate <= 0:
            return feats
        q = F.normalize(feats[0], p=2, dim=-1)  # [T, 768]
        sim = torch.matmul(q, self.index_bank.T)  # [T, N]
        top_vals, top_idx = torch.topk(sim, k=min(8, self.index_bank.shape[0]), dim=-1)
        weights = F.softmax(top_vals * 20.0, dim=-1).unsqueeze(-1)  # [T, k, 1]
        knn = (self.index_raw[top_idx] * weights).sum(dim=1).unsqueeze(0)  # [1, T, 768]
        return (1.0 - index_rate) * feats + index_rate * knn

    def _apply_eq_curve(self, wav: np.ndarray) -> np.ndarray:
        if self.eq_curve is None or len(wav) < 2048:
            return wav
        n_fft = 2048
        hop = 512
        win = np.hanning(n_fft).astype(np.float32)
        pad = n_fft
        x = np.pad(wav, (pad, pad), mode="reflect")
        out = np.zeros_like(x)
        norm = np.zeros_like(x)
        for i in range(0, len(x) - n_fft, hop):
            spec = np.fft.rfft(x[i : i + n_fft] * win) * self.eq_curve
            frame = np.fft.irfft(spec, n=n_fft).astype(np.float32)
            out[i : i + n_fft] += frame * win
            norm[i : i + n_fft] += win**2
        out = out / np.maximum(norm, 1e-6)
        return out[pad : pad + len(wav)]

    def convert_16k(self, audio_16k: np.ndarray) -> tuple[np.ndarray, int]:
        """Convert 16kHz mono float32 audio to learned bao.wav Adam voice at self.sr (32kHz)."""
        pad_len = 1280
        x = np.pad(audio_16k.astype(np.float32), (pad_len, pad_len), mode="constant")

        with torch.no_grad():
            feats = torch.from_numpy(x).unsqueeze(0)
            feats = self.hubert(feats)["last_hidden_state"]
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2.0, mode="nearest").permute(0, 2, 1)
            feats = self._retrieve_index(feats, index_rate=0.65)

            p_len = min(x.shape[0] // 160, feats.shape[1])
            feats = feats[:, :p_len, :]

            pitch, pitchf = self._extract_target_f0(x, p_len, self.target_f0_hz)
            p_len_t = torch.LongTensor([p_len])
            sid = torch.LongTensor([0])

            out = self.net_g.infer(feats, p_len_t, pitch, pitchf, sid, noise_scale=0.35)[0, 0].cpu().numpy()

        trim = int(pad_len * (self.sr / 16000.0))
        if out.shape[0] > 2 * trim:
            out = out[trim:-trim]
        out = self._apply_eq_curve(out)
        return out, self.sr

    @staticmethod
    def _f0_to_tensors(f0: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        f0_min = 50.0
        f0_max = 1100.0
        f0_mel_min = 1127.0 * math.log(1.0 + f0_min / 700.0)
        f0_mel_max = 1127.0 * math.log(1.0 + f0_max / 700.0)
        f0bak = f0.copy()
        f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int64)
        return torch.from_numpy(f0_coarse).unsqueeze(0), torch.from_numpy(f0bak.astype(np.float32)).unsqueeze(0)

    @classmethod
    def _extract_target_f0(cls, x: np.ndarray, p_len: int, target_f0_hz: float) -> tuple[torch.Tensor, torch.Tensor]:
        snd = parselmouth.Sound(x, sampling_frequency=16000)
        f0 = snd.to_pitch_ac(
            time_step=0.01,
            voicing_threshold=0.55,
            pitch_floor=50.0,
            pitch_ceiling=500.0,
        ).selected_array["frequency"]

        if len(f0) < p_len:
            pad_total = p_len - len(f0)
            lpad = pad_total // 2
            rpad = pad_total - lpad
            f0 = np.pad(f0, (lpad, rpad), mode="constant")
        else:
            f0 = f0[:p_len]

        # Shift median pitch to match bao.wav's natural median F0 (141.2Hz) while preserving full Vietnamese tonal contour
        voiced = f0 > 0
        if np.any(voiced):
            med = float(np.median(f0[voiced]))
            if med > 0:
                f0[voiced] = f0[voiced] * (target_f0_hz / med)

        return cls._f0_to_tensors(f0)
