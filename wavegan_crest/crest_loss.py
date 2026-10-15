"""Crest-factor loss for waveform batches.

The loss matches per-waveform peak amplitudes and crest factors (peak / RMS)
between real and generated audio.  It is intentionally standalone so training
code can opt in without changing its existing objective.
"""

import torch
from torch import nn
from torch.nn import functional as functional


class CrestFactorLoss(nn.Module):
  """Match waveform peak amplitude and crest factor using L1 distances.

  Inputs may have shape ``(batch, samples)`` or ``(batch, channels, samples)``.
  The final dimension is treated as the temporal axis; leading dimensions are
  retained when calculating the batch-mean L1 losses.
  """

  def __init__(self, rms_epsilon=1e-7, denominator_epsilon=1e-5,
               crest_weight=0.5):
    super().__init__()
    self.rms_epsilon = rms_epsilon
    self.denominator_epsilon = denominator_epsilon
    self.crest_weight = crest_weight

  def forward(self, real_audio, fake_audio):
    """Return peak-amplitude loss plus weighted crest-factor loss."""
    if real_audio.shape != fake_audio.shape:
      raise ValueError(
          'real_audio and fake_audio must have identical shapes; got {} and {}.'.format(
              tuple(real_audio.shape), tuple(fake_audio.shape)))
    if real_audio.ndim < 1:
      raise ValueError('Audio inputs must include a sample dimension.')

    real_peak = torch.max(torch.abs(real_audio), dim=-1).values
    fake_peak = torch.max(torch.abs(fake_audio), dim=-1).values

    real_rms = torch.sqrt(torch.mean(real_audio.square(), dim=-1) + self.rms_epsilon)
    fake_rms = torch.sqrt(torch.mean(fake_audio.square(), dim=-1) + self.rms_epsilon)

    # 1. 先计算出具体的 Crest 比值
    real_crest = real_peak / (real_rms + self.denominator_epsilon)
    fake_crest = fake_peak / (fake_rms + self.denominator_epsilon)

    # 2. 【核心修改】沿着 Batch 维度 (dim=0) 从小到大排序，实现 1D-Wasserstein 分布对齐
    sorted_real_peak, _ = torch.sort(real_peak, dim=0)
    sorted_fake_peak, _ = torch.sort(fake_peak, dim=0)
    
    sorted_real_crest, _ = torch.sort(real_crest, dim=0)
    sorted_fake_crest, _ = torch.sort(fake_crest, dim=0)

    # 3. 计算排序后的 L1 距离 (消除随机配对导致的方差)
    peak_distance = functional.l1_loss(sorted_fake_peak, sorted_real_peak)
    crest_distance = functional.l1_loss(sorted_fake_crest, sorted_real_crest)
    
    return peak_distance + self.crest_weight * crest_distance