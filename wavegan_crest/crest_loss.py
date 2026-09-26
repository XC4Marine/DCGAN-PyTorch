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

    peak_distance = functional.l1_loss(fake_peak, real_peak)
    crest_distance = functional.l1_loss(
        fake_peak / (fake_rms + self.denominator_epsilon),
        real_peak / (real_rms + self.denominator_epsilon))
    return peak_distance + self.crest_weight * crest_distance
