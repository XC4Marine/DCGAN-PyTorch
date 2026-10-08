"""Match batch distributions of peak amplitudes and crest factors by sorted L1."""

import torch
from torch import nn
from torch.nn import functional as functional


class SortedCrestFactorLoss(nn.Module):
  """Match peak and crest marginal distributions independently along the batch.

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
    if real_audio.ndim < 2:
      raise ValueError('Audio inputs must include batch and sample dimensions.')

    real_peak = torch.max(torch.abs(real_audio), dim=-1).values
    fake_peak = torch.max(torch.abs(fake_audio), dim=-1).values

    real_rms = torch.sqrt(torch.mean(real_audio.square(), dim=-1) + self.rms_epsilon)
    fake_rms = torch.sqrt(torch.mean(fake_audio.square(), dim=-1) + self.rms_epsilon)

    peak_distance = functional.l1_loss(
        fake_peak.sort(dim=0).values, real_peak.sort(dim=0).values)
    crest_distance = functional.l1_loss(
        (fake_peak / (fake_rms + self.denominator_epsilon)).sort(dim=0).values,
        (real_peak / (real_rms + self.denominator_epsilon)).sort(dim=0).values)
    return peak_distance + self.crest_weight * crest_distance
