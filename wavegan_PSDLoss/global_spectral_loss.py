"""Batch-level power spectral density loss for unpaired WaveGAN training."""

import torch
from torch import nn
from torch.nn import functional as functional


class GlobalSpectralLoss(nn.Module):
  """Match real and generated batches by their mean log power spectrum.

  Each waveform is transformed independently.  Averaging happens only across
  the batch, so the objective does not assume a real waveform is temporally
  aligned with the generated waveform at the same batch index.
  """

  def __init__(self, n_fft=128, hop_length=32, epsilon=1e-6):
    super().__init__()
    if n_fft <= 0:
      raise ValueError('n_fft must be positive.')
    if hop_length <= 0:
      raise ValueError('hop_length must be positive.')
    if epsilon <= 0:
      raise ValueError('epsilon must be positive.')
    self.n_fft = n_fft
    self.hop_length = hop_length
    self.epsilon = epsilon
    self.register_buffer('window', torch.hann_window(n_fft))

  def _mean_power_spectrum(self, audio):
    if audio.ndim != 3 or audio.size(1) != 1:
      raise ValueError('Expected audio with shape [batch, 1, samples], got {}.'.format(tuple(audio.shape)))
    if audio.size(-1) < self.n_fft:
      raise ValueError('n_fft {} exceeds waveform length {}.'.format(self.n_fft, audio.size(-1)))
    spectrum = torch.stft(
        audio[:, 0], n_fft=self.n_fft, hop_length=self.hop_length,
        window=self.window, center=False, return_complex=True)
    power = spectrum.abs().square()
    return power.mean(dim=(0, 2))

  def forward(self, real_audio, fake_audio):
    """Return L1 distance between batch-average real and fake log PSD curves."""
    real_global_psd = self._mean_power_spectrum(real_audio)
    fake_global_psd = self._mean_power_spectrum(fake_audio)
    return functional.l1_loss(
        torch.log(fake_global_psd + self.epsilon),
        torch.log(real_global_psd + self.epsilon))
