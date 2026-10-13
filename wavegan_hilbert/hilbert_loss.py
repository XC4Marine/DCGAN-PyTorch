"""Match each generated envelope to the mean envelope of the current real batch."""

import torch


def hilbert_envelope(waveforms):
  """Magnitude of the FFT analytic signal, differentiable along the last axis."""
  length = waveforms.size(-1)
  multiplier = waveforms.new_zeros(length)
  multiplier[0] = 1
  multiplier[1:(length + 1) // 2] = 2
  if length % 2 == 0:
    multiplier[length // 2] = 1
  analytic = torch.fft.ifft(torch.fft.fft(waveforms, dim=-1) * multiplier, dim=-1)
  return analytic.abs()


def hilbert_envelope_loss(real_waveforms, generated_waveforms):
  """MSE for each fake envelope, against the pointwise real-batch mean envelope."""
  with torch.no_grad():
    target = hilbert_envelope(real_waveforms).mean(dim=0, keepdim=True)
  return (hilbert_envelope(generated_waveforms) - target).square().mean()
