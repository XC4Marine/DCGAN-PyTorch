"""Numerical sanity checks for the two reused auxiliary losses."""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'wavegan_crest'))
from crest_loss import CrestFactorLoss
sys.path.insert(0, str(PROJECT_ROOT / 'wavegan_PSDLoss'))
from global_spectral_loss import GlobalSpectralLoss


def main():
  audio = torch.randn(4, 1, 128, requires_grad=True)
  crest_loss = CrestFactorLoss()(audio, audio)
  psd_loss = GlobalSpectralLoss(128, 32)(audio, audio)
  assert torch.allclose(crest_loss, torch.zeros_like(crest_loss), atol=1e-6)
  assert torch.allclose(psd_loss, torch.zeros_like(psd_loss), atol=1e-6)
  target = torch.randn(4, 1, 128)
  loss = CrestFactorLoss()(target, audio) + GlobalSpectralLoss(128, 32)(target, audio)
  loss.backward()
  assert torch.isfinite(audio.grad).all()
  print('Crest and PSD loss sanity checks passed.')


if __name__ == '__main__':
  main()
