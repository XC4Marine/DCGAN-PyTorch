"""Plot generated-real waveform and spectrum pairs from three NN-L2 ranges."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'wavegan_improve'))
import evaluate_wavegan_torch as shared_evaluator
from pytorch_wavegan import WaveGANGenerator


def nearest_real_indices(real_waveforms, generated_waveforms, device, query_batch_size=256):
  real = shared_evaluator.normalize_waveforms_to_unit_range(real_waveforms)
  generated = shared_evaluator.normalize_waveforms_to_unit_range(generated_waveforms)
  real_tensor = torch.as_tensor(real, dtype=torch.float32, device=device)
  real_norms = (real_tensor ** 2).sum(dim=1)
  all_indices, all_distances = [], []
  for start in range(0, generated.shape[0], query_batch_size):
    generated_tensor = torch.as_tensor(generated[start:start + query_batch_size], dtype=torch.float32, device=device)
    squared_distances = (
        (generated_tensor ** 2).sum(dim=1, keepdim=True) + real_norms.unsqueeze(0) -
        2.0 * generated_tensor.matmul(real_tensor.t()))
    distances, indices = torch.sqrt(squared_distances.clamp_min(0.0)).min(dim=1)
    all_indices.append(indices.cpu())
    all_distances.append(distances.cpu())
    completed = min(start + query_batch_size, generated.shape[0])
    filled = round(30 * completed / generated.shape[0])
    print('Overall progress: [{}] {:6.2f}% | nearest-neighbor matching'.format(
        '#' * filled + '-' * (30 - filled), 100 * completed / generated.shape[0]), flush=True)
  return torch.cat(all_indices).numpy(), torch.cat(all_distances).numpy()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path,
                      default=SCRIPT_DIR / 'checkpoints' / 'best' / 'representative_lowest_score.pt')
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  parser.add_argument('--output-dir', type=Path,
                      default=SCRIPT_DIR / 'images' / 'nn_l2_ranges')
  parser.add_argument('--seed', type=int, default=369)
  parser.add_argument('--candidate-count', type=int, default=16384)
  parser.add_argument('--sample-rate', type=int, default=576000)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  args = parser.parse_args()

  device_name = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device
  device = torch.device(device_name)
  torch.manual_seed(args.seed)
  if device.type == 'cuda':
    torch.cuda.manual_seed_all(args.seed)

  checkpoint = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
  checkpoint_args = checkpoint['args']
  generator = WaveGANGenerator(checkpoint_args['latent_dim'], dim=checkpoint_args['model_dim'],
                                kernel_len=checkpoint_args['kernel_len']).to(device)
  generator.load_state_dict(checkpoint['generator'])
  real_waveforms = shared_evaluator.load_real_waveforms(args.data_dir, args.sample_rate, 256, 0)
  generated_waveforms = shared_evaluator.generate_waveforms(
      generator, args.candidate_count, checkpoint_args['latent_dim'], 256, device)
  nearest_indices, distances = nearest_real_indices(real_waveforms, generated_waveforms, device)
  ranges = (
      ('NN-L2 < 0.5', distances < 0.5, 'nn_l2_lt_0_5_waveform_spectrum.png'),
      ('0.5 <= NN-L2 <= 1.0', (distances >= 0.5) & (distances <= 1.0),
       'nn_l2_0_5_to_1_0_waveform_spectrum.png'),
      ('NN-L2 > 1.0', distances > 1.0, 'nn_l2_gt_1_0_waveform_spectrum.png'),
  )
  args.output_dir.mkdir(parents=True, exist_ok=True)
  for title, mask, filename in ranges:
    selected = np.flatnonzero(mask)
    print('Candidates for {}: {}.'.format(title, selected.size))
    if selected.size == 0:
      raise RuntimeError('No candidates found for {}.'.format(title))
    selected = selected[np.argsort(distances[selected])]
    selected = selected[selected.size // 2]
    generated = generated_waveforms[selected]
    real = real_waveforms[nearest_indices[selected]]
    figure, axes = plt.subplots(2, 1, figsize=(8, 7))
    samples = np.arange(generated.size)
    frequencies_khz = np.fft.rfftfreq(generated.size, d=1.0 / args.sample_rate) / 1000.0
    waveform_axis = axes[0]
    waveform_axis.plot(samples, generated, color='#e76f51', linewidth=1.5, label='Generated')
    waveform_axis.plot(samples, real, color='#1769aa', linewidth=1.2, alpha=0.85, label='Nearest real')
    waveform_axis.set_title('{} | selected NN-L2 = {:.3f}'.format(title, distances[selected]))
    waveform_axis.set_xlabel('Sample index')
    waveform_axis.set_ylabel('Normalized amplitude')
    waveform_axis.grid(alpha=0.2)
    waveform_axis.legend(fontsize=9)

    spectrum_axis = axes[1]
    generated_power = np.abs(np.fft.rfft(generated)) ** 2 / generated.size
    real_power = np.abs(np.fft.rfft(real)) ** 2 / real.size
    spectrum_axis.semilogy(frequencies_khz, generated_power + 1e-12, color='#e76f51', linewidth=1.5,
                           label='Generated')
    spectrum_axis.semilogy(frequencies_khz, real_power + 1e-12, color='#1769aa', linewidth=1.2,
                           alpha=0.85, label='Nearest real')
    spectrum_axis.set_xlabel('Frequency (kHz)')
    spectrum_axis.set_ylabel('Power / bin')
    spectrum_axis.grid(alpha=0.2)
    spectrum_axis.legend(fontsize=9)
    figure.suptitle('Generated and nearest-real waveform and spectrum comparison', fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output = args.output_dir / filename
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print('Selected NN-L2={:.3f} for {}.'.format(distances[selected], title))
    print('Saved waveform and spectrum comparison to {}.'.format(output.resolve()))


if __name__ == '__main__':
  main()
