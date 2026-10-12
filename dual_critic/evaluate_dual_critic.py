"""Evaluate dual-critic WaveGAN fidelity and real-sample coverage."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'wavegan_improve'))
import evaluate_wavegan_torch as shared_evaluator
sys.path.insert(0, str(SCRIPT_DIR))
from train_dual_critic import ClickWaveformDataset
sys.path.insert(0, str(PROJECT_ROOT / 'wavegan_crest'))
from pytorch_wavegan import WaveGANGenerator


def extract_features(waveforms, sample_rate):
  features = shared_evaluator.extract_features(waveforms, sample_rate)
  power = np.abs(np.fft.rfft(waveforms, axis=1)) ** 2 / waveforms.shape[1]
  frequencies = np.fft.rfftfreq(waveforms.shape[1], d=1.0 / sample_rate)
  bandwidths = []
  for spectrum in power:
    peak_index = int(np.argmax(spectrum[1:])) + 1
    threshold = spectrum[peak_index] / 2.0
    left = peak_index
    right = peak_index
    while left > 1 and spectrum[left - 1] >= threshold:
      left -= 1
    while right < spectrum.size - 1 and spectrum[right + 1] >= threshold:
      right += 1
    bandwidths.append(frequencies[right] - frequencies[left])
  features['minus_3db_bandwidth_hz'] = np.asarray(bandwidths, dtype=np.float32)
  return features


def load_real_waveforms(data_dir, file_list, sample_rate, batch_size):
  dataset = ClickWaveformDataset(data_dir, sample_rate, file_list)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  return np.concatenate([batch[:, 0].numpy() for batch in loader], axis=0)


def feature_density_and_coverage(real_features, generated_features, feature_names, device,
                                 neighbors, batch_size):
  real = np.column_stack([real_features[name] for name in feature_names]).astype(np.float32)
  generated = np.column_stack([generated_features[name] for name in feature_names]).astype(np.float32)
  std = real.std(axis=0)
  valid = std > 0
  feature_names = [name for name, keep in zip(feature_names, valid) if keep]
  real = real[:, valid]
  generated = generated[:, valid]
  mean = real.mean(axis=0, keepdims=True)
  std = real.std(axis=0, keepdims=True)
  real = (real - mean) / std
  generated = (generated - mean) / std
  real_tensor = torch.as_tensor(real, device=device)
  generated_tensor = torch.as_tensor(generated, device=device)
  covered_count = 0
  density_count = 0
  with torch.no_grad():
    for start in range(0, real_tensor.size(0), batch_size):
      end = min(start + batch_size, real_tensor.size(0))
      real_distances = torch.cdist(real_tensor[start:end], real_tensor)
      rows = torch.arange(end - start, device=device)
      columns = torch.arange(start, end, device=device)
      real_distances[rows, columns] = torch.inf
      radii = real_distances.kthvalue(neighbors, dim=1).values
      generated_distances = torch.cdist(real_tensor[start:end], generated_tensor)
      memberships = generated_distances <= radii.unsqueeze(1)
      covered_count += memberships.any(dim=1).sum().item()
      density_count += memberships.sum().item()
  return {
      'density': density_count / (neighbors * generated_tensor.size(0)),
      'coverage': covered_count / real_tensor.size(0),
      'feature_names': feature_names,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  parser.add_argument('--file-list', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--metrics-output', type=Path, required=True)
  parser.add_argument('--sample-rate', type=int, default=576000)
  parser.add_argument('--num-samples', type=int, default=None)
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--coverage-neighbors', type=int, default=5)
  parser.add_argument('--evaluation-seed', type=int, default=369)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  args = parser.parse_args()

  device_name = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device
  device = torch.device(device_name)
  torch.manual_seed(args.evaluation_seed)
  if device.type == 'cuda':
    torch.cuda.manual_seed_all(args.evaluation_seed)
  checkpoint = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
  checkpoint_args = checkpoint['args']
  generator = WaveGANGenerator(checkpoint_args['latent_dim'], dim=checkpoint_args['model_dim'],
                                kernel_len=checkpoint_args['kernel_len']).to(device)
  generator.load_state_dict(checkpoint['generator'])
  real_waveforms = load_real_waveforms(args.data_dir, args.file_list, args.sample_rate, args.batch_size)
  generated_waveforms = shared_evaluator.generate_waveforms(
      generator, args.num_samples or real_waveforms.shape[0], checkpoint_args['latent_dim'], args.batch_size, device)
  real_features = extract_features(real_waveforms, args.sample_rate)
  generated_features = extract_features(generated_waveforms, args.sample_rate)
  from scipy.stats import wasserstein_distance
  feature_names = [name for name in real_features if name not in ('mean_psd', 'frequencies_hz')]
  distribution_metrics = feature_density_and_coverage(
      real_features, generated_features, feature_names, device, args.coverage_neighbors, args.batch_size)
  metrics = {
      'checkpoint': str(args.checkpoint.resolve()),
      'checkpoint_epoch': int(checkpoint['epoch']),
      'evaluation_seed': args.evaluation_seed,
      'sample_rate_hz': args.sample_rate,
      'real_sample_count': int(real_waveforms.shape[0]),
      'generated_sample_count': int(generated_waveforms.shape[0]),
      'feature_wasserstein_distance': {
          name: float(wasserstein_distance(real_features[name], generated_features[name])) for name in feature_names},
      'mean_psd_absolute_error': float(abs(real_features['mean_psd'] - generated_features['mean_psd']).mean()),
      'mean_log10_psd_absolute_error': float(abs(
          torch.log10(torch.as_tensor(real_features['mean_psd']) + 1e-12) -
          torch.log10(torch.as_tensor(generated_features['mean_psd']) + 1e-12)).mean()),
      'feature_density': distribution_metrics['density'],
      'feature_coverage': distribution_metrics['coverage'],
      'feature_coverage_neighbors': args.coverage_neighbors,
      'feature_coverage_features': distribution_metrics['feature_names'],
      'normalized_waveform_nearest_neighbor_l2': shared_evaluator.normalized_waveform_nearest_neighbor_metrics(
          real_waveforms, generated_waveforms, device, 1024),
  }
  args.output_dir.mkdir(parents=True, exist_ok=True)
  args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
  with args.metrics_output.open('w', encoding='utf-8') as handle:
    json.dump(metrics, handle, indent=2, ensure_ascii=False)
  shared_evaluator.plot_feature_distributions(real_features, generated_features,
                                              args.output_dir / 'feature_distributions.png')
  shared_evaluator.plot_mean_psd(real_features, generated_features,
                                 args.output_dir / 'mean_psd_comparison.png')
  print('Evaluated {} with fixed evaluation seed {}. Feature density: {:.4f}; coverage: {:.4f}.'.format(
      args.checkpoint, args.evaluation_seed, metrics['feature_density'], metrics['feature_coverage']))


if __name__ == '__main__':
  main()
