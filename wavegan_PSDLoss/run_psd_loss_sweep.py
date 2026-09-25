"""Train and evaluate WaveGAN runs across batch PSD-loss weights."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_WEIGHTS = tuple(round(index / 10, 1) for index in range(1, 11))


def weight_name(weight):
  return '{:.1f}'.format(weight)


def progress_bar(completed, total, label):
  width = 30
  filled = round(width * completed / total)
  print('Overall progress: [{}] {:6.2f}% | {}'.format(
      '#' * filled + '-' * (width - filled), 100.0 * completed / total, label), flush=True)


def run_and_log(command, log_path, run_index, run_count, epochs, weight):
  epoch_pattern = re.compile(r'Epoch (\d+)/(\d+) \| batch (\d+)/(\d+)')
  last_reported_epoch = 0
  with log_path.open('w', encoding='utf-8') as log_handle:
    log_handle.write('Command: {}\n\n'.format(' '.join(map(str, command))))
    process = subprocess.Popen(command, cwd=str(SCRIPT_DIR), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    assert process.stdout is not None
    for line in process.stdout:
      log_handle.write(line)
      match = epoch_pattern.search(line)
      if match:
        epoch, total_epochs, batch, total_batches = map(int, match.groups())
        if batch == total_batches and (epoch == 1 or epoch == total_epochs or epoch - last_reported_epoch >= 10):
          last_reported_epoch = epoch
          progress_bar(run_index * epochs + epoch, run_count * epochs,
                       'weight {} training epoch {}/{}'.format(weight_name(weight), epoch, total_epochs))
    if process.wait():
      raise subprocess.CalledProcessError(process.returncode, command)


def plot_sweep_summary(results, output_path):
  weights = [result['weight'] for result in results]
  metrics = [result['metrics'] for result in results]
  plots = [
      ('Mean PSD absolute error', [item['mean_psd_absolute_error'] for item in metrics]),
      ('Mean log10 PSD absolute error', [item['mean_log10_psd_absolute_error'] for item in metrics]),
      ('Dominant-frequency Wasserstein distance', [item['feature_wasserstein_distance']['dominant_frequency_hz'] for item in metrics]),
      ('Spectral-centroid Wasserstein distance', [item['feature_wasserstein_distance']['spectral_centroid_hz'] for item in metrics]),
  ]
  figure, axes = plt.subplots(2, 2, figsize=(12, 8))
  for axis, (title, values) in zip(axes.flat, plots):
    axis.plot(weights, values, marker='o', color='#e76f51')
    axis.set_title(title)
    axis.set_xlabel('PSD-loss weight')
    axis.grid(alpha=0.25)
  figure.suptitle('WaveGAN batch PSD-loss sweep', fontsize=15)
  figure.tight_layout(rect=(0, 0, 1, 0.95))
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  parser.add_argument('--seed', type=int, default=369)
  parser.add_argument('--checkpoint-every', type=int, default=200)
  parser.add_argument('--weights', type=float, nargs='+', default=DEFAULT_WEIGHTS)
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  parser.add_argument('--psd-n-fft', type=int, default=128)
  parser.add_argument('--psd-hop-length', type=int, default=32)
  args = parser.parse_args()
  if args.epochs < 1:
    raise ValueError('--epochs must be positive.')
  if not args.weights or any(weight < 0 for weight in args.weights):
    raise ValueError('--weights must contain at least one non-negative value.')

  log_dir, image_dir, checkpoint_dir = (SCRIPT_DIR / name for name in ('log', 'images', 'checkpoints'))
  for directory in (log_dir, image_dir, checkpoint_dir):
    directory.mkdir(parents=True, exist_ok=True)
  results = []
  total_runs = len(args.weights)
  progress_bar(0, total_runs * args.epochs, 'starting PSD-loss sweep')
  for index, weight in enumerate(args.weights):
    name = weight_name(weight)
    run_dir = checkpoint_dir / 'weight_{}'.format(name)
    image_output_dir = image_dir / 'weight_{}'.format(name)
    checkpoint = run_dir / 'wavegan_epoch_{:04d}.pt'.format(args.epochs)
    epoch_log = log_dir / 'weight_{}_epochs.jsonl'.format(name)
    train_command = [
        sys.executable, str(SCRIPT_DIR / 'train_wavegan_torch.py'), '--data-dir', str(args.data_dir),
        '--output-dir', str(run_dir), '--epochs', str(args.epochs), '--batch-size', str(args.batch_size),
        '--device', args.device, '--seed', str(args.seed), '--checkpoint-every', str(args.checkpoint_every),
        '--psd-loss-weight', name, '--psd-n-fft', str(args.psd_n_fft),
        '--psd-hop-length', str(args.psd_hop_length), '--log-file', str(epoch_log)]
    print('\nStarting PSD-loss weight={} ({}/{})'.format(name, index + 1, total_runs), flush=True)
    run_and_log(train_command, log_dir / 'weight_{}_training.log'.format(name), index, total_runs, args.epochs, weight)
    evaluate_command = [
        sys.executable, str(SCRIPT_DIR / 'evaluate_wavegan_torch.py'), '--checkpoint', str(checkpoint),
        '--data-dir', str(args.data_dir), '--batch-size', str(args.batch_size), '--device', args.device,
        '--output-dir', str(image_output_dir), '--metrics-output', str(log_dir / 'weight_{}_metrics.json'.format(name))]
    run_and_log(evaluate_command, log_dir / 'weight_{}_evaluation.log'.format(name), index, total_runs, args.epochs, weight)
    with (log_dir / 'weight_{}_metrics.json'.format(name)).open(encoding='utf-8') as handle:
      results.append({'weight': weight, 'metrics': json.load(handle)})
    progress_bar((index + 1) * args.epochs, total_runs * args.epochs,
                 'weight {} training and evaluation complete'.format(name))
  plot_sweep_summary(results, image_dir / 'psd_loss_weight_sweep_summary.png')
  with (log_dir / 'psd_loss_weight_sweep_summary.json').open('w', encoding='utf-8') as handle:
    json.dump(results, handle, indent=2, ensure_ascii=False)
  print('\nCompleted all {} PSD-loss weights.'.format(total_runs))
  print('Logs: {}'.format(log_dir.resolve()))
  print('Images: {}'.format(image_dir.resolve()))


if __name__ == '__main__':
  main()
