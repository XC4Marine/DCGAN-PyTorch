"""Evaluate Crest Loss checkpoints with the same artifacts as wavegan_improve.

For each loss weight this script writes feature-distribution and PSD images,
an evaluation transcript, and a metrics JSON.  It then produces the sweep
summary plus the metric-comparison PNG, CSV, and JSON.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EVALUATION_SCRIPT = PROJECT_ROOT / 'wavegan_improve' / 'evaluate_wavegan_torch.py'
DEFAULT_WEIGHTS = tuple(round(index / 10, 1) for index in range(1, 11))
FEATURE_METRICS = (
    'peak_amplitude', 'peak_to_peak', 'rms', 'peak_location_samples',
    'fwhm_samples', 'dominant_frequency_hz', 'spectral_centroid_hz',
    'spectral_bandwidth_hz',
)
COMPARISON_METRICS = (
    *[(name, 'feature_wasserstein_distance.{}'.format(name)) for name in FEATURE_METRICS],
    ('mean_psd_absolute_error', 'mean_psd_absolute_error'),
    ('mean_log10_psd_absolute_error', 'mean_log10_psd_absolute_error'),
    ('NN L2: generated to real', 'normalized_waveform_nearest_neighbor_l2.generated_to_real.mean_l2'),
    ('NN L2: real to generated', 'normalized_waveform_nearest_neighbor_l2.real_to_generated.mean_l2'),
)


def alpha_name(weight):
  return '{:.1f}'.format(weight)


def progress_bar(completed, total, label):
  width = 30
  filled = round(width * completed / total)
  bar = '#' * filled + '-' * (width - filled)
  print('Overall progress: [{}] {:6.2f}% | {}'.format(
      bar, 100.0 * completed / total, label), flush=True)


def copy_training_artifacts(log_dir, source_prefix, destination_prefix, weight):
  """Expose existing training output using the alpha_* naming used by wavegan_improve."""
  name = alpha_name(weight)
  for suffix in ('epochs.jsonl', 'training.log'):
    source = log_dir / '{}_{}_{}'.format(source_prefix, name, suffix)
    destination = log_dir / '{}_{}_{}'.format(destination_prefix, name, suffix)
    if source.exists() and source.resolve() != destination.resolve():
      shutil.copy2(source, destination)


def run_evaluation(command, log_path):
  """Run the shared label-free evaluator and retain its full transcript."""
  with log_path.open('w', encoding='utf-8') as handle:
    handle.write('Command: {}\n\n'.format(' '.join(map(str, command))))
    result = subprocess.run(
        command, cwd=str(SCRIPT_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace')
    handle.write(result.stdout)
  if result.returncode:
    raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)


def nested_value(values, dotted_path):
  result = values
  for key in dotted_path.split('.'):
    result = result[key]
  return float(result)


def comparison_rows(results):
  rows = []
  for display_name, value_path in COMPARISON_METRICS:
    values = {result['alpha']: nested_value(result['metrics'], value_path) for result in results}
    minimum = min(values.values())
    winners = [alpha for alpha, value in values.items() if np.isclose(value, minimum, rtol=1e-12, atol=1e-15)]
    rows.append({
        'metric': display_name,
        'lower_is_better': True,
        'values': values,
        'best_alpha': winners,
    })
  return rows


def write_comparison_data(rows, output_dir):
  alphas = sorted(rows[0]['values'])
  csv_path = output_dir / 'crest_loss_alpha_metric_comparison.csv'
  fields = ['metric', 'lower_is_better', 'best_alpha'] + ['alpha_{:.1f}'.format(alpha) for alpha in alphas]
  with csv_path.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({
          'metric': row['metric'],
          'lower_is_better': row['lower_is_better'],
          'best_alpha': ', '.join('{:.1f}'.format(alpha) for alpha in row['best_alpha']),
          **{'alpha_{:.1f}'.format(alpha): row['values'][alpha] for alpha in alphas},
      })
  json_path = output_dir / 'crest_loss_alpha_metric_comparison.json'
  payload = [{
      'metric': row['metric'],
      'lower_is_better': row['lower_is_better'],
      'best_alpha': row['best_alpha'],
      'values': {'{:.1f}'.format(alpha): value for alpha, value in row['values'].items()},
  } for row in rows]
  with json_path.open('w', encoding='utf-8') as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
  return csv_path, json_path


def plot_sweep_summary(results, output_path):
  """Match wavegan_improve's six-panel alpha sweep summary for Crest Loss."""
  alphas = [result['alpha'] for result in results]
  metrics = [result['metrics'] for result in results]
  panels = [
      ('Peak amplitude Wasserstein distance',
       [metric['feature_wasserstein_distance']['peak_amplitude'] for metric in metrics]),
      ('RMS Wasserstein distance',
       [metric['feature_wasserstein_distance']['rms'] for metric in metrics]),
      ('Peak-to-peak Wasserstein distance',
       [metric['feature_wasserstein_distance']['peak_to_peak'] for metric in metrics]),
      ('Mean PSD absolute error', [metric['mean_psd_absolute_error'] for metric in metrics]),
      ('Mean log10 PSD absolute error', [metric['mean_log10_psd_absolute_error'] for metric in metrics]),
      ('Generated peak-amplitude mean',
       [metric['feature_summary']['generated']['peak_amplitude']['mean'] for metric in metrics]),
  ]
  real_peak_mean = metrics[0]['feature_summary']['real']['peak_amplitude']['mean']
  figure, axes = plt.subplots(2, 3, figsize=(15, 8))
  for axis, (title, values) in zip(axes.flat, panels):
    axis.plot(alphas, values, marker='o', color='#e76f51')
    axis.set_title(title)
    axis.set_xlabel('Crest-loss weight (alpha)')
    axis.grid(alpha=0.25)
  axes.flat[-1].axhline(real_peak_mean, color='#1769aa', linestyle='--', label='Real-data mean')
  axes.flat[-1].legend()
  figure.suptitle('WaveGAN Crest Loss sweep: generated versus training-data statistics', fontsize=15)
  figure.tight_layout(rect=(0, 0, 1, 0.95))
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def plot_comparison_table(rows, output_path):
  alphas = sorted(rows[0]['values'])
  column_labels = ['Metric\n(lower is better)'] + ['alpha={:.1f}'.format(alpha) for alpha in alphas]
  cell_text = [[row['metric']] + ['{:.6g}'.format(row['values'][alpha]) for alpha in alphas] for row in rows]
  figure, axis = plt.subplots(figsize=(20, 4.8))
  axis.axis('off')
  table = axis.table(
      cellText=cell_text, colLabels=column_labels, cellLoc='center', loc='upper center',
      colWidths=[0.22] + [0.078] * len(alphas))
  table.auto_set_font_size(False)
  table.set_fontsize(8)
  table.scale(1, 1.55)
  for column_index in range(len(column_labels)):
    cell = table[(0, column_index)]
    cell.set_facecolor('#1f4e78')
    cell.set_text_props(color='white', weight='bold')
  for row_index, row in enumerate(rows, start=1):
    table[(row_index, 0)].set_facecolor('#eaf2f8')
    for column_index, alpha in enumerate(alphas, start=1):
      cell = table[(row_index, column_index)]
      if alpha in row['best_alpha']:
        cell.set_facecolor('#fff2f2')
        cell.set_text_props(color='#d00000', weight='bold')
  axis.set_title('WaveGAN Crest Loss sweep (red = lowest / best)', fontsize=15, weight='bold', pad=18)
  figure.tight_layout()
  figure.savefig(output_path, dpi=220, bbox_inches='tight')
  plt.close(figure)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint-root', type=Path, default=SCRIPT_DIR / 'checkpoints')
  parser.add_argument('--checkpoint-prefix', default='alpha',
                      help='Checkpoint subdirectory prefix, e.g. alpha or weight.')
  parser.add_argument('--checkpoint-epoch', type=int, default=200)
  parser.add_argument('--weights', type=float, nargs='+', default=DEFAULT_WEIGHTS)
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  args = parser.parse_args()

  if not args.weights or any(weight <= 0 for weight in args.weights):
    raise ValueError('--weights must contain positive values.')
  if not EVALUATION_SCRIPT.is_file():
    raise FileNotFoundError('Shared evaluator not found: {}'.format(EVALUATION_SCRIPT))

  log_dir = SCRIPT_DIR / 'log'
  image_dir = SCRIPT_DIR / 'images'
  log_dir.mkdir(parents=True, exist_ok=True)
  image_dir.mkdir(parents=True, exist_ok=True)
  results = []
  progress_bar(0, len(args.weights), 'starting Crest Loss checkpoint evaluation')
  for index, weight in enumerate(args.weights, start=1):
    name = alpha_name(weight)
    checkpoint = args.checkpoint_root / '{}_{}'.format(args.checkpoint_prefix, name) / (
        'wavegan_epoch_{:04d}.pt'.format(args.checkpoint_epoch))
    if not checkpoint.is_file():
      raise FileNotFoundError('Checkpoint not found: {}'.format(checkpoint))
    copy_training_artifacts(log_dir, args.checkpoint_prefix, 'alpha', weight)
    output_dir = image_dir / 'alpha_{}'.format(name)
    metrics_path = log_dir / 'alpha_{}_metrics.json'.format(name)
    evaluation_log = log_dir / 'alpha_{}_evaluation.log'.format(name)
    command = [
        sys.executable, str(EVALUATION_SCRIPT), '--checkpoint', str(checkpoint),
        '--data-dir', str(args.data_dir), '--batch-size', str(args.batch_size),
        '--device', args.device, '--output-dir', str(output_dir),
        '--metrics-output', str(metrics_path),
    ]
    run_evaluation(command, evaluation_log)
    with metrics_path.open(encoding='utf-8') as handle:
      results.append({'alpha': weight, 'metrics': json.load(handle)})
    progress_bar(index, len(args.weights), 'evaluated Crest Loss alpha {}'.format(name))

  summary_json = log_dir / 'crest_loss_alpha_sweep_summary.json'
  with summary_json.open('w', encoding='utf-8') as handle:
    json.dump(results, handle, indent=2, ensure_ascii=False)
  plot_sweep_summary(results, image_dir / 'crest_loss_alpha_sweep_summary.png')
  rows = comparison_rows(results)
  csv_path, comparison_json = write_comparison_data(rows, log_dir)
  comparison_image = image_dir / 'crest_loss_alpha_metric_comparison.png'
  plot_comparison_table(rows, comparison_image)
  print('Completed {} Crest Loss checkpoint evaluations.'.format(len(results)))
  print('Summary: {}'.format(summary_json.resolve()))
  print('Comparison: {}, {}, {}'.format(comparison_image.resolve(), csv_path.resolve(), comparison_json.resolve()))


if __name__ == '__main__':
  main()
