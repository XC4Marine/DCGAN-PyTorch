"""Run the staged Crest + PSD weight search and create its result artifacts."""

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
CREST_GRID = (0.1, 0.3, 0.5, 0.7, 1.0)
PSD_GRID = (0.1, 0.2, 0.4, 0.7, 1.0)


def weight_name(weight):
  return '{:g}'.format(weight)


def run_directory(crest_weight, psd_weight, seed):
  return SCRIPT_DIR / 'runs' / 'crest_{}_psd_{}'.format(
      weight_name(crest_weight), weight_name(psd_weight)) / 'seed_{}'.format(seed)


def read_json(path):
  with path.open(encoding='utf-8') as handle:
    return json.load(handle)


def write_json(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8') as handle:
    json.dump(value, handle, indent=2, ensure_ascii=False)


def progress(done, total, label):
  width = 30
  filled = round(width * done / total)
  print('Overall progress: [{}] {:6.2f}% | {}'.format(
      '#' * filled + '-' * (width - filled), 100 * done / total, label), flush=True)


def run_logged(command, log_path):
  with log_path.open('w', encoding='utf-8') as handle:
    handle.write('Command: {}\n\n'.format(' '.join(map(str, command))))
    process = subprocess.Popen(command, cwd=str(SCRIPT_DIR), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    for line in process.stdout:
      handle.write(line)
      print(line, end='', flush=True)
    if process.wait():
      raise subprocess.CalledProcessError(process.returncode, command)


def checkpoint_path(run_dir, epochs):
  return run_dir / 'checkpoints' / 'wavegan_epoch_{:04d}.pt'.format(epochs)


def run_pair(crest_weight, psd_weight, seed, args, completed, total):
  run_dir = run_directory(crest_weight, psd_weight, seed)
  checkpoint = checkpoint_path(run_dir, args.epochs)
  metrics_path = run_dir / 'metrics.json'
  if checkpoint.is_file() and metrics_path.is_file():
    progress(completed + 1, total, 'skip crest={} psd={} seed={}'.format(
        weight_name(crest_weight), weight_name(psd_weight), seed))
    return read_json(metrics_path)

  run_dir.mkdir(parents=True, exist_ok=True)
  train_command = [
      sys.executable, str(SCRIPT_DIR / 'train_wavegan_torch.py'),
      '--data-dir', str(args.data_dir), '--output-dir', str(run_dir / 'checkpoints'),
      '--log-file', str(run_dir / 'epochs.jsonl'), '--epochs', str(args.epochs),
      '--batch-size', str(args.batch_size), '--device', args.device,
      '--crest-loss-weight', weight_name(crest_weight), '--psd-loss-weight', weight_name(psd_weight),
      '--psd-n-fft', str(args.psd_n_fft), '--psd-hop-length', str(args.psd_hop_length), '--seed', str(seed),
  ]
  run_logged(train_command, run_dir / 'training.log')
  evaluation_command = [
      sys.executable, str(SCRIPT_DIR / 'evaluate_wavegan_torch.py'),
      '--checkpoint', str(checkpoint), '--data-dir', str(args.data_dir),
      '--output-dir', str(run_dir / 'images'), '--metrics-output', str(metrics_path),
      '--batch-size', str(args.evaluation_batch_size), '--device', args.device,
      '--evaluation-seed', str(args.evaluation_seed),
  ]
  run_logged(evaluation_command, run_dir / 'evaluation.log')
  progress(completed + 1, total, 'complete crest={} psd={} seed={}'.format(
      weight_name(crest_weight), weight_name(psd_weight), seed))
  return read_json(metrics_path)


def metric_groups(metrics):
  feature = metrics['feature_wasserstein_distance']
  nn_metrics = metrics['normalized_waveform_nearest_neighbor_l2']
  return {
      'time': [feature[name] for name in (
          'peak_amplitude', 'peak_to_peak', 'rms', 'fwhm_samples', 'peak_location_samples')],
      'frequency': [metrics['mean_psd_absolute_error'], metrics['mean_log10_psd_absolute_error']] + [
          feature[name] for name in ('dominant_frequency_hz', 'spectral_centroid_hz', 'spectral_bandwidth_hz')],
      'distribution': [nn_metrics['generated_to_real']['mean_l2'], nn_metrics['real_to_generated']['mean_l2']],
  }


def flat_metrics(metrics):
  groups = metric_groups(metrics)
  names = {
      'time': ('peak_amplitude', 'peak_to_peak', 'rms', 'fwhm', 'peak_location'),
      'frequency': ('psd', 'log_psd', 'dominant_frequency', 'spectral_centroid', 'spectral_bandwidth'),
      'distribution': ('generated_to_real_nn_l2', 'real_to_generated_nn_l2'),
  }
  return {'{}.{}'.format(group, name): float(value)
          for group, values in groups.items() for name, value in zip(names[group], values)}


def score_metrics(metrics, baseline):
  values = metric_groups(metrics)
  baseline_values = metric_groups(baseline)
  group_scores = {
      group: float(np.mean(np.log(np.asarray(values[group]) / np.asarray(baseline_values[group]))))
      for group in values
  }
  return {**group_scores, 'composite_score': float(np.mean(list(group_scores.values())))}


def evaluate_checkpoint(checkpoint, destination, args):
  metrics_path = destination / 'metrics.json'
  if metrics_path.is_file():
    return read_json(metrics_path)
  destination.mkdir(parents=True, exist_ok=True)
  command = [
      sys.executable, str(SCRIPT_DIR / 'evaluate_wavegan_torch.py'), '--checkpoint', str(checkpoint),
      '--data-dir', str(args.data_dir), '--output-dir', str(destination / 'images'),
      '--metrics-output', str(metrics_path), '--batch-size', str(args.evaluation_batch_size),
      '--device', args.device, '--evaluation-seed', str(args.evaluation_seed),
  ]
  run_logged(command, destination / 'evaluation.log')
  return read_json(metrics_path)


def evaluate_controls(args):
  control_root = SCRIPT_DIR / 'controls'
  baseline_checkpoint = PROJECT_ROOT / 'wavegan' / 'train_click_wavegan_torch_gpu' / 'wavegan_epoch_0200.pt'
  baseline = evaluate_checkpoint(baseline_checkpoint, control_root / 'raw_wavegan', args)
  candidates = {}
  for family, source_directory in (
      ('crest_only', PROJECT_ROOT / 'wavegan_crest' / 'checkpoints'),
      ('psd_only', PROJECT_ROOT / 'wavegan_PSDLoss' / 'checkpoints')):
    family_candidates = []
    for weight in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
      source_weight_name = '{:.1f}'.format(weight)
      checkpoint = source_directory / 'weight_{}'.format(source_weight_name) / 'wavegan_epoch_0200.pt'
      metrics = evaluate_checkpoint(checkpoint, control_root / family / 'weight_{}'.format(source_weight_name), args)
      family_candidates.append({
          'weight': weight, 'checkpoint': str(checkpoint), 'metrics': metrics,
          **score_metrics(metrics, baseline),
      })
    candidates[family] = min(family_candidates, key=lambda item: item['composite_score'])
  result = {'baseline': baseline, **candidates}
  write_json(SCRIPT_DIR / 'results' / 'controls.json', result)
  return result


def collect_rows(pairs, seeds, baseline):
  rows = []
  for crest_weight, psd_weight in pairs:
    for seed in seeds:
      metrics_path = run_directory(crest_weight, psd_weight, seed) / 'metrics.json'
      if metrics_path.is_file():
        metrics = read_json(metrics_path)
        rows.append({
            'crest_loss_weight': crest_weight, 'psd_loss_weight': psd_weight, 'seed': seed,
            'checkpoint': metrics['checkpoint'], **score_metrics(metrics, baseline),
        })
  return rows


def write_search_rows(rows):
  results_dir = SCRIPT_DIR / 'results'
  write_json(results_dir / 'search_results.json', rows)
  with (results_dir / 'search_results.csv').open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


def local_axis(current, grid):
  index = grid.index(current)
  if index == 0:
    return (current / 2.0, current, (current + grid[1]) / 2.0)
  if index == len(grid) - 1:
    return ((grid[-2] + current) / 2.0, current, current + (current - grid[-2]) / 2.0)
  return ((grid[index - 1] + current) / 2.0, current, (current + grid[index + 1]) / 2.0)


def plot_heatmap(rows, crest_values, psd_values, output_path, title):
  values = {(row['crest_loss_weight'], row['psd_loss_weight']): row['composite_score'] for row in rows}
  matrix = np.array([[values[(crest, psd)] for psd in psd_values] for crest in crest_values])
  figure, axis = plt.subplots(figsize=(7, 5.5))
  image = axis.imshow(matrix, cmap='viridis_r', aspect='auto')
  axis.set_xticks(range(len(psd_values)), [weight_name(value) for value in psd_values])
  axis.set_yticks(range(len(crest_values)), [weight_name(value) for value in crest_values])
  axis.set_xlabel('PSD loss weight')
  axis.set_ylabel('Crest loss weight')
  axis.set_title(title)
  for y, crest in enumerate(crest_values):
    for x, psd in enumerate(psd_values):
      axis.text(x, y, '{:.3f}'.format(matrix[y, x]), ha='center', va='center', color='white')
  figure.colorbar(image, ax=axis, label='Composite score (lower is better)')
  figure.tight_layout()
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def rank_weights(rows, seeds):
  by_weight = {}
  for row in rows:
    by_weight.setdefault((row['crest_loss_weight'], row['psd_loss_weight']), []).append(row)
  ranking = []
  for (crest_weight, psd_weight), records in by_weight.items():
    if sorted(record['seed'] for record in records) != sorted(seeds):
      continue
    composite_scores = np.asarray([record['composite_score'] for record in records])
    group_means = {name: float(np.mean([record[name] for record in records]))
                   for name in ('time', 'frequency', 'distribution')}
    ranking.append({
        'crest_loss_weight': crest_weight, 'psd_loss_weight': psd_weight,
        'seed_records': records,
        'mean_composite_score': float(composite_scores.mean()),
        'std_composite_score': float(composite_scores.std()),
        'mean_group_scores': group_means,
        'worst_group_score': max(group_means.values()),
    })
  ranking.sort(key=lambda item: (item['mean_composite_score'], item['std_composite_score'],
                                 item['worst_group_score'], item['crest_loss_weight'] + item['psd_loss_weight']))
  return ranking


def copy_best_checkpoints(best):
  target = SCRIPT_DIR / 'checkpoints' / 'best'
  target.mkdir(parents=True, exist_ok=True)
  representative = min(best['seed_records'], key=lambda item: item['composite_score'])
  for record in best['seed_records']:
    source = Path(record['checkpoint'])
    shutil.copy2(source, target / 'seed_{}_wavegan_epoch_0200.pt'.format(record['seed']))
  shutil.copy2(Path(representative['checkpoint']), target / 'representative_lowest_score.pt')
  return target / 'representative_lowest_score.pt'


def plot_final_comparison(best, controls, output_path):
  labels = ['Joint', 'Raw WaveGAN', 'Crest-only', 'PSD-only']
  values = [
      [best['mean_group_scores'][name] for name in ('time', 'frequency', 'distribution')],
      [0.0, 0.0, 0.0],
      [controls['crest_only'][name] for name in ('time', 'frequency', 'distribution')],
      [controls['psd_only'][name] for name in ('time', 'frequency', 'distribution')],
  ]
  positions = np.arange(3)
  figure, axis = plt.subplots(figsize=(9, 5))
  for index, (label, group_values) in enumerate(zip(labels, values)):
    axis.bar(positions + (index - 1.5) * 0.2, group_values, width=0.2, label=label)
  axis.axhline(0.0, color='black', linewidth=0.8)
  axis.set_xticks(positions, ['Time', 'Frequency', 'Distribution'])
  axis.set_ylabel('Baseline-normalized group score')
  axis.set_title('Joint model versus controls (lower is better)')
  axis.legend()
  figure.tight_layout()
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def plot_seed_error_bars(best, output_path):
  names = ('time', 'frequency', 'distribution', 'composite_score')
  labels = ('Time', 'Frequency', 'Distribution', 'Composite')
  means = [np.mean([record[name] for record in best['seed_records']]) for name in names]
  stds = [np.std([record[name] for record in best['seed_records']]) for name in names]
  figure, axis = plt.subplots(figsize=(7, 4.5))
  axis.bar(labels, means, yerr=stds, capsize=5, color='#2a9d8f')
  axis.axhline(0.0, color='black', linewidth=0.8)
  axis.set_ylabel('Baseline-normalized score')
  axis.set_title('Best joint configuration across three seeds')
  figure.tight_layout()
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def build_best_configuration(best, controls, baseline):
  baseline_values = flat_metrics(baseline)
  seed_metrics = [read_json(run_directory(best['crest_loss_weight'], best['psd_loss_weight'], record['seed']) / 'metrics.json')
                  for record in best['seed_records']]
  averaged = {name: float(np.mean([flat_metrics(metrics)[name] for metrics in seed_metrics]))
              for name in baseline_values}
  improvements = {name: float((baseline_values[name] - averaged[name]) / baseline_values[name] * 100)
                  for name in baseline_values}
  representative = copy_best_checkpoints(best)
  control_report = {
      name: best['mean_composite_score'] < controls[name]['composite_score']
      for name in ('crest_only', 'psd_only')
  }
  return {
      'crest_loss_weight': best['crest_loss_weight'],
      'psd_loss_weight': best['psd_loss_weight'],
      'mean_composite_score': best['mean_composite_score'],
      'std_composite_score': best['std_composite_score'],
      'mean_group_scores': best['mean_group_scores'],
      'improvement_percent_relative_to_raw_wavegan': improvements,
      'seed_scores': best['seed_records'],
      'representative_checkpoint': str(representative.resolve()),
      'joint_simultaneously_beats_raw_crest_psd_controls': (
          best['mean_composite_score'] < 0.0 and all(control_report.values())),
      'beats_crest_only': control_report['crest_only'],
      'beats_psd_only': control_report['psd_only'],
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--evaluation-batch-size', type=int, default=256)
  parser.add_argument('--psd-n-fft', type=int, default=128)
  parser.add_argument('--psd-hop-length', type=int, default=32)
  parser.add_argument('--evaluation-seed', type=int, default=369)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  args = parser.parse_args()

  (SCRIPT_DIR / 'results').mkdir(parents=True, exist_ok=True)
  (SCRIPT_DIR / 'images').mkdir(parents=True, exist_ok=True)
  controls = evaluate_controls(args)
  baseline = controls['baseline']

  coarse_pairs = [(crest, psd) for crest in CREST_GRID for psd in PSD_GRID]
  progress(0, 25 + 8 + 6, 'starting 25 coarse-search runs')
  for index, (crest_weight, psd_weight) in enumerate(coarse_pairs):
    run_pair(crest_weight, psd_weight, 369, args, index, 25 + 8 + 6)
  coarse_rows = collect_rows(coarse_pairs, (369,), baseline)
  write_search_rows(coarse_rows)
  plot_heatmap(coarse_rows, CREST_GRID, PSD_GRID, SCRIPT_DIR / 'images' / 'coarse_search_heatmap.png',
               'Coarse search composite scores')

  coarse_best = min(coarse_rows, key=lambda item: item['composite_score'])
  crest_fine = local_axis(coarse_best['crest_loss_weight'], CREST_GRID)
  psd_fine = local_axis(coarse_best['psd_loss_weight'], PSD_GRID)
  fine_pairs = [(crest, psd) for crest in crest_fine for psd in psd_fine]
  for index, (crest_weight, psd_weight) in enumerate(fine_pairs):
    run_pair(crest_weight, psd_weight, 369, args, 25 + index, 25 + 8 + 6)
  all_pairs = list(dict.fromkeys(coarse_pairs + fine_pairs))
  single_seed_rows = collect_rows(all_pairs, (369,), baseline)
  write_search_rows(single_seed_rows)
  fine_rows = [row for row in single_seed_rows if (row['crest_loss_weight'], row['psd_loss_weight']) in fine_pairs]
  plot_heatmap(fine_rows, crest_fine, psd_fine, SCRIPT_DIR / 'images' / 'fine_search_heatmap.png',
               'Local refinement composite scores')

  top_three = sorted(single_seed_rows, key=lambda item: item['composite_score'])[:3]
  for combo_index, row in enumerate(top_three):
    for seed_index, seed in enumerate((370, 371), start=1):
      run_pair(row['crest_loss_weight'], row['psd_loss_weight'], seed, args,
               33 + combo_index * 2 + seed_index - 1, 25 + 8 + 6)
  final_rows = collect_rows([(row['crest_loss_weight'], row['psd_loss_weight']) for row in top_three],
                            (369, 370, 371), baseline)
  ranking = rank_weights(final_rows, (369, 370, 371))
  write_json(SCRIPT_DIR / 'results' / 'final_ranking.json', ranking)
  best_configuration = build_best_configuration(ranking[0], controls, baseline)
  write_json(SCRIPT_DIR / 'results' / 'best_configuration.json', best_configuration)
  plot_final_comparison(ranking[0], controls, SCRIPT_DIR / 'images' / 'joint_vs_controls.png')
  plot_seed_error_bars(ranking[0], SCRIPT_DIR / 'images' / 'best_three_seed_error_bars.png')
  print('Best configuration: crest={} psd={}'.format(
      weight_name(best_configuration['crest_loss_weight']), weight_name(best_configuration['psd_loss_weight'])))


if __name__ == '__main__':
  main()
