"""Summarize matched batch-size experiments using the reference time/frequency metrics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
TIME_METRICS = ('peak_amplitude', 'peak_to_peak', 'rms', 'fwhm_samples')
FREQUENCY_METRICS = ('log_psd', 'minus_3db_bandwidth_hz')


def read_json(path):
  return json.loads(path.read_text(encoding='utf-8'))


def metric_values(metrics):
  features = metrics['feature_wasserstein_distance']
  nn_l2 = metrics['normalized_waveform_nearest_neighbor_l2']
  envelope = metrics['hilbert_envelope']
  return {
      **{name: features[name] for name in TIME_METRICS + ('minus_3db_bandwidth_hz',)},
      'log_psd': metrics['mean_log10_psd_absolute_error'],
      'generated_to_real_nn_l2': nn_l2['generated_to_real']['mean_l2'],
      'real_to_generated_nn_l2': nn_l2['real_to_generated']['mean_l2'],
      'sample_to_test_mean_envelope_mse': envelope['sample_to_test_mean_mse'],
      'mean_envelope_mse': envelope['mean_envelope_mse'],
      'real_envelope_std_mean': envelope['real_pointwise_std_mean'],
      'generated_envelope_std_mean': envelope['generated_pointwise_std_mean'],
  }


def score(values, baseline):
  groups = [float(np.mean([np.log(max(values[name], 1e-12) / max(baseline[name], 1e-12))
                           for name in names])) for names in (TIME_METRICS, FREQUENCY_METRICS)]
  return {'time_score': groups[0], 'frequency_score': groups[1],
          'composite_score': float(np.mean(groups))}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--experiment-dir', type=Path, default=SCRIPT_DIR / 'batch_size_experiment')
  args = parser.parse_args()
  experiment_dir = args.experiment_dir.resolve()
  config = read_json(experiment_dir / 'experiment_config.json')
  split = read_json(Path(config['split_dir']) / 'split.json')
  results_dir = experiment_dir / 'results'
  results_dir.mkdir(parents=True, exist_ok=True)
  rows, raw_by_batch = [], {}
  for batch_size in config['batch_sizes']:
    for method in ('raw_wavegan', 'hilbert'):
      run_dir = experiment_dir / 'runs' / 'batch_{}'.format(batch_size) / method / 'seed_{}'.format(config['seed'])
      metrics = read_json(run_dir / 'metrics.json')
      row = {'batch_size': batch_size, 'method': method, 'seed': config['seed'],
             'hilbert_loss_weight': metrics['hilbert_loss_weight'],
             **metric_values(metrics), 'checkpoint': metrics['checkpoint'],
             'metrics_path': str(run_dir / 'metrics.json')}
      rows.append(row)
      if method == 'raw_wavegan':
        raw_by_batch[batch_size] = row
  reference_batch_size = 64 if 64 in raw_by_batch else config['batch_sizes'][0]
  reference = raw_by_batch[reference_batch_size]
  for row in rows:
    row.update(score(row, reference))
    row.update({'matched_' + name: value for name, value in score(row, raw_by_batch[row['batch_size']]).items()})
  summary = {'configuration': config, 'split': split, 'reference_batch_size': reference_batch_size,
             'score_epsilon': 1e-12, 'results': rows}
  (results_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
  with (results_dir / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

  figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
  for method, label in (('raw_wavegan', 'Raw WaveGAN'), ('hilbert', 'Hilbert')):
    selected = [row for row in rows if row['method'] == method]
    axes[0].plot([row['batch_size'] for row in selected], [row['composite_score'] for row in selected],
                 marker='o', label=label)
  selected = [row for row in rows if row['method'] == 'hilbert']
  for key, label in (('time', 'Time'), ('frequency', 'Frequency'), ('composite', 'Composite')):
    axes[1].plot([row['batch_size'] for row in selected],
                 [row['matched_' + key + '_score'] for row in selected], marker='o', label=label)
  axes[0].set_title('Common reference: Raw batch {}'.format(reference_batch_size))
  axes[1].set_title('Hilbert relative to Raw at the same batch size')
  for axis in axes:
    axis.set_xscale('log', base=2)
    axis.set_xticks(config['batch_sizes'], [str(value) for value in config['batch_sizes']])
    axis.set_xlabel('Training batch size')
    axis.set_ylabel('Score (lower is better)')
    axis.axhline(0, color='black', linewidth=0.8)
    axis.grid(alpha=0.2)
    axis.legend()
  figure.tight_layout()
  figure.savefig(results_dir / 'batch_size_comparison.png', dpi=180)
  plt.close(figure)

  lines = [
      '# WaveGAN Hilbert 包络 batch size 实验', '',
      '训练 {} epochs，seed={}；Hilbert MSE 权重={}。'.format(
          config['epochs'], config['seed'], config['hilbert_loss_weight']),
      '复用航次互斥划分：训练 {} 条，测试 {} 条；测试航次 {}。'.format(
          split['train_click_count'], split['test_click_count'], '、'.join(split['test_voyages'])),
      '每条生成样本的包络分别拟合当前真实训练 batch 的逐点平均包络。', '',
      '时域分由峰值、峰峰值、RMS、FWHM 的 Wasserstein 距离组成；频域分由 log-PSD 误差和 -3 dB 带宽距离组成。',
      '每组先平均 log(指标/基线指标)，再将时域、频域等权平均；指标下限取 1e-12。分数越低越好。',
      '跨 batch 比较统一相对 Raw batch {}；同 batch 增益相对各自 Raw。NN-L2 与包络指标单独报告。'.format(reference_batch_size), '',
      '| Batch | Method | Time | Frequency | Composite | Matched composite |',
      '| ---: | --- | ---: | ---: | ---: | ---: |',
  ]
  for row in rows:
    lines.append('| {batch_size} | {method} | {time_score:.5f} | {frequency_score:.5f} | '
                 '{composite_score:.5f} | {matched_composite_score:.5f} |'.format(**row))
  lines.extend(['', '| Batch | Method | Generated→Real NN-L2 | Real→Generated NN-L2 | Sample envelope MSE | Mean envelope MSE | Envelope std |',
                '| ---: | --- | ---: | ---: | ---: | ---: | ---: |'])
  for row in rows:
    lines.append('| {batch_size} | {method} | {generated_to_real_nn_l2:.6g} | {real_to_generated_nn_l2:.6g} | '
                 '{sample_to_test_mean_envelope_mse:.6g} | {mean_envelope_mse:.6g} | '
                 '{generated_envelope_std_mean:.6g} |'.format(**row))
  best = min((row for row in rows if row['method'] == 'hilbert'), key=lambda row: row['composite_score'])
  lines.extend([
      '', '本轮 Hilbert 组中综合分最低的 batch size={}，综合分={:.5f}，相对同 batch Raw 的综合分={:.5f}。'.format(
          best['batch_size'], best['composite_score'], best['matched_composite_score']),
      '包络评估使用整个测试集的平均包络；Envelope std 是逐时间点标准差的均值，真实测试值为 {:.6g}。'.format(
          rows[0]['real_envelope_std_mean']),
      '所有组固定 epoch 数；batch size 改变时，优化器更新次数也会改变。结果描述的是该训练预算下的整体效果。',
      '这是单训练种子的 batch size 探索结果；测试集比较不作为独立验证后的最优性能估计。',
      '', '## Checkpoints', '',
  ])
  for row in rows:
    lines.append('- Batch {} / {}: `{}`'.format(row['batch_size'], row['method'], row['checkpoint']))
  (results_dir / 'EXPERIMENT_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  print('Summary saved to {}'.format(results_dir), flush=True)
  print('Best Hilbert batch size in this run: {}'.format(best['batch_size']), flush=True)


if __name__ == '__main__':
  main()
