"""Summarize the fifteen-seed voyage-disjoint test-set experiment."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR / 'voyage_split_full'
CONFIGURATIONS = (
    ('raw_wavegan', 'Raw WaveGAN', 0.0, 0.0),
    ('crest_only_peak', 'Crest-only', 1.0, 0.0),
    ('psd_only_log_psd', 'PSD-only', 0.0, 0.4),
    ('joint_best', 'Joint', 0.6, 0.15),
)
SEEDS = tuple(range(369, 384))
METRICS = {
    'peak_amplitude': ('time', 'feature_wasserstein_distance', 'peak_amplitude'),
    'peak_to_peak': ('time', 'feature_wasserstein_distance', 'peak_to_peak'),
    'rms': ('time', 'feature_wasserstein_distance', 'rms'),
    'fwhm_samples': ('time', 'feature_wasserstein_distance', 'fwhm_samples'),
    'log_psd': ('frequency', 'mean_log10_psd_absolute_error', None),
    'minus_3db_bandwidth_hz': ('frequency', 'feature_wasserstein_distance', 'minus_3db_bandwidth_hz'),
}
SENSITIVITY_EXCLUDED_SEEDS = (371, 373, 379, 382, 383)
NN_L2_METRICS = ('generated_to_real_mean_l2', 'real_to_generated_mean_l2')


def read_json(path):
  with path.open(encoding='utf-8') as handle:
    return json.load(handle)


def metric_values(metrics):
  values = {}
  for name, (_, parent, child) in METRICS.items():
    values[name] = metrics[parent] if child is None else metrics[parent][child]
  return values


def nn_l2_values(metrics):
  values = metrics['normalized_waveform_nearest_neighbor_l2']
  return {
      'generated_to_real_mean_l2': values['generated_to_real']['mean_l2'],
      'real_to_generated_mean_l2': values['real_to_generated']['mean_l2'],
  }


def score(values, baseline):
  group_scores = {}
  for group in ('time', 'frequency'):
    names = [name for name, (metric_group, _, _) in METRICS.items() if metric_group == group]
    group_scores[group] = float(np.mean([np.log(values[name] / baseline[name]) for name in names]))
  group_scores['composite'] = float(np.mean(list(group_scores.values())))
  return group_scores


def main():
  runs_dir = EXPERIMENT_DIR / 'runs'
  output_dir = EXPERIMENT_DIR / 'results'
  output_dir.mkdir(parents=True, exist_ok=True)
  split = read_json(EXPERIMENT_DIR / 'split' / 'split.json')
  records = {}
  for key, label, crest_weight, psd_weight in CONFIGURATIONS:
    records[key] = []
    for seed in SEEDS:
      metrics = read_json(runs_dir / key / 'seed_{}'.format(seed) / 'metrics.json')
      records[key].append({
          'seed': seed, 'metrics': metric_values(metrics), 'nn_l2': nn_l2_values(metrics)})

  baseline = {name: float(np.mean([record['metrics'][name] for record in records['raw_wavegan']]))
              for name in METRICS}
  summary = {'split': split, 'seeds': SEEDS, 'baseline_metric_means': baseline, 'configurations': {}}
  rows = []
  for key, label, crest_weight, psd_weight in CONFIGURATIONS:
    metric_mean = {name: float(np.mean([record['metrics'][name] for record in records[key]]))
                   for name in METRICS}
    metric_std = {name: float(np.std([record['metrics'][name] for record in records[key]]))
                  for name in METRICS}
    nn_l2_mean = {name: float(np.mean([record['nn_l2'][name] for record in records[key]]))
                  for name in NN_L2_METRICS}
    nn_l2_std = {name: float(np.std([record['nn_l2'][name] for record in records[key]]))
                 for name in NN_L2_METRICS}
    scores = ({'time': 0.0, 'frequency': 0.0, 'composite': 0.0} if key == 'raw_wavegan' else
              score(metric_mean, baseline))
    seed_results = [
        {'seed': record['seed'], 'metrics': record['metrics'],
         'nn_l2': record['nn_l2'], 'scores': score(record['metrics'], baseline)}
        for record in records[key]
    ]
    summary['configurations'][key] = {
        'label': label, 'crest_loss_weight': crest_weight, 'psd_loss_weight': psd_weight,
        'metric_mean': metric_mean, 'metric_std': metric_std, 'scores': scores,
        'nn_l2_mean': nn_l2_mean, 'nn_l2_std': nn_l2_std,
        'seed_results': seed_results,
    }
    rows.append({'configuration': key, 'label': label, 'crest_loss_weight': crest_weight,
                 'psd_loss_weight': psd_weight, **metric_mean, **{
                     '{}_std'.format(name): metric_std[name] for name in METRICS}, **scores})

  with (output_dir / 'summary.json').open('w', encoding='utf-8') as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False)
  with (output_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

  labels = [item[1] for item in CONFIGURATIONS]
  values = [[summary['configurations'][item[0]]['scores'][name] for item in CONFIGURATIONS]
            for name in ('time', 'frequency', 'composite')]
  figure, axis = plt.subplots(figsize=(8, 4.8))
  positions = np.arange(len(labels))
  for offset, (name, group_values) in enumerate(zip(('Time', 'Frequency', 'Composite'), values)):
    axis.bar(positions + (offset - 1) * 0.24, group_values, width=0.24, label=name)
  axis.axhline(0.0, color='black', linewidth=0.8)
  axis.set_xticks(positions, labels)
  axis.set_ylabel('Baseline-normalized score')
  axis.set_title('Voyage-disjoint test-set results (lower is better)')
  axis.legend()
  figure.tight_layout()
  figure.savefig(output_dir / 'score_comparison.png', dpi=180)
  plt.close(figure)

  lines = [
      '# 航次互斥 80/20 十五种子实验报告', '',
      '测试航次：{}。训练 click {} 条，测试 click {} 条。'.format(
          '、'.join(split['test_voyages']), split['train_click_count'], split['test_click_count']), '',
      '综合分沿用当前报告口径：时域（峰值、峰峰值、RMS、FWHM）与频域（log-PSD、-3 dB 带宽）组内分别取相对原始 WaveGAN 的对数比值均值，再等权平均；越低越好。', '',
      '| 模型 | Crest | PSD | 时域分 | 频域分 | 综合分 |',
      '| --- | ---: | ---: | ---: | ---: | ---: |',
  ]
  for key, label, crest_weight, psd_weight in CONFIGURATIONS:
    scores = summary['configurations'][key]['scores']
    lines.append('| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} |'.format(
        label, crest_weight, psd_weight, scores['time'], scores['frequency'], scores['composite']))
  lines.extend(['', '| 指标 | ' + ' | '.join(labels) + ' |',
                '| --- |' + ' ---: |' * len(labels)])
  for name in METRICS:
    cells = []
    for key, _, _, _ in CONFIGURATIONS:
      item = summary['configurations'][key]
      cells.append('{:.5g} ± {:.3g}'.format(item['metric_mean'][name], item['metric_std'][name]))
    lines.append('| {} | {} |'.format(name, ' | '.join(cells)))
  lines.extend([
      '', '## NN-L2 波形近邻距离', '',
      'NN-L2 采用逐波形 min-max 归一化到 [-1, 1] 后的 128 点 L2 距离，越低越好；不纳入综合分。单元格为十五个种子的均值 ± 标准差。', '',
      '| 模型 | 生成→真实 NN-L2 | 真实→生成 NN-L2 |',
      '| --- | ---: | ---: |',
  ])
  for key, label, _, _ in CONFIGURATIONS:
    item = summary['configurations'][key]
    lines.append('| {} | {:.5g} ± {:.3g} | {:.5g} ± {:.3g} |'.format(
        label, item['nn_l2_mean']['generated_to_real_mean_l2'],
        item['nn_l2_std']['generated_to_real_mean_l2'],
        item['nn_l2_mean']['real_to_generated_mean_l2'],
        item['nn_l2_std']['real_to_generated_mean_l2']))
  lines.extend(['', '## 每个随机种子的测试结果', '',
                '以下指标均在同一独立测试集上计算；评分相对五个原始 WaveGAN 种子的指标均值，越低越好。'])
  metric_names = tuple(METRICS)
  for key, label, crest_weight, psd_weight in CONFIGURATIONS:
    item = summary['configurations'][key]
    lines.extend(['', '### {}（Crest={}，PSD={}）'.format(label, crest_weight, psd_weight),
                  '| Seed | {} | Time score | Frequency score | Composite score |'.format(
                      ' | '.join(metric_names) +
                      ' | Generated→Real NN-L2 | Real→Generated NN-L2'),
                  '| ---: |' + ' ---: |' * (len(metric_names) + 5)])
    for seed_result in item['seed_results']:
      metrics = seed_result['metrics']
      nn_l2 = seed_result['nn_l2']
      scores = seed_result['scores']
      lines.append('| {} | {} | {:.5g} | {:.5g} | {:.4f} | {:.4f} | {:.4f} |'.format(
          seed_result['seed'], ' | '.join('{:.5g}'.format(metrics[name]) for name in metric_names),
          nn_l2['generated_to_real_mean_l2'], nn_l2['real_to_generated_mean_l2'],
          scores['time'], scores['frequency'], scores['composite']))
  sensitivity_seeds = [seed for seed in SEEDS if seed not in SENSITIVITY_EXCLUDED_SEEDS]
  sensitivity_baseline = {
      name: float(np.mean([record['metrics'][name] for record in records['raw_wavegan']
                           if record['seed'] in sensitivity_seeds]))
      for name in METRICS
  }
  sensitivity = {}
  for key, label, crest_weight, psd_weight in CONFIGURATIONS:
    selected = [record['metrics'] for record in records[key] if record['seed'] in sensitivity_seeds]
    metric_mean = {name: float(np.mean([metrics[name] for metrics in selected])) for name in METRICS}
    metric_std = {name: float(np.std([metrics[name] for metrics in selected])) for name in METRICS}
    sensitivity[key] = {'metric_mean': metric_mean, 'metric_std': metric_std,
                        'scores': score(metric_mean, sensitivity_baseline)}
  lines.extend([
      '', '## 暂存：十种子事后敏感性分析', '',
      '此表同时排除 seeds 371、373、379、382、383，仅用于查看排除后的数值变化，不替代完整十五种子正式结论。', '',
      '| 模型 | 时域分 | 频域分 | 综合分 |',
      '| --- | ---: | ---: | ---: |',
  ])
  for key, label, _, _ in CONFIGURATIONS:
    scores = sensitivity[key]['scores']
    lines.append('| {} | {:.4f} | {:.4f} | {:.4f} |'.format(
        label, scores['time'], scores['frequency'], scores['composite']))
  lines.extend(['', '| 指标 | ' + ' | '.join(labels) + ' |',
                '| --- |' + ' ---: |' * len(labels)])
  for name in METRICS:
    cells = []
    for key, _, _, _ in CONFIGURATIONS:
      item = sensitivity[key]
      cells.append('{:.5g} ± {:.3g}'.format(item['metric_mean'][name], item['metric_std'][name]))
    lines.append('| {} | {} |'.format(name, ' | '.join(cells)))
  report_path = output_dir / 'EXPERIMENT_REPORT.md'
  addition = ''
  addition_marker = '\n## Addition：保留记录与产出溯源\n'
  if report_path.is_file():
    previous_report = report_path.read_text(encoding='utf-8')
    if addition_marker in previous_report:
      addition = addition_marker + previous_report.split(addition_marker, 1)[1]
  report_path.write_text('\n'.join(lines) + '\n' + addition, encoding='utf-8')
  print('Saved fifteen-seed voyage-disjoint summary to {}.'.format(output_dir))


if __name__ == '__main__':
  main()
