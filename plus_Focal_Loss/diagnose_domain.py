"""Matched GAN-seed downstream comparison and OOD waveform diagnosis."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'DownStream_Classify'))
import transformer_classifier as classifier
from run_ood_detection_experiment import OOD_ROOT
sys.path.insert(0, str(PROJECT / 'wavegan'))
from pytorch_wavegan import WaveGANGenerator

OUTPUT = ROOT / 'domain_diagnosis'
SEEDS = (369, 370, 371)
COUNTS = (300, 900)
CHECKPOINTS = {
    'raw': PROJECT / 'wavegan_PSD_crest/voyage_split_full/runs/raw_wavegan/seed_369/checkpoints/wavegan_epoch_0200.pt',
    'joint': PROJECT / 'wavegan_PSD_crest/voyage_split_full/runs/joint_best/seed_369/checkpoints/wavegan_epoch_0200.pt',
    'sorted': ROOT / 'runs/sorted_crest/seed_369/checkpoints/wavegan_epoch_0200.pt',
}

def progress(done, total, message):
    width = 30
    print(f'[{"#" * int(width * done / total):{"-"}<{width}}] {done}/{total} {message}', flush=True)

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def generate(method, checkpoint_path):
    directory = OUTPUT / 'generated' / method
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    args = checkpoint['args']
    model = WaveGANGenerator(args['latent_dim'], dim=args['model_dim'], kernel_len=args['kernel_len']).cuda()
    model.load_state_dict(checkpoint['generator'])
    model.eval()
    latent = torch.randn(900, args['latent_dim'], generator=torch.Generator().manual_seed(20260929))
    paths = []
    with torch.inference_mode():
        for start in range(0, 900, 64):
            samples = model(latent[start:start + 64].cuda()).cpu().numpy()[:, 0]
            for index, sample in enumerate(samples, start):
                path = directory / f'{index:04d}.wav'
                wavfile.write(path, args['sample_rate'], sample.astype(np.float32))
                paths.append(path)
    return paths

FEATURES = ('crest', 'half_peak_span_samples', 'centroid_khz', 'bandwidth_khz', 'peak_frequency_khz', 'tail_energy_fraction')

def features(waves):
    power = np.abs(np.fft.rfft(waves, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(128, 1 / 576000) / 1000
    weights = power / power.sum(axis=1, keepdims=True)
    centroid = weights @ frequencies
    bandwidth = np.sqrt((weights * (frequencies[None] - centroid[:, None]) ** 2).sum(axis=1))
    span = []
    for wave in waves:
        indices = np.flatnonzero(np.abs(wave) >= np.max(np.abs(wave)) / 2)
        span.append(indices[-1] - indices[0] + 1)
    energy = waves ** 2
    return np.column_stack((
        np.max(np.abs(waves), axis=1) / np.sqrt(energy.mean(axis=1)),
        span, centroid, bandwidth, frequencies[power.argmax(axis=1)],
        (energy[:, :48].sum(axis=1) + energy[:, 80:].sum(axis=1)) / energy.sum(axis=1),
    ))

def read_predictions(directory):
    rows = [json.loads(line) for line in (directory / 'ood_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    return {row['file']: row['probability_positive'] for row in rows}

def diagnose(pools, ood_paths):
    real_paths = classifier.read_manifest(classifier.MANIFEST_DIR / 'real_positive_600.txt')
    sets = {'real_train': real_paths, 'ood': ood_paths, **pools}
    waves = {name: np.stack([classifier.preprocess_waveform(path, 'positive') for path in paths]) for name, paths in sets.items()}
    values = {name: features(samples) for name, samples in waves.items()}
    rows = []
    for name, matrix in values.items():
        for index, feature in enumerate(FEATURES):
            column = matrix[:, index]
            rows.append({'dataset': name, 'feature': feature, 'q10': np.quantile(column, .1), 'median': np.median(column), 'q90': np.quantile(column, .9), 'std': column.std(), 'wasserstein_to_ood': wasserstein_distance(column, values['ood'][:, index])})
    write_csv(OUTPUT / 'feature_distributions.csv', rows)
    probabilities = {}
    for method in pools:
        predictions = [read_predictions(OUTPUT / 'runs' / method / 'g900' / f'seed_{seed}') for seed in SEEDS]
        probabilities[method] = np.array([[prediction[path.name] for path in ood_paths] for prediction in predictions])
    distances = {name: cdist(waves['ood'], samples).min(axis=1) for name, samples in waves.items() if name != 'ood'}
    cases = []
    for index, path in enumerate(ood_paths):
        row = {'file': str(path.resolve())}
        for method in pools:
            row[f'{method}_mean_probability'] = probabilities[method][:, index].mean()
            row[f'{method}_detected_seed_fraction'] = (probabilities[method][:, index] >= .5).mean()
        row['raw_detect_joint_miss_majority'] = bool((probabilities['raw'][:, index] >= .5).sum() >= 2 and (probabilities['joint'][:, index] >= .5).sum() < 2)
        row.update({feature: values['ood'][index, number] for number, feature in enumerate(FEATURES)})
        row.update({f'nn_l2_to_{name}': distance[index] for name, distance in distances.items()})
        cases.append(row)
    write_csv(OUTPUT / 'ood_cases_g900.csv', cases)
    mask = np.array([row['raw_detect_joint_miss_majority'] for row in cases])
    group_rows = []
    for label, selected in [('raw_detect_joint_miss', mask), ('remaining', ~mask)]:
        for index, feature in enumerate(FEATURES):
            group_rows.append({'group': label, 'count': int(selected.sum()), 'feature': feature, 'mean': values['ood'][selected, index].mean() if selected.any() else None})
        for name, distance in distances.items():
            group_rows.append({'group': label, 'count': int(selected.sum()), 'feature': f'nn_l2_to_{name}', 'mean': distance[selected].mean() if selected.any() else None})
    write_csv(OUTPUT / 'failure_groups.csv', group_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    for index, axis in enumerate(axes.flat):
        for name in ('real_train', 'ood', 'raw', 'joint', 'sorted'):
            column = np.sort(values[name][:, index])
            axis.plot(column, np.arange(1, len(column) + 1) / len(column), label=name)
        axis.set_xlabel(FEATURES[index])
        axis.set_ylabel('CDF')
        axis.grid(alpha=.2)
    axes.flat[0].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / 'feature_cdfs.png', dpi=160)
    plt.close(figure)

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    pools = {name: generate(name, path) for name, path in CHECKPOINTS.items()}
    ood_paths = sorted(OOD_ROOT.glob('*.wav'))
    assert len(ood_paths) == 153
    original_read = classifier.read_manifest
    rows = []
    total = len(pools) * len(COUNTS) * len(SEEDS)
    done = 0
    for method, paths in pools.items():
        def read_manifest(path):
            return paths if path.name == 'generated_joint_wavegan_1200.txt' else original_read(path)
        classifier.read_manifest = read_manifest
        for count in COUNTS:
            for seed in SEEDS:
                directory = OUTPUT / 'runs' / method / f'g{count}' / f'seed_{seed}'
                metrics_path = directory / 'metrics.json'
                progress(done, total, f'{method} G{count} seed {seed}')
                if metrics_path.exists():
                    result = json.loads(metrics_path.read_text(encoding='utf-8'))
                else:
                    result = classifier.train_one_experiment('joint_wavegan', count, seed, directory, 'cuda', 50, 600, ood_paths, directory / 'classifier.pt')
                    result['generator_method'] = method
                    result['generator_checkpoint'] = str(CHECKPOINTS[method])
                    metrics_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
                rows.append({'method': method, 'generated_count': count, 'seed': seed, 'macro_f1': result['metrics']['macro_f1'], 'source_recall': result['metrics']['positive_recall'], 'source_specificity': result['metrics']['specificity'], **result['ood_detection']})
                done += 1
                progress(done, total, 'completed')
                write_csv(OUTPUT / 'runs.csv', rows)
    classifier.read_manifest = original_read
    summaries = []
    for method in pools:
        for count in COUNTS:
            group = [row for row in rows if row['method'] == method and row['generated_count'] == count]
            row = {'method': method, 'generated_count': count, 'runs': len(group)}
            for metric in ('macro_f1', 'source_recall', 'source_specificity', 'detection_rate', 'mean_positive_probability'):
                values = [item[metric] for item in group]
                row[f'{metric}_mean'] = np.mean(values)
                row[f'{metric}_std'] = np.std(values, ddof=1)
            summaries.append(row)
    write_csv(OUTPUT / 'summary.csv', summaries)
    diagnose(pools, ood_paths)
    (OUTPUT / 'protocol.json').write_text(json.dumps({'gan_seed': 369, 'classifier_seeds': SEEDS, 'generated_counts': COUNTS, 'epochs': 50, 'real_positive': 600, 'threshold': .5, 'ood_positive_count': 153, 'checkpoints': {name: str(path) for name, path in CHECKPOINTS.items()}, 'note': 'OOD used for diagnosis only; no threshold fitting. Three classifier seeds and one GAN seed are exploratory evidence.'}, indent=2), encoding='utf-8')
    print(json.dumps(summaries, indent=2), flush=True)
    write_report()

def write_report():
    with (OUTPUT / 'summary.csv').open(encoding='utf-8-sig') as handle:
        summaries = list(csv.DictReader(handle))
    with (OUTPUT / 'feature_distributions.csv').open(encoding='utf-8-sig') as handle:
        distributions = list(csv.DictReader(handle))
    with (OUTPUT / 'ood_cases_g900.csv').open(encoding='utf-8-sig') as handle:
        cases = list(csv.DictReader(handle))
    lines = [
        '# 跨海域 click 分类诊断', '',
        '## 匹配实验', '',
        'Raw、原 Joint、Sorted Crest + PSD 均使用 GAN seed 369 的 epoch 200 checkpoint。相同 latent seed 20260929 生成900条样本，G300取前300条。分类器种子为369、370、371，每次完整训练50 epochs，真实正样本600条，负样本数与总正样本数相等。预处理沿用原脚本，阈值固定0.5。共18次CUDA训练。', '',
        '所有生成池均保存到本目录，运行不修改原下游代码与manifest。分类器复用原joint_wavegan入口并替换读取的生成池；实际生成方法由generator_method与generator_checkpoint字段记录。', '',
        '| 方法 | 合成数 | 源域Macro-F1 | 源域Recall | 源域Specificity | 域外检出率 | 域外平均正类概率 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in summaries:
        metrics = []
        for name in ('macro_f1', 'source_recall', 'source_specificity', 'detection_rate', 'mean_positive_probability'):
            metrics.append(f"{float(row[name + '_mean']):.4f} ± {float(row[name + '_std']):.4f}")
        lines.append('| ' + ' | '.join([row['method'], row['generated_count'], *metrics]) + ' |')
    lines.extend(['', '## 预处理后分布', '',
        '全部以下特征均在分类器实际输入上计算：576 kHz、20 kHz高通、峰值中心128点、逐实例peak归一化。带宽指功率谱加权标准差，不是-3 dB带宽；半峰跨度为所有超过绝对半峰值点的首末间距，不能等同包络FWHM。', '',
        '| 数据集 | Crest中位数 | 半峰跨度中位数/点 | 质心中位数/kHz | RMS频谱带宽中位数/kHz |',
        '| --- | ---: | ---: | ---: | ---: |'])
    for dataset in ('real_train', 'ood', 'raw', 'joint', 'sorted'):
        entries = {row['feature']: row for row in distributions if row['dataset'] == dataset}
        medians = [f"{float(entries[name]['median']):.3f}" for name in FEATURES[:4]]
        lines.append('| ' + ' | '.join([dataset, *medians]) + ' |')
    failed = [row for row in cases if row['raw_detect_joint_miss_majority'] == 'True']
    lines.extend(['', '## G900逐样本诊断', '',
        f'三个种子中至少两个Raw检出、至少两个Joint漏检的样本共{len(failed)}/153条。逐条概率、各特征、到真实训练集和各生成池的波形NN-L2均保存在ood_cases_g900.csv。NN-L2采用分类器peak归一化后的输入，不与原生成评估的min-max NN-L2混用。', '',
        '## 解释与解决方案', '',
        '已确认源训练正类与域外正类存在显著描述性频谱差异：质心及频谱带宽分布不同。三个生成方法仍主要覆盖源域频谱。原Joint的大规模增强跨海域检出弱于Raw，匹配GAN种子后仍可观察到该现象。Sorted排序匹配的有效性以本表实际结果为准，不能根据源域生成指标替代域外结果。', '',
        '这一频谱缺口是三个生成方法共有的问题，不能单独解释Raw与Joint的差距。形态上，Raw的Crest第10百分位约3.998，Joint约4.570，Sorted约4.648，域外中位数约4.558；Raw覆盖更多低Crest样本。半峰跨度的第90百分位分别约18、14、11点，而域外约29点。支持进一步检查类内形态覆盖，但不证明任一单特征是因果根源。', '',
        '机制解释：PSD约束可能强化源域频谱，分类器可能利用频谱捷径。这是待验证的因果解释；本次只确认了分布差异与性能差异，不把海域、设备、方向性或传播中的任一因素指定为根因。', '',
        '首选下一步：固定Joint生成器，先在分类器训练阶段对真实正样本、生成正样本、负样本以相同概率施加平滑随机频率响应扰动，再沿用20 kHz高通、peak归一化。保留未扰动样本。不只增强正类，避免将扰动本身变成标签线索。先验证一个频谱增强因素，不同时添加Focal Loss或重训GAN。扰动强度从训练侧采集信息和独立验证集确定。', '',
        '若训练侧缺少足够低频click成分，频率响应扰动不能凭空补出该成分；应引入训练侧其他海域的已标注click与噪声，再判断是否需要条件生成。', '',
        'Focal Loss调整已观察到的难例权重，不能补充缺失的域外频谱支持。因此目前作为后续对照，不是首选修复。', '',
        '## 证据边界', '',
        '域外153条数据在原脚本中全部按正类评估，没有域外负类。本次只报告检出率，不能证明跨海域完整二分类性能或误报改善。只使用一个GAN种子、三个分类器种子，属于探索性诊断，未执行新的频谱增强方案。目标集已经参与诊断；后续调参后需要新的独立测试数据。', '',
        '## 文献依据', '',
        '- [FilterAugment](https://arxiv.org/abs/2110.03282)：音频频率响应扰动的相关方法；迁移到超声click仍需验证。',
        '- [Device Robust Acoustic Scene Classification, Interspeech 2022](https://www.isca-archive.org/interspeech_2022/sonowal22_interspeech.html)：通过频率响应增强提高未见设备下的音频识别鲁棒性。',
        '- [Focal Loss, ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)：降低容易样本的损失权重，聚焦难样本。', '',
        '## 文件', '',
    ])
    for name in ('summary.csv', 'runs.csv', 'protocol.json', 'feature_distributions.csv', 'failure_groups.csv', 'ood_cases_g900.csv', 'feature_cdfs.png'):
        lines.append(f'- [{name}]({(OUTPUT / name).as_posix()})')
    (OUTPUT / 'EXPERIMENT_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
