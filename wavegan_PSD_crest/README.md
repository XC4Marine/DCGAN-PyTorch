# WaveGAN Crest + PSD 联合损失实验

本目录实现 WaveGAN 的 Crest 与 PSD 联合辅助损失实验：

```text
L_G = L_adv + lambda_crest * L_crest + lambda_psd * L_psd
```

当前正式结论来自固定航次互斥 80/20 划分下的十五种子实验。完整数值、逐种子结果、图片和产出溯源见：

- `voyage_split_full/results/EXPERIMENT_REPORT.md`
- `voyage_split_full/results/summary.json`
- `voyage_split_full/results/summary.csv`

## 当前正式实验

- 数据：`data/wav` 中全部 18,418 条 click。
- 训练集：14,734 条；测试集：3,684 条。
- 测试航次：Voyage_Ori_02、Voyage_Ori_08、Voyage_Ori_15。
- 同一航次不会同时出现在训练集和测试集。
- 四种方法使用同一固定划分和相同 seeds 369–383。
- 训练 200 epochs，batch size 64，评估 latent seed 固定为 369。

| 方法 | Crest 权重 | PSD 权重 | 时域分 | 频域分 | 综合分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw WaveGAN | 0.0 | 0.0 | 0.0000 | 0.0000 | 0.0000 |
| Crest-only | 1.0 | 0.0 | -0.8058 | 0.0132 | -0.3963 |
| PSD-only | 0.0 | 0.4 | 0.0628 | -0.6219 | -0.2796 |
| Joint | 0.6 | 0.15 | -0.6342 | -0.5223 | **-0.5782** |

正式综合分使用：

- 时域：峰值、峰峰值、RMS、FWHM 距离。
- 频域：log-PSD、-3 dB 带宽距离。
- 单指标得分：`log(模型指标 / Raw WaveGAN 指标)`。
- 时域与频域先分别取组内均值，再等权平均。
- 分数越低越好；负数表示相对 Raw WaveGAN 的误差更小。

NN-L2 单独报告，不纳入综合分。

## 代码入口

| 文件 | 作用 |
| --- | --- |
| `train_wavegan_torch.py` | 训练 WaveGAN，记录 adversarial、Crest、PSD 和总生成器损失。 |
| `evaluate_wavegan_torch.py` | 使用固定 latent seed 计算测试指标并生成评估图。 |
| `prepare_voyage_split.py` | 根据 PulseTrain 与航次映射生成固定、航次互斥的数据列表。 |
| `run_voyage_split_experiment.py` | 执行四种方法 × 十五个训练种子的正式实验。 |
| `summarize_voyage_split_experiment.py` | 汇总 60 组 `metrics.json`，生成 CSV、JSON、评分图和报告。 |
| `run_weight_search.py` | 执行初始粗搜索、局部细搜索和前三组合三种子复验。 |
| `plot_nearest_waveforms.py` | 生成三个 NN-L2 区间的波形及频谱对比图。 |
| `sanity_check_losses.py` | 检查 Crest 与 PSD loss 的同输入值及反向梯度。 |

## 结果目录

```text
wavegan_PSD_crest/
├── runs/ controls/ results/ images/ checkpoints/
│   └── 初始分阶段权重搜索记录
├── voyage_split_full/
│   ├── split/                 固定训练集、测试集及划分信息
│   ├── runs/                  四种方法 × 十五种子的训练和评估结果
│   ├── results/               正式汇总、评分图和实验报告
│   └── logs/                  十五种子实验总控日志
└── archive/initial_weight_search/
    └── 初始搜索的旧版报告、一次性脚本和总控日志
```

初始搜索图采用当时的历史评分口径，包含后来从正式评分中删除的指标。它们只用于追溯权重筛选过程，不能与十五种子正式综合分直接比较。

## 运行环境

使用 `wavegan_torch` 环境。当前机器对应的解释器为：

```text
D:\Python_env\wavegan_torch\python.exe
```

可先激活环境：

```powershell
conda activate wavegan_torch
```

以下命令均从仓库根目录 `D:\Project_Github\DCGAN-PyTorch` 执行。

## 复现实验

### 1. 损失数值检查

```powershell
python .\wavegan_PSD_crest\sanity_check_losses.py
```

### 2. 生成固定航次划分

```powershell
python .\wavegan_PSD_crest\prepare_voyage_split.py
```

生成：

- `voyage_split_full/split/train_files.txt`
- `voyage_split_full/split/test_files.txt`
- `voyage_split_full/split/split.json`

### 3. 运行十五种子正式实验

```powershell
python .\wavegan_PSD_crest\run_voyage_split_experiment.py --device cuda
```

当某组同时存在 epoch 200 checkpoint 和 `metrics.json` 时会自动跳过，因此中断后可直接使用同一命令继续。

### 4. 重新汇总结果

```powershell
python .\wavegan_PSD_crest\summarize_voyage_split_experiment.py
```

该命令重建 `summary.json`、`summary.csv`、`score_comparison.png` 和报告主体，并保留报告中的 Addition 溯源部分。

### 5. 生成 NN-L2 分区示例图

```powershell
python .\wavegan_PSD_crest\plot_nearest_waveforms.py --device cuda
```

输出区间为 NN-L2 `<0.5`、`0.5–1.0` 和 `>1.0`。这些图片使用初始搜索代表 checkpoint 和完整 `data/wav`，属于定性示例，不是航次互斥测试集统计结果。

### 6. 重新执行初始权重搜索

```powershell
python .\wavegan_PSD_crest\run_weight_search.py --device cuda
```

仅在需要重新验证权重搜索过程时运行。当前正式实验不依赖重新执行该步骤。

## 单组运行产出

每个 `voyage_split_full/runs/<方法>/seed_<种子>/` 包含：

- `training.log`、`epochs.jsonl`、`evaluation.log`、`metrics.json`
- `checkpoints/wavegan_epoch_0200.pt`
- `checkpoints/preview_epoch_0200.wav`
- `images/training_losses.png`
- `images/feature_distributions.png`
- `images/mean_psd_comparison.png`

每类产出使用的数据、生成代码和下游汇总关系，见正式实验报告中的 `Addition：保留记录与产出溯源`。
