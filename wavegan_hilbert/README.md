# WaveGAN Hilbert 包络损失实验

在现有 128 点 WaveGAN 生成器损失中加入当前真实 batch 平均希尔伯特包络约束：

```text
E(x) = abs(IFFT(FFT(x) * h))
target[t] = mean_j E(real_j)[t]
L_hilbert = mean_{i,t} (E(generated_i)[t] - target[t]) ** 2
L_G = -mean(D(generated)) + lambda_hilbert * L_hilbert
```

128 点 FFT 的 h：直流和 Nyquist 项为 1，正频率项为 2，负频率项为 0。真实包络目标停止梯度；生成分支通过 PyTorch FFT 和模长运算反向传播。

先求每条真实波形的包络，再逐时间点求均值。每条生成样本都独立拟合这个目标；生成包络不先做 batch 平均。沿用现有逐波形峰值归一化、右侧补零到 128 点的预处理，包络不做额外归一化、对齐或平滑。

方法依据：[WaveGAN 论文](https://arxiv.org/abs/1802.04208)；[SciPy Hilbert 解析信号及包络定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html)。

## 已确认的首轮配置

| 参数 | 设置 |
| --- | --- |
| Training batch sizes | 16、32、64、128 |
| 方法 | Raw WaveGAN（λ=0）与 Hilbert（λ=1），共 8 组 |
| Epochs / seed | 200 / 369 |
| 训练 / 测试样本数 | 14,734 / 3,684 |
| 测试航次 | Voyage_Ori_02、Voyage_Ori_08、Voyage_Ori_15 |
| 模型 / 对抗训练 | 原目录 128 点 WaveGAN，WGAN-GP，n_critic=5 |
| Evaluation batch size / latent seed | 固定 256 / 369 |
| 每组生成样本数 | 与测试样本数一致，3,684 |
| Checkpoints | epoch 50、100、150、200，全部保留 |

训练列表：`D:\Project_Github\DCGAN-PyTorch\wavegan_PSD_crest\voyage_split_full\split\train_files.txt`。

测试列表：`D:\Project_Github\DCGAN-PyTorch\wavegan_PSD_crest\voyage_split_full\split\test_files.txt`。

不同 batch size 使用相同 epoch 数，因此优化器更新次数随 batch size 改变；每轮日志记录生成器更新数。结果用于比较该预算下的生成性能。

## 运行全部训练与评估

在 PowerShell 执行，解释器固定为指定环境，无需先激活：

```powershell
& 'D:\Python_env\wavegan_torch\python.exe' -X utf8 -u 'D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\run_batch_size_experiment.py' --device cuda
```

输出窗口同时显示全实验阶段进度和当前训练进度。训练和评估日志实时保存。每个 batch size 先运行 Raw，再运行 Hilbert；全部完成后自动汇总 CSV、JSON、比较图和实验报告。

中断后执行相同命令：已完成的训练及评估会跳过；未完成训练从最近保留的 checkpoint 恢复，最多重做尚未保存的 epochs。保存内容包含生成器、判别器、两个优化器、训练参数、global step、固定预览噪声和 PyTorch 随机状态。

改变实验参数时使用新的绝对输出目录，避免把已有结果混入新配置：

```powershell
& 'D:\Python_env\wavegan_torch\python.exe' -X utf8 -u 'D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\run_batch_size_experiment.py' --device cuda --output-dir 'D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment_repeat'
```

## 单独重新汇总

```powershell
& 'D:\Python_env\wavegan_torch\python.exe' -X utf8 'D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\summarize_batch_size_experiment.py' --experiment-dir 'D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment'
```

## 评估口径

沿用 `D:\Project_Github\DCGAN-PyTorch\wavegan_PSD_crest\evaluate_wavegan_torch.py` 的评估口径，并复用 `D:\Project_Github\DCGAN-PyTorch\wavegan_improve\evaluate_wavegan_torch.py` 的特征、生成、近邻和绘图函数：

- 时域：峰值、峰峰值、RMS、FWHM 的 Wasserstein 距离。
- 频域：平均 log10 PSD 绝对误差、-3 dB 带宽的 Wasserstein 距离。
- 综合分：两类指标分别计算相对基线的 log 比值均值，再等权平均。为使零误差仍可计算，指标下限取 1e-12。越低越好。
- 跨 batch 排序统一以本次 Raw batch 64 为基线；同 batch 配对分数分别以本次对应 Raw 为基线。若显式指定的 batch 列表不含 64，共同基线使用列表首项，并记录到报告。
- NN-L2：生成→真实、真实→生成的逐波形 min-max 归一化 L2 近邻距离，单独报告。
- 包络：每条生成包络对测试集平均包络的 MSE、两组平均包络 MSE、真实及生成包络的逐点标准差均值，单独报告。

包络评估使用整个测试集的均值，训练损失始终只使用当前真实训练 batch 的均值。包络离散程度与近邻指标一起用于查看生成样本差异。包络指标和 NN-L2 均不纳入参考综合分。

这是单训练种子的 batch size 探索；测试集比较不能视为另有独立验证的最优性能估计。

## 产出位置

总目录：`D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment`。

每组目录按 batch、方法和 seed 区分，例如：
`D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\runs\batch_64\hilbert\seed_369`。

该组会保存：

- Checkpoint 示例：`D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\runs\batch_64\hilbert\seed_369\checkpoints\wavegan_epoch_0200.pt`。
- 训练逐轮记录：`D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\runs\batch_64\hilbert\seed_369\epochs.jsonl`。
- 评估指标：`D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\runs\batch_64\hilbert\seed_369\metrics.json`。
- 生成波形数组：`D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\runs\batch_64\hilbert\seed_369\generated_waveforms.npy`，shape 为 (3684, 128)。
- 损失、特征分布、平均 PSD、包络对比图：`D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\runs\batch_64\hilbert\seed_369\images`。

总报告：`D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\results\EXPERIMENT_REPORT.md`。

汇总数据：`D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\results\summary.csv`、`D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\results\summary.json`。

当前仅完成实现和必要检查；正式训练、checkpoint 和评估结果将在 Chico 执行后产生。
