# WaveGAN Hilbert 包络 batch size 实验

训练 200 epochs，seed=369；Hilbert MSE 权重=1.0。
复用航次互斥划分：训练 14734 条，测试 3684 条；测试航次 Voyage_Ori_02、Voyage_Ori_08、Voyage_Ori_15。
每条生成样本的包络分别拟合当前真实训练 batch 的逐点平均包络。

时域分由峰值、峰峰值、RMS、FWHM 的 Wasserstein 距离组成；频域分由 log-PSD 误差和 -3 dB 带宽距离组成。
每组先平均 log(指标/基线指标)，再将时域、频域等权平均；指标下限取 1e-12。分数越低越好。
跨 batch 比较统一相对 Raw batch 64；同 batch 增益相对各自 Raw。NN-L2 与包络指标单独报告。

| Batch | Method | Time | Frequency | Composite | Matched composite |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | raw_wavegan | -0.21920 | 0.03388 | -0.09266 | 0.00000 |
| 16 | hilbert | -0.17720 | -0.03020 | -0.10370 | -0.01104 |
| 32 | raw_wavegan | -0.09680 | -0.03569 | -0.06624 | 0.00000 |
| 32 | hilbert | -0.20828 | 0.06714 | -0.07057 | -0.00433 |
| 64 | raw_wavegan | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| 64 | hilbert | 0.06498 | -0.08065 | -0.00784 | -0.00784 |
| 128 | raw_wavegan | 0.36520 | 0.07399 | 0.21959 | 0.00000 |
| 128 | hilbert | 0.37022 | 0.02681 | 0.19852 | -0.02107 |

| Batch | Method | Generated→Real NN-L2 | Real→Generated NN-L2 | Sample envelope MSE | Mean envelope MSE | Envelope std |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | raw_wavegan | 1.43958 | 1.2756 | 0.0221422 | 0.00106082 | 0.103552 |
| 16 | hilbert | 1.4499 | 1.26011 | 0.0215963 | 0.000860711 | 0.104719 |
| 32 | raw_wavegan | 1.46204 | 1.30228 | 0.0235067 | 0.00172785 | 0.105181 |
| 32 | hilbert | 1.42978 | 1.27311 | 0.0215882 | 0.00157329 | 0.10227 |
| 64 | raw_wavegan | 1.46683 | 1.36753 | 0.0226105 | 0.00146913 | 0.10299 |
| 64 | hilbert | 1.49285 | 1.4053 | 0.022185 | 0.00148745 | 0.104522 |
| 128 | raw_wavegan | 1.35835 | 1.42844 | 0.0170626 | 0.000973364 | 0.0894004 |
| 128 | hilbert | 1.3663 | 1.42396 | 0.0159293 | 0.00103257 | 0.082264 |

本轮 Hilbert 组中综合分最低的 batch size=16，综合分=-0.10370，相对同 batch Raw 的综合分=-0.01104。
包络评估使用整个测试集的平均包络；Envelope std 是逐时间点标准差的均值，真实测试值为 0.0922738。
所有组固定 epoch 数；batch size 改变时，优化器更新次数也会改变。结果描述的是该训练预算下的整体效果。
这是单训练种子的 batch size 探索结果；测试集比较不作为独立验证后的最优性能估计。

## Checkpoints

- Batch 16 / raw_wavegan: `D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\runs\batch_16\raw_wavegan\seed_369\checkpoints\wavegan_epoch_0200.pt`
- Batch 16 / hilbert: `D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\runs\batch_16\hilbert\seed_369\checkpoints\wavegan_epoch_0200.pt`
- Batch 32 / raw_wavegan: `D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\runs\batch_32\raw_wavegan\seed_369\checkpoints\wavegan_epoch_0200.pt`
- Batch 32 / hilbert: `D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\runs\batch_32\hilbert\seed_369\checkpoints\wavegan_epoch_0200.pt`
- Batch 64 / raw_wavegan: `D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\runs\batch_64\raw_wavegan\seed_369\checkpoints\wavegan_epoch_0200.pt`
- Batch 64 / hilbert: `D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\runs\batch_64\hilbert\seed_369\checkpoints\wavegan_epoch_0200.pt`
- Batch 128 / raw_wavegan: `D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\runs\batch_128\raw_wavegan\seed_369\checkpoints\wavegan_epoch_0200.pt`
- Batch 128 / hilbert: `D:\Project_Github\DCGAN-PyTorch\wavegan_hilbert\batch_size_experiment\runs\batch_128\hilbert\seed_369\checkpoints\wavegan_epoch_0200.pt`
