# 域外微调实验

入口：`D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\run_ood_finetuning_experiment.py`。

## 运行

```powershell
python "D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\run_ood_finetuning_experiment.py" --device cuda
```

默认微调 200 轮，Transformer 训练 50 轮，分类器种子为 369～373。程序逐轮显示总体进度条。可通过 `--finetune-epochs`、`--classifier-epochs`、`--classifier-seeds` 调整。

## 固定实验设置

- 数据：`D:\Project_Github\雷洲湾标注\data\初筛`，153 条单通道波形，180 kHz，每条 37 点。
- 按文件名前缀 train1/train2/train3 分层随机划分；训练分别为 45/16/15 条，测试分别为 46/16/15 条，总计 76/77 条。划分种子 20260929。
- GAN 训练的真实波形逐条峰值归一化，尾部补 3 个零至 40 点；测试集不参与微调、训练或参数选择。
- 比较 Raw WaveGAN 和 PSD+Crest WaveGAN。两者使用相同网络结构，预训练损失不同，沿用现有下游实验的 seed_374、epoch_0200 权重。
- 原始权重分别位于 `D:\Project_Github\DCGAN-PyTorch\wavegan_PSD_crest\voyage_split_full\runs\raw_wavegan\seed_374\checkpoints\wavegan_epoch_0200.pt` 和 `D:\Project_Github\DCGAN-PyTorch\wavegan_PSD_crest\voyage_split_full\runs\joint_best\seed_374\checkpoints\wavegan_epoch_0200.pt`。
- 原生成器输出 128 点；新增五相低通卷积通过 stride=16 得到 `[Batch,5,8]`，按时间交错为 `[Batch,1,40]`。滤波器初始化等价于 SciPy `resample_poly(x,5,16)` 的默认低通 FIR，随后参与学习。
- 仅原生成器 `upconv_1` 和新增滤波器参与微调。判别器随机初始化，最后的线性层适配 40 点，phase shuffle 半径为 1。
- 生成器最后一层和新增滤波器的学习率为 checkpoint 原值的 1/5，即 2e-5；判别器从头训练，学习率保留原值 1e-4。Adam、WGAN-GP 系数和 n_critic 沿用原配置。
- Raw WaveGAN 保留纯对抗损失；PSD+Crest 保留 crest 权重 0.6、PSD 权重 0.15。PSD 的 FFT/步长从 128/32 按采样率比例调整为 40/10。
- 每种方法微调种子为 374；生成种子为 369，固定生成 100 条。保存未经截幅的浮点 WAV 和原始 NumPy 数组。

## 分布与相似度

100 条生成波形与 77 条域外测试波形比较：幅值、峰峰值、RMS、峰值因子、峰位置、主频、谱质心和谱带宽的 Wasserstein 距离，以及平均 PSD 和 log10 PSD 误差。

波形相似度使用逐条峰值归一化后的最近邻 L2 距离，分别统计生成→测试、测试→生成，并提供测试→测试（排除自身）的参考值。距离越小越接近，不进行平移对齐或极性翻转。这里计算的是距离，不转换成百分比。

## Transformer 增强

| 生成比例 | X | 真实正样本 | 训练负样本 |
| --- | ---: | ---: | ---: |
| 10% | 8 | 76 | 84 |
| 30% | 23 | 76 | 99 |
| 50% | 38 | 76 | 114 |
| 70% | 53 | 76 | 129 |
| 80% | 61 | 76 | 137 |
| 100% | 76 | 76 | 152 |

X 按 76 乘比例后四舍五入。各档从同一 100 条生成池中取前 X 条。负样本来自现有清单 `D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\manifests\train_negative_1797.txt`，使用固定顺序的前 76+X 条；不同方法使用相同负样本。

所有分类器输入均处于 180 kHz；负样本先重采样至 180 kHz。随后统一执行现有的 20 kHz 高通、峰值居中的 40 点裁剪/补零及峰值归一化。

Transformer 从头训练，沿用现有结构、优化器和学习率计划，位置编码长度适配 40 点。测试仅使用保留的 77 条域外正样本，阈值固定 0.5；报告检出率、漏检率和平均正类概率。五个分类器种子的均值和标准差用于比较两种方法；GAN 微调本身只运行一个固定种子。

## 输出

默认根目录：`D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\results_ood_finetuning`。

- `D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\results_ood_finetuning\configuration.json`：完整实验参数。
- `D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\results_ood_finetuning\split.json`：训练、测试及负样本的绝对路径清单。
- `D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\results_ood_finetuning\summary.json`：分布指标和分类汇总。
- `D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\results_ood_finetuning\summary.csv`：每档检出率均值及标准差。
- `D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\results_ood_finetuning\runs.csv`：各随机种子的结果。
- `D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\results_ood_finetuning\EXPERIMENT_REPORT.md`：相似度及检出率对照表。

各方法目录保存微调权重、逐步损失、100 条生成波形及分布指标；各生成档位/种子目录保存 Transformer 权重、逐轮损失和逐条测试预测。

## 真实样本基线与负样本补充评估

```powershell
& "D:\Python_env\wavegan_torch\python.exe" "D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\run_ood_baseline_evaluation.py" --device cuda
```

该入口新增 X=0 的真实样本基线：76 条域外正样本、76 条原训练负样本，50 轮、五个种子。原有 60 个增强分类器直接复用保存的权重，不重新训练。所有模型统一在 77 条域外正样本和原测试清单的 203 条负样本上评估。

经 Chico 确认，此次使用原测试负样本补充评估，未加入新的域外负样本。外部目录 `D:\Project_Github\雷洲湾标注\data\负样本\clicks_original_rate` 中的 490 条波形与原训练/测试负样本完全重复，因此不作为新增域外数据使用。

输出目录为 `D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\results_ood_baseline_negative_test`。报告 `D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\results_ood_baseline_negative_test\EXPERIMENT_REPORT.md` 同时给出检出率、误报率、平衡准确率、AUROC、宏 F1 及训练准确率；完整数值和逐条预测保存在该目录。

## 源域 Transformer 全网微调

```powershell
& "D:\Python_env\wavegan_torch\python.exe" "D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\run_ood_baseline_evaluation.py" --device cuda --finetune-source-transformer
```

不生成、Raw 增强和 PSD+Crest 增强均加载源域 baseline/g0 的 Transformer 权重，源目录为 `D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\results_ood_initial_screen_gpu\checkpoints\baseline\g0`。seed_369～seed_373 分别加载对应的源域种子，同一个种子的各条件具有相同起点。所有网络层参与微调，基础学习率 2e-5，沿用五轮 warmup、余弦衰减、50 轮训练和原 AdamW 配置，重新初始化优化器。

复用既有的正负样本划分和生成池，保持 40 点输入。65 次训练全部重新执行。测试仍为 77 条域外正样本和 203 条原测试负样本。

结果另存至 `D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\results_ood_transformer_finetuning`。报告 `D:\Project_Github\DCGAN-PyTorch\DownStream_Classify\results_ood_transformer_finetuning\EXPERIMENT_REPORT.md` 同时列出微调结果及与从头训练的平衡准确率、AUROC 对照；各条件保存源权重路径、基础学习率、微调权重、训练损失及逐条测试预测。
