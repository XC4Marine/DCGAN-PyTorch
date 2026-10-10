# 域外 Transformer 错误样本与 Joint/Raw 差异分析

## 分析范围

主要比较全网微调后的 X=76 条件：Raw、Joint 各五个分类器种子，并参考不生成样本的五种子基线。再检查所有生成比例，共 65 次模型评估。每次测试为 77 条域外正样本及 203 条原测试负样本。后者不是新增域外负样本。

本分析复用已有预测、波形及训练日志，未重新训练模型。波形形态与覆盖分析使用分类器实际输入：180 kHz、20 kHz 高通、峰值居中、40 点、逐条峰值归一化。这里的最近邻距离与此前对未经分类器预处理的 100 条生成波形计算的距离口径不同。

## 1. 哪些正样本反复漏检

以下七条在基线、Raw X=76、Joint X=76 中均五个种子全部漏检。概率为相应方法五个种子的平均正类概率；阈值为 0.5。

| 波形 | Raw 概率 | Joint 概率 | 所有 65 次评估中的漏检次数 |
| --- | ---: | ---: | ---: |
| [train1_click_0085_peaknorm.wav](D:/Project_Github/雷洲湾标注/data/初筛/train1_click_0085_peaknorm.wav) | 0.2591 | 0.2288 | 65 |
| [train1_click_0086_peaknorm.wav](D:/Project_Github/雷洲湾标注/data/初筛/train1_click_0086_peaknorm.wav) | 0.3604 | 0.3579 | 64 |
| [train2_click_0031_peaknorm.wav](D:/Project_Github/雷洲湾标注/data/初筛/train2_click_0031_peaknorm.wav) | 0.2657 | 0.2558 | 65 |
| [train3_click_0008_peaknorm.wav](D:/Project_Github/雷洲湾标注/data/初筛/train3_click_0008_peaknorm.wav) | 0.1564 | 0.1596 | 65 |
| [train3_click_0016_peaknorm.wav](D:/Project_Github/雷洲湾标注/data/初筛/train3_click_0016_peaknorm.wav) | 0.0697 | 0.0619 | 65 |
| [train3_click_0011_peaknorm.wav](D:/Project_Github/雷洲湾标注/data/初筛/train3_click_0011_peaknorm.wav) | 0.2214 | 0.2060 | 65 |
| [train3_click_0009_peaknorm.wav](D:/Project_Github/雷洲湾标注/data/初筛/train3_click_0009_peaknorm.wav) | 0.3400 | 0.3204 | 64 |

按录音组计算的漏检率如下，分母为该组测试样本数乘五个种子：

| 录音组 | 测试正样本数 | Raw 漏检率 | Joint 漏检率 |
| --- | ---: | ---: | ---: |
| train1 | 46 | 6.09% | 6.52% |
| train2 | 16 | 10.00% | 11.25% |
| train3 | 15 | 36.00% | 36.00% |

错误集中于 train3，且这些难样本在不生成样本的基线中也漏检，属于三种方法共同的问题。

## 2. 难样本的波形特征

对照组为 Raw X=76 五个种子均识别正确的 59 条正样本。下表是各组特征中位数。

| 特征 | 七条共同漏检正样本 | 59 条稳定识别正样本 |
| --- | ---: | ---: |
| 峰值归一化后的 RMS | 0.335 | 0.218 |
| 峰值因子 | 2.99 | 4.58 |
| 主频 | 31.5 kHz | 49.5 kHz |
| 谱质心 | 40.97 kHz | 54.07 kHz |

难样本具有更强的后续振荡、较低的峰值因子及较低的频谱能量中心。它们并不是因为整体幅值很小而被漏检：进入模型时所有波形的绝对峰值均已归一化至 1。

七条难样本到真实训练正样本的最近邻 L2 平均距离为 1.219；稳定识别组为 0.375。这个距离说明当前训练集对难样本形态的覆盖较弱，但不代表标签一定有误。

[查看实际波形及最近训练正样本对照图](D:/Project_Github/DCGAN-PyTorch/DownStream_Classify/results_ood_transformer_finetuning/error_analysis/error_waveforms.png)

## 3. 误报主要来自背景瞬态

| 原测试负样本类别 | 数量 | Raw X=76 误报率 | Joint X=76 误报率 |
| --- | ---: | ---: | ---: |
| 背景瞬态 | 188 | 55.11% | 55.00% |
| 其他物种 | 15 | 6.67% | 6.67% |

Raw 与 Joint 各有 44 条负样本在五个种子中全部误报，其中共同的有 42 条；另有 36 条背景瞬态在全部 65 次模型评估中均被误报。

代表样本：

| 文件标识 | Raw 平均正类概率 | Joint 平均正类概率 |
| --- | ---: | ---: |
| FalseClick_Ori_Recording_08_0381.wav | 0.7341 | 0.7312 |
| FalseClick_Ori_Recording_08_1692.wav | 0.7200 | 0.7462 |
| FalseClick_Ori_Recording_08_0307.wav | 0.7177 | 0.7434 |

完整绝对路径及错误次数见文末样本清单。

在 Raw X=76 中，峰值因子与平均正类概率的 Spearman 相关系数为：正样本 0.815、负样本 0.920。这提示分类器较强地依赖脉冲尖锐度：一部分低峰值因子正样本被漏掉，而尖锐的背景瞬态被判为正类。这里的相关性是诊断线索，尚不能单独证明模型决策的因果机制。

[查看峰值因子与预测概率、生成样本覆盖图](D:/Project_Github/DCGAN-PyTorch/DownStream_Classify/results_ood_transformer_finetuning/error_analysis/crest_factor_diagnostic.png)

## 4. Joint 比 Raw 差多少，差在哪里

| 生成数量 X | Joint−Raw 平衡准确率差（百分点） |
| --- | ---: |
| 8 | +1.008 |
| 23 | −0.094 |
| 38 | −0.537 |
| 53 | −0.408 |
| 61 | −0.063 |
| 76 | −0.210 |

Joint 并非所有条件都更差；X=8 时平衡准确率更高，主要因为误报减少。

X=76 时，Raw/Joint 平衡准确率为 67.87%/67.66%，AUROC 为 0.6930/0.6893。五个种子的配对平衡准确率差依次为 −0.896、−0.246、+0.493、−0.649、+0.246 个百分点，方向不完全一致。

在 280 条测试样本 × 5 个种子 = 1400 次预测中，两种方法仅有 17 次预测类别不同，Raw 正确的有 9 次，Joint 正确的有 8 次。累计漏检正样本次数为 Raw 49、Joint 51；累计误报次数为 Raw 523、Joint 522。Joint 平均差距主要是多漏检了两次正样本，同时少误报一次负样本。

| 种子 | 正样本 | Raw 概率 | Joint 概率 |
| --- | --- | ---: | ---: |
| 369 | [train1_click_0029_peaknorm.wav](D:/Project_Github/雷洲湾标注/data/初筛/train1_click_0029_peaknorm.wav) | 0.5182 | 0.4596 |
| 372 | [train2_click_0029_peaknorm.wav](D:/Project_Github/雷洲湾标注/data/初筛/train2_click_0029_peaknorm.wav) | 0.5264 | 0.4899 |

这些是阈值附近的个别翻转，不能把当前结果解释成 Joint 全面失效。七条最难正样本则是两种方法共同的失败案例。

## 5. 为什么更好的平均频谱没有带来更好的分类

### 已有数据支持的解释

分类器实际使用的前 76 条生成样本中，Joint 对困难形态的覆盖略弱：

| 诊断量 | Raw | Joint |
| --- | ---: | ---: |
| 峰值因子不超过七条难样本最大值 3.659 的生成样本数 | 16/76 | 8/76 |
| 77 条测试正样本到生成池的平均最近邻 L2 | 0.834 | 0.872 |
| 七条难样本到生成池的平均最近邻 L2 | 1.535 | 1.603 |
| 生成池内部平均成对 L2 | 2.015 | 1.863 |

较低的池内距离表明本次 Joint 生成池的形态分散程度较低，但单个生成池不足以据此判定模式崩塌。部分难样本的 Joint 最近邻反而更近，因此覆盖差距也不是逐条一致的。

Joint 的平均频谱指标改善与分类边界改善是不同目标。此前的幅值尺度改善也会被分类器逐条峰值归一化消除。生成增强的效用依赖真实感与多样性，不能由单项分布指标替代下游验证；这一点与 GAN-train/GAN-test 的研究思路一致。[How good is my GAN?](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Konstantin_Shmelkov_How_good_is_ECCV_2018_paper.pdf)

### 代码层面值得验证的机制

`D:\Project_Github\DCGAN-PyTorch\wavegan_crest\crest_loss.py` 对随机同批位置的真实与生成波形逐条计算峰值、峰值因子的 L1 差异；两条波形之间并无真实配对关系。对于随机配对的一维统计量，最小化预期 L1 差异会倾向其中位数，而不是保证整个统计分布被覆盖。因此，该损失可能削弱少数低峰值因子形态的覆盖。

`D:\Project_Github\DCGAN-PyTorch\wavegan_PSDLoss\global_spectral_loss.py` 匹配批均值 log 功率谱；改善批均值并不保证难样本的局部形态和多样性得到改善。

Joint 微调最后 20 次生成器更新中，平均对抗项约 0.221、加权 Crest 项约 0.503、加权 PSD 项约 0.188。辅助项在损失数值中占较大部分；这些数值不是梯度贡献，不能据此直接断言辅助项压制了对抗学习。200 个 epoch 实际包含 400 次判别器更新、80 次生成器更新，模型对域外数据的适配步数仍有限。

以上机制是待验证解释。Raw/Joint 使用了不同预训练生成器，并且当前各只有一次 GAN 微调及一份生成池；现有结果不能把差距单独归因于微调时的某个损失项。

## 6. 建议的最小改进顺序

1. **先处理共同错误：低峰值因子正样本与尖锐背景负样本。** 从训练池提高这两类样本的采样比例，并考虑按录音组平衡正样本采样。新标注优先覆盖 train3 类的低频、较强尾部振荡形态，以及容易混淆的背景瞬态。错误分析中列出的测试样本继续用于评估，不直接并入训练。
2. **对 Joint 做一个单变量对照。** 使用同一 Joint 预训练权重、同一噪声和数据划分，仅将 Crest 的随机逐条匹配改为批内统计量排序后的分布匹配，保持 PSD、优化器和微调范围一致。检验低峰值因子覆盖数、难样本覆盖距离及低误报率下检出率是否改善。随后再根据验证集决定是否降低辅助损失权重，避免同时改多个因素。
3. **让增强比较更干净。** 固定训练负样本池和分类器优化步数。当前各组负样本随 X 增加，50 个 epoch 对应的更新次数也增加，因此与基线的差值不完全等于生成样本本身的贡献。Raw 与 Joint 在同一 X 下的比较不存在这两个差异。
4. **用验证集选择工作阈值。** 从训练数据留出验证集，在目标误报率下比较检出率。当前约 55% 的背景误报率比 Raw/Joint 的 0.21 个百分点差距更值得优先解决。阈值提高会牺牲检出率，不能替代特征学习；不在当前测试集上挑选最优阈值。

后续若训练集区分能力仍不足，可再单独对照分类器输入的时间尺度：源模型使用 576 kHz/128 点，当前为 180 kHz/40 点，同一位置索引对应的物理时间不同。把分类器输入重采样回源模型时间网格是可检验的适配方案，但不会恢复 90 kHz 以上已经丢失的频率信息。此因素是两种方法共同的适配问题，不能单独解释 Joint/Raw 的微小差距。

## 分析产物

- [280 条测试样本在 X=76 的错误次数及特征](D:/Project_Github/DCGAN-PyTorch/DownStream_Classify/results_ood_transformer_finetuning/error_analysis/sample_errors_x76.csv)
- [所有 65 次评估的逐样本错误次数](D:/Project_Github/DCGAN-PyTorch/DownStream_Classify/results_ood_transformer_finetuning/error_analysis/sample_errors_all_conditions.csv)
- [77 条正样本与真实训练集、生成池的覆盖距离](D:/Project_Github/DCGAN-PyTorch/DownStream_Classify/results_ood_transformer_finetuning/error_analysis/positive_errors_and_coverage.csv)
- [X=76 时 Raw/Joint 的全部 17 次预测分歧](D:/Project_Github/DCGAN-PyTorch/DownStream_Classify/results_ood_transformer_finetuning/error_analysis/raw_joint_disagreements_x76.csv)
