# 跨海域 click 分类诊断

## 匹配实验

Raw、原 Joint、Sorted Crest + PSD 均使用 GAN seed 369 的 epoch 200 checkpoint。相同 latent seed 20260929 生成900条样本，G300取前300条。分类器种子为369、370、371，每次完整训练50 epochs，真实正样本600条，负样本数与总正样本数相等。预处理沿用原脚本，阈值固定0.5。共18次CUDA训练。

所有生成池均保存到本目录，运行不修改原下游代码与manifest。分类器复用原joint_wavegan入口并替换读取的生成池；实际生成方法由generator_method与generator_checkpoint字段记录。

| 方法 | 合成数 | 源域Macro-F1 | 源域Recall | 源域Specificity | 域外检出率 | 域外平均正类概率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 300 | 0.9409 ± 0.0101 | 0.9914 ± 0.0037 | 0.9031 ± 0.0158 | 0.5926 ± 0.1350 | 0.5558 ± 0.0806 |
| raw | 900 | 0.9428 ± 0.0058 | 0.9957 ± 0.0037 | 0.9031 ± 0.0075 | 0.5795 ± 0.0586 | 0.5261 ± 0.0248 |
| joint | 300 | 0.9419 ± 0.0126 | 0.9957 ± 0.0037 | 0.9015 ± 0.0197 | 0.5447 ± 0.1933 | 0.5303 ± 0.1188 |
| joint | 900 | 0.9531 ± 0.0113 | 0.9978 ± 0.0037 | 0.9195 ± 0.0173 | 0.3878 ± 0.1155 | 0.4387 ± 0.0739 |
| sorted | 300 | 0.9447 ± 0.0032 | 0.9957 ± 0.0037 | 0.9064 ± 0.0049 | 0.5839 ± 0.1581 | 0.5426 ± 0.0800 |
| sorted | 900 | 0.9568 ± 0.0058 | 0.9957 ± 0.0037 | 0.9278 ± 0.0114 | 0.3573 ± 0.0247 | 0.4023 ± 0.0146 |

## 预处理后分布

全部以下特征均在分类器实际输入上计算：576 kHz、20 kHz高通、峰值中心128点、逐实例peak归一化。带宽指功率谱加权标准差，不是-3 dB带宽；半峰跨度为所有超过绝对半峰值点的首末间距，不能等同包络FWHM。

| 数据集 | Crest中位数 | 半峰跨度中位数/点 | 质心中位数/kHz | RMS频谱带宽中位数/kHz |
| --- | ---: | ---: | ---: | ---: |
| real_train | 5.835 | 7.000 | 91.151 | 27.209 |
| ood | 4.558 | 11.000 | 50.962 | 14.901 |
| raw | 5.175 | 8.000 | 92.219 | 29.230 |
| joint | 5.492 | 7.000 | 90.458 | 28.453 |
| sorted | 5.579 | 6.000 | 90.448 | 29.401 |

## G900逐样本诊断

三个种子中至少两个Raw检出、至少两个Joint漏检的样本共39/153条。逐条概率、各特征、到真实训练集和各生成池的波形NN-L2均保存在ood_cases_g900.csv。NN-L2采用分类器peak归一化后的输入，不与原生成评估的min-max NN-L2混用。

## 解释与解决方案

已确认源训练正类与域外正类存在显著描述性频谱差异：质心及频谱带宽分布不同。三个生成方法仍主要覆盖源域频谱。原Joint的大规模增强跨海域检出弱于Raw，匹配GAN种子后仍可观察到该现象。Sorted排序匹配的有效性以本表实际结果为准，不能根据源域生成指标替代域外结果。

这一频谱缺口是三个生成方法共有的问题，不能单独解释Raw与Joint的差距。形态上，Raw的Crest第10百分位约3.998，Joint约4.570，Sorted约4.648，域外中位数约4.558；Raw覆盖更多低Crest样本。半峰跨度的第90百分位分别约18、14、11点，而域外约29点。支持进一步检查类内形态覆盖，但不证明任一单特征是因果根源。

机制解释：PSD约束可能强化源域频谱，分类器可能利用频谱捷径。这是待验证的因果解释；本次只确认了分布差异与性能差异，不把海域、设备、方向性或传播中的任一因素指定为根因。

首选下一步：固定Joint生成器，先在分类器训练阶段对真实正样本、生成正样本、负样本以相同概率施加平滑随机频率响应扰动，再沿用20 kHz高通、peak归一化。保留未扰动样本。不只增强正类，避免将扰动本身变成标签线索。先验证一个频谱增强因素，不同时添加Focal Loss或重训GAN。扰动强度从训练侧采集信息和独立验证集确定。

若训练侧缺少足够低频click成分，频率响应扰动不能凭空补出该成分；应引入训练侧其他海域的已标注click与噪声，再判断是否需要条件生成。

Focal Loss调整已观察到的难例权重，不能补充缺失的域外频谱支持。因此目前作为后续对照，不是首选修复。

## 证据边界

域外153条数据在原脚本中全部按正类评估，没有域外负类。本次只报告检出率，不能证明跨海域完整二分类性能或误报改善。只使用一个GAN种子、三个分类器种子，属于探索性诊断，未执行新的频谱增强方案。目标集已经参与诊断；后续调参后需要新的独立测试数据。

## 文献依据

- [FilterAugment](https://arxiv.org/abs/2110.03282)：音频频率响应扰动的相关方法；迁移到超声click仍需验证。
- [Device Robust Acoustic Scene Classification, Interspeech 2022](https://www.isca-archive.org/interspeech_2022/sonowal22_interspeech.html)：通过频率响应增强提高未见设备下的音频识别鲁棒性。
- [Focal Loss, ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)：降低容易样本的损失权重，聚焦难样本。

## 文件

- [summary.csv](D:/Project_Github/DCGAN-PyTorch/plus_Focal_Loss/domain_diagnosis/summary.csv)
- [runs.csv](D:/Project_Github/DCGAN-PyTorch/plus_Focal_Loss/domain_diagnosis/runs.csv)
- [protocol.json](D:/Project_Github/DCGAN-PyTorch/plus_Focal_Loss/domain_diagnosis/protocol.json)
- [feature_distributions.csv](D:/Project_Github/DCGAN-PyTorch/plus_Focal_Loss/domain_diagnosis/feature_distributions.csv)
- [failure_groups.csv](D:/Project_Github/DCGAN-PyTorch/plus_Focal_Loss/domain_diagnosis/failure_groups.csv)
- [ood_cases_g900.csv](D:/Project_Github/DCGAN-PyTorch/plus_Focal_Loss/domain_diagnosis/ood_cases_g900.csv)
- [feature_cdfs.png](D:/Project_Github/DCGAN-PyTorch/plus_Focal_Loss/domain_diagnosis/feature_cdfs.png)
