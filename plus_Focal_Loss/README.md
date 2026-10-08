# Sorted Crest + PSD 实验

本实验只将原来的逐项 crest 特征匹配改为 batch 内排序匹配，不包含 Focal Loss。
峰值与 crest 分别沿 batch 排序后计算 L1，再按原来的内部权重 0.5 相加。
它匹配两个特征各自的边缘分布，不约束二者的联合分布。

WGAN-GP、网络、PSD、优化器和数据预处理沿用原实现。使用原正式实验的
crest 权重 0.6、PSD 权重 0.15、200 epochs、batch size 64 和固定航次划分。
训练脚本输出总体进度条；本目录复用仓库中的网络和 PSD 模块。

## 训练（PowerShell）

```powershell
& 'D:\Python_env\wavegan_torch\python.exe' 'D:\Project_Github\DCGAN-PyTorch\plus_Focal_Loss\train_wavegan_torch.py' `
  --file-list 'D:\Project_Github\DCGAN-PyTorch\wavegan_PSD_crest\voyage_split_full\split\train_files.txt' `
  --output-dir 'D:\Project_Github\DCGAN-PyTorch\plus_Focal_Loss\runs\sorted_crest\seed_369\checkpoints' `
  --log-file 'D:\Project_Github\DCGAN-PyTorch\plus_Focal_Loss\runs\sorted_crest\seed_369\epochs.jsonl' `
  --crest-loss-weight 0.6 --psd-loss-weight 0.15 --epochs 200 --batch-size 64 --seed 369
```

## 评估（复用原评估入口）

```powershell
& 'D:\Python_env\wavegan_torch\python.exe' 'D:\Project_Github\DCGAN-PyTorch\wavegan_PSD_crest\evaluate_wavegan_torch.py' `
  --checkpoint 'D:\Project_Github\DCGAN-PyTorch\plus_Focal_Loss\runs\sorted_crest\seed_369\checkpoints\wavegan_epoch_0200.pt' `
  --file-list 'D:\Project_Github\DCGAN-PyTorch\wavegan_PSD_crest\voyage_split_full\split\test_files.txt' `
  --output-dir 'D:\Project_Github\DCGAN-PyTorch\plus_Focal_Loss\runs\sorted_crest\seed_369\images' `
  --metrics-output 'D:\Project_Github\DCGAN-PyTorch\plus_Focal_Loss\runs\sorted_crest\seed_369\metrics.json' `
  --evaluation-seed 369
```

先与同种子的原 Joint 结果比较：
`D:\Project_Github\DCGAN-PyTorch\wavegan_PSD_crest\voyage_split_full\runs\joint_best\seed_369\metrics.json`。
对照特征分布图中生成样本对真实分布的覆盖，同时检查 PSD 和特征距离是否退化。
原评估指标和单种子结果不足以单独证明整体波形多样性改善。
本次仅完成代码与损失 sanity checks，完整训练和结果检查由 Chico 执行。

## 跨海域诊断（2026-10-04）

已复用完成的 Sorted Crest checkpoint，与同 GAN seed 369 的 Raw、原 Joint
进行匹配下游实验：G300/G900、三个分类器种子、50 epochs，共18次CUDA训练。
本次没有加入 Focal Loss，也没有重新训练 GAN。

诊断入口：`D:\Project_Github\DCGAN-PyTorch\plus_Focal_Loss\diagnose_domain.py`。
结果和方案见[诊断报告](D:/Project_Github/DCGAN-PyTorch/plus_Focal_Loss/domain_diagnosis/EXPERIMENT_REPORT.md)。
报告中的频谱增强是下一步待验证方案，不是本次已经取得的提升。

```powershell
& 'D:\Python_env\wavegan_torch\python.exe' 'D:\Project_Github\DCGAN-PyTorch\plus_Focal_Loss\diagnose_domain.py'
```
