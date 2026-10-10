"""Add a real-only baseline and evaluate OOD positives with original test negatives."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from run_ood_finetuning_experiment import (
    CHECKPOINTS, CLASSIFIER_SEEDS, RATIOS, SCRIPT_DIR, WAVEFORM_LENGTH,
    classifier_waveform, load_ood, progress, read_waveform, train_classifier, write_json,
)
from transformer_classifier import (
    ClassicTransformerClassifier, binary_metrics, negative_source, read_manifest, resolve_device,
)


def predict(model, waveforms, device):
  batches = []
  with torch.inference_mode():
    for start in range(0, len(waveforms), 128):
      inputs = torch.from_numpy(waveforms[start:start + 128]).unsqueeze(1).to(device)
      batches.append(model(inputs).cpu())
  return torch.cat(batches)


def evaluate(method, ratio, count, seed, checkpoint_path, training, training_labels,
             test, test_labels, test_paths, test_sources, device, output_dir):
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  model = ClassicTransformerClassifier(waveform_length=WAVEFORM_LENGTH).to(device)
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()
  train_logits = predict(model, training, device)
  train_probabilities = train_logits.softmax(1)[:, 1].numpy()
  train_predictions = (train_probabilities >= 0.5).astype(np.int64)
  train_positive = training_labels == 1
  train_negative = training_labels == 0
  probabilities = predict(model, test, device).softmax(1)[:, 1].numpy()
  predictions = (probabilities >= 0.5).astype(np.int64)
  metrics = binary_metrics(test_labels, probabilities, predictions, test_sources)
  result = {
      "method": method, "ratio": ratio, "generated_count": count, "seed": seed,
      "train_real_positive": 76, "train_negative": int(train_negative.sum()),
      "test_positive": 77, "test_negative": 203, "threshold": 0.5,
      "detected_count": int(predictions[:77].sum()),
      "false_positive_count": int(predictions[77:].sum()),
      "detection_rate": metrics["positive_recall"],
      "false_positive_rate": 1 - metrics["specificity"],
      **metrics,
      "train_loss_eval": float(F.cross_entropy(train_logits, torch.from_numpy(training_labels))),
      "train_accuracy": float(np.mean(train_predictions == training_labels)),
      "train_positive_recall": float(train_predictions[train_positive].mean()),
      "train_specificity": float(np.mean(train_predictions[train_negative] == 0)),
      "train_positive_probability_mean": float(train_probabilities[train_positive].mean()),
      "train_negative_probability_mean": float(train_probabilities[train_negative].mean()),
      "test_positive_probability_mean": float(probabilities[:77].mean()),
      "test_negative_probability_mean": float(probabilities[77:].mean()),
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  write_json(output_dir / "evaluation.json", result)
  write_json(output_dir / "test_predictions.json", [
      {"path": str(path), "label": int(label), "source": source,
       "positive_probability": float(p), "prediction": int(prediction)}
      for path, label, source, p, prediction in zip(
          test_paths, test_labels, test_sources, probabilities, predictions)])
  return result


def summarize(rows, output_dir, runtime):
  finetuning = runtime.get("training_mode") == "source_finetuning"
  summaries = []
  metric_names = (
      "detection_rate", "false_positive_rate", "balanced_accuracy", "auroc", "auprc",
      "positive_precision", "macro_f1", "train_accuracy", "train_loss_eval",
      "train_positive_probability_mean", "train_negative_probability_mean",
      "test_positive_probability_mean", "test_negative_probability_mean",
  )
  conditions = [("baseline", 0)] + [(method, ratio) for method in CHECKPOINTS for ratio in RATIOS]
  for method, ratio in conditions:
    group = [row for row in rows if row["method"] == method and row["ratio"] == ratio]
    summary = {"method": method, "ratio": ratio, "generated_count": group[0]["generated_count"],
               "runs": len(group)}
    for name in metric_names:
      values = np.array([row[name] for row in group])
      summary[name + "_mean"] = float(values.mean())
      summary[name + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    summaries.append(summary)
  for name, records in (("runs", rows), ("summary", summaries)):
    with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(records[0]))
      writer.writeheader()
      writer.writerows(records)
  write_json(output_dir / "summary.json", {"runtime": runtime, "classification": summaries})
  report = [
      "# 源域 Transformer 全网微调比较" if finetuning else "# 真实样本基线与负样本补充评估", "",
      "训练正样本为固定的 76 条域外真实波形，测试正样本为原先保留的 77 条。", "",
      "经 Chico 确认，测试负样本使用原测试清单中的 203 条，不是新增域外负样本。"
      "训练负样本仍来自原训练池，数量与各条件的正样本总数相等。", "",
      ("不生成、Raw 增强、PSD+Crest 增强共 65 次训练，均从同种子的源域基线权重开始，"
       "全网微调 50 轮，基础学习率 2e-5，沿用五轮 warmup 和余弦衰减。"
       "复用此前固定的 100 条生成池，未重新训练 GAN。" if finetuning else
       "新增不加入生成样本的基线，训练 50 轮、五个种子。"
       "Raw/PSD+Crest 的 60 个条件直接评估此前保存的分类器权重；未重新训练 GAN 或增强分类器。"), "",
      f'运行环境：`{runtime["python"]}`；PyTorch `{runtime["torch"]}`。', "",
      "阈值固定 0.5；所有表格为五个种子的均值。平衡准确率为正类检出率与负类特异度的平均值。", "",
      "| 方法 | X | 检出率 | 误报率 | 平衡准确率 | AUROC | 宏 F1 | 训练准确率 |",
      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for row in summaries:
    report.append(
        f'| {row["method"]} | {row["generated_count"]} | '
        f'{row["detection_rate_mean"]:.2%} | {row["false_positive_rate_mean"]:.2%} | '
        f'{row["balanced_accuracy_mean"]:.2%} | {row["auroc_mean"]:.4f} | '
        f'{row["macro_f1_mean"]:.4f} | {row["train_accuracy_mean"]:.2%} |')
  report.extend([
      "", "| 方法 | X | 检出率标准差 | 误报率标准差 | 平衡准确率标准差 | AUROC 标准差 |",
      "| --- | ---: | ---: | ---: | ---: | ---: |",
  ])
  for row in summaries:
    report.append(
        f'| {row["method"]} | {row["generated_count"]} | '
        f'{row["detection_rate_std"]:.2%} | {row["false_positive_rate_std"]:.2%} | '
        f'{row["balanced_accuracy_std"]:.2%} | {row["auroc_std"]:.4f} |')
  report.extend([
      "", "逐种子结果、训练损失与两类概率均值见同目录 runs.csv；逐条测试预测见各条件目录。",
      "真实域外负样本尚未加入；这些结果评价的是域外正样本与原负样本的区分效果。",
  ])
  baseline = summaries[0]
  augmented = summaries[1:]
  if finetuning:
    previous = json.loads(Path(runtime["from_scratch_summary"]).read_text(encoding="utf-8"))
    previous = {(row["method"], row["ratio"]): row for row in previous["classification"]}
    report.extend([
        "", "## 与从头训练的对照", "",
        "使用相同的 76/77 正样本划分、训练负样本顺序、203 条测试负样本和生成池。", "",
        "| 方法 | X | 从头训练平衡准确率 | 微调平衡准确率 | 差值（百分点） | 从头训练 AUROC | 微调 AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summaries:
      old = previous[(row["method"], row["ratio"])]
      report.append(
          f'| {row["method"]} | {row["generated_count"]} | '
          f'{old["balanced_accuracy_mean"]:.2%} | {row["balanced_accuracy_mean"]:.2%} | '
          f'{100 * (row["balanced_accuracy_mean"] - old["balanced_accuracy_mean"]):+.2f} | '
          f'{old["auroc_mean"]:.4f} | {row["auroc_mean"]:.4f} |')
    report.extend([
        "", f'源域权重目录：`{runtime["source_checkpoint_dir"]}`。'
        "使用 seed_369～seed_373 的 baseline/g0 权重；同一个种子的各增强条件使用相同起点。",
        f'不加入生成样本的微调基线平衡准确率为 {baseline["balanced_accuracy_mean"]:.2%}，'
        f'AUROC 为 {baseline["auroc_mean"]:.4f}，训练准确率为 {baseline["train_accuracy_mean"]:.2%}。',
    ])
    report.extend(["", "## 结果解读", ""])
    old_baseline = previous[("baseline", 0)]
    report.append(
        f'源域权重加低学习率全网微调后，真实样本基线的平衡准确率从 '
        f'{old_baseline["balanced_accuracy_mean"]:.2%} 提升至 {baseline["balanced_accuracy_mean"]:.2%}，'
        f'提升 {100 * (baseline["balanced_accuracy_mean"] - old_baseline["balanced_accuracy_mean"]):.2f} 个百分点。')
    for method in CHECKPOINTS:
      row = next(row for row in summaries if row["method"] == method and row["generated_count"] == 76)
      report.extend(["",
          f'{method} 加入 76 条生成样本后，平衡准确率为 {row["balanced_accuracy_mean"]:.2%}，'
          f'相对微调基线变化 {100 * (row["balanced_accuracy_mean"] - baseline["balanced_accuracy_mean"]):+.2f} 个百分点；'
          f'种子间标准差为 {100 * row["balanced_accuracy_std"]:.2f} 个百分点，'
          f'AUROC 为 {row["auroc_mean"]:.4f}，误报率为 {row["false_positive_rate_mean"]:.2%}。'])
    report.extend(["", "生成增强的平均收益应与种子间波动一起解读；高检出率仍需结合误报率评价。"])
    (output_dir / "EXPERIMENT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return
  report.extend([
      "", "## 结果解读", "",
      f'不加入生成样本的基线平衡准确率为 {baseline["balanced_accuracy_mean"]:.2%}，'
      f'AUROC 为 {baseline["auroc_mean"]:.4f}，训练准确率为 {baseline["train_accuracy_mean"]:.2%}。', "",
      f'增强条件的平衡准确率范围为 {min(r["balanced_accuracy_mean"] for r in augmented):.2%}～'
      f'{max(r["balanced_accuracy_mean"] for r in augmented):.2%}，'
      '接近随机分类水平；本次设置未观察到生成增强在固定阈值下的分类收益。', "",
      f'增强条件的 AUROC 均值范围为 {min(r["auroc_mean"] for r in augmented):.4f}～'
      f'{max(r["auroc_mean"] for r in augmented):.4f}。'
      '预测概率保留了一定排序信息，但当前阈值下的检出率较高时，误报率也很高。', "",
      f'增强条件的训练准确率均值范围为 {min(r["train_accuracy_mean"] for r in augmented):.2%}～'
      f'{max(r["train_accuracy_mean"] for r in augmented):.2%}。'
      '模型在训练集上也未表现出有效的分类学习，优先检查训练与输入处理流程，'
      '当前结果不足以将问题归因于样本数量，也不足以支持两种生成方法的下游性能排名。',
  ])
  (output_dir / "EXPERIMENT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-dir", type=Path, default=SCRIPT_DIR / "results_ood_finetuning")
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
  parser.add_argument("--finetune-source-transformer", action="store_true")
  parser.add_argument("--source-checkpoint-dir", type=Path,
                      default=SCRIPT_DIR / "results_ood_initial_screen_gpu" / "checkpoints" / "baseline" / "g0")
  args = parser.parse_args()
  args.source_dir = args.source_dir.resolve()
  if args.output_dir is None:
    args.output_dir = SCRIPT_DIR / ("results_ood_transformer_finetuning" if args.finetune_source_transformer
                                   else "results_ood_baseline_negative_test")
  args.output_dir = args.output_dir.resolve()
  args.source_checkpoint_dir = args.source_checkpoint_dir.resolve()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  device = resolve_device(args.device)
  split = json.loads((args.source_dir / "split.json").read_text(encoding="utf-8"))
  positive_train_paths = [Path(p) for p in split["train_positive"]]
  positive_test_paths = [Path(p) for p in split["test_positive"]]
  negative_train_paths = [Path(p) for p in split["negative_pool"]]
  negative_test_paths = read_manifest(SCRIPT_DIR / "manifests" / "test_negative_203.txt")
  assert len(negative_test_paths) == 203
  assert not set(negative_train_paths).intersection(negative_test_paths)
  real = np.stack([classifier_waveform(x) for x in load_ood(positive_train_paths)])
  positive_test = np.stack([classifier_waveform(x) for x in load_ood(positive_test_paths)])
  negatives = np.stack([classifier_waveform(read_waveform(p)) for p in negative_train_paths])
  negative_test = np.stack([classifier_waveform(read_waveform(p)) for p in negative_test_paths])
  test = np.concatenate((positive_test, negative_test))
  test_labels = np.array([1] * 77 + [0] * 203, dtype=np.int64)
  test_paths = positive_test_paths + negative_test_paths
  test_sources = ["positive"] * 77 + [negative_source(p) for p in negative_test_paths]
  runtime = {"python": str(Path(sys.executable).resolve()), "torch": torch.__version__}
  if args.finetune_source_transformer:
    runtime.update({
        "training_mode": "source_finetuning", "source_checkpoint_dir": str(args.source_checkpoint_dir),
        "base_learning_rate": 2e-5, "trainable_layers": "all",
        "from_scratch_summary": str(SCRIPT_DIR / "results_ood_baseline_negative_test" / "summary.json"),
    })
  write_json(args.output_dir / "configuration.json", {
      **vars(args), **runtime, "classifier_epochs": 50, "classifier_seeds": CLASSIFIER_SEEDS,
      "sample_rate": 180000, "waveform_length": WAVEFORM_LENGTH, "threshold": 0.5,
      "negative_test_source": "original_test_negative_203; not_new_ood_negatives",
  })
  write_json(args.output_dir / "split.json", {
      "train_positive": positive_train_paths, "test_positive": positive_test_paths,
      "negative_train_pool": negative_train_paths, "test_negative": negative_test_paths,
  })
  # Baseline uses the same optimizer schedule and the first 76 training negatives.
  training_args = argparse.Namespace(classifier_epochs=50)
  generated = np.empty((0, WAVEFORM_LENGTH), dtype=np.float32)
  done, total = 0, 65 * 51 if args.finetune_source_transformer else 5 * 50 + 65
  progress(done, total, "starting source fine-tuning" if args.finetune_source_transformer else "starting real-only baseline")
  if not args.finetune_source_transformer:
    for seed in CLASSIFIER_SEEDS:
      run_dir = args.output_dir / "baseline" / "g0" / f"seed_{seed}"
      run_dir.mkdir(parents=True, exist_ok=True)
      _, done = train_classifier(
          "baseline", 0, 0, seed, real, generated, negatives, positive_test, positive_test_paths,
          training_args, device, run_dir, done, total)
  rows = []
  conditions = [("baseline", 0)] + [(method, ratio) for method in CHECKPOINTS for ratio in RATIOS]
  for method, ratio in conditions:
    count = int(76 * ratio + 0.5)
    generated = (np.empty((0, WAVEFORM_LENGTH), dtype=np.float32) if method == "baseline" else
                 np.stack([classifier_waveform(x) for x in
                           np.load(args.source_dir / method / "generated_100.npy")[:count]]))
    positive = np.concatenate((real, generated))
    training = np.concatenate((positive, negatives[:len(positive)]))
    training_labels = np.array([1] * len(positive) + [0] * len(positive), dtype=np.int64)
    for seed in CLASSIFIER_SEEDS:
      relative = Path(method) / f"g{count}" / f"seed_{seed}"
      checkpoint_root = args.output_dir if method == "baseline" or args.finetune_source_transformer else args.source_dir
      if args.finetune_source_transformer:
        run_dir = args.output_dir / relative
        run_dir.mkdir(parents=True, exist_ok=True)
        _, done = train_classifier(
            method, ratio, count, seed, real, generated, negatives, positive_test, positive_test_paths,
            training_args, device, run_dir, done, total,
            source_checkpoint=args.source_checkpoint_dir / f"seed_{seed}.pt", base_learning_rate=2e-5)
      rows.append(evaluate(
          method, ratio, count, seed, checkpoint_root / relative / "classifier.pt",
          training, training_labels, test, test_labels, test_paths, test_sources,
          device, args.output_dir / relative))
      done += 1
      progress(done, total, f"{method} X={count} seed={seed} evaluation")
  summarize(rows, args.output_dir, runtime)
  progress(done, total, f"complete: {args.output_dir}")


if __name__ == "__main__":
  main()
