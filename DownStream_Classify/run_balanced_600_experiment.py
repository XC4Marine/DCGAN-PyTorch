"""Run the balanced 600-real-positive PSD+Crest WaveGAN experiment."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from transformer_classifier import train_one_experiment


SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_COUNTS = (0, 150, 300, 450, 600, 750, 900, 1050, 1197)
SEEDS = (369, 370, 371, 372, 373)
METRIC_NAMES = (
    "macro_f1",
    "balanced_accuracy",
    "auroc",
    "auprc",
    "positive_precision",
    "positive_recall",
    "positive_f1",
    "specificity",
    "background_specificity",
    "other_species_specificity",
)
METHODS = ("raw_wavegan", "joint_wavegan")
METHOD_LABELS = {
    "baseline": "No augmentation",
    "raw_wavegan": "Raw WaveGAN",
    "joint_wavegan": "PSD+Crest WaveGAN",
}
CONDITIONS = (("baseline", 0),) + tuple(
    (method, count) for method in METHODS for count in GENERATED_COUNTS[1:]
)


def progress(done, total, message):
  width = 30
  filled = int(width * done / total)
  bar = "#" * filled + "-" * (width - filled)
  print(f"[{bar}] {done}/{total} {message}", flush=True)


def flatten_result(result):
  row = {
      "method": result["method"],
      "generated_count": result["train_generated_positive"],
      "seed": result["seed"],
      "epochs": result["epochs"],
      "train_real_positive": result["train_real_positive"],
      "train_negative": result["train_negative"],
  }
  row.update(result["metrics"])
  return row


def paired_bootstrap(rows, left_method, left_count, right_method, right_count,
                     seed_offset, iterations=10000):
  left = {
      row["seed"]: row["macro_f1"]
      for row in rows
      if row["method"] == left_method and row["generated_count"] == left_count
  }
  right = {
      row["seed"]: row["macro_f1"]
      for row in rows
      if row["method"] == right_method and row["generated_count"] == right_count
  }
  differences = np.array([
      left[seed] - right[seed] for seed in sorted(left)
  ])
  generator = np.random.default_rng(20260929 + seed_offset)
  samples = generator.choice(
      differences, size=(iterations, len(differences)), replace=True
  ).mean(axis=1)
  return {
      "mean_difference": float(differences.mean()),
      "ci95_low": float(np.percentile(samples, 2.5)),
      "ci95_high": float(np.percentile(samples, 97.5)),
  }


def summarize(rows, output_dir):
  summary_rows = []
  for method, generated_count in CONDITIONS:
    group = [
        row for row in rows
        if row["method"] == method and row["generated_count"] == generated_count
    ]
    summary = {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "generated_count": generated_count,
        "positive_count": 600 + generated_count,
        "negative_count": 600 + generated_count,
        "runs": len(group),
    }
    for metric in METRIC_NAMES:
      values = np.array([row[metric] for row in group], dtype=np.float64)
      summary[f"{metric}_mean"] = float(values.mean())
      summary[f"{metric}_std"] = float(values.std(ddof=1))
    summary_rows.append(summary)

  comparisons = []
  offset = 0
  for method in METHODS:
    for count in GENERATED_COUNTS[1:]:
      offset += 1
      comparisons.append({
          "comparison": f"{method}_g{count}_minus_baseline",
          "generated_count": count,
          **paired_bootstrap(rows, method, count, "baseline", 0, offset),
      })
  for count in GENERATED_COUNTS[1:]:
    offset += 1
    comparisons.append({
        "comparison": f"joint_minus_raw_g{count}",
        "generated_count": count,
        **paired_bootstrap(
            rows, "joint_wavegan", count, "raw_wavegan", count, offset
        ),
    })
  output_dir.mkdir(parents=True, exist_ok=True)
  with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
    writer.writeheader()
    writer.writerows(summary_rows)
  with (output_dir / "paired_comparisons.csv").open(
      "w", newline="", encoding="utf-8-sig"
  ) as handle:
    writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
    writer.writeheader()
    writer.writerows(comparisons)
  (output_dir / "summary.json").write_text(
      json.dumps({
          "real_positive_count": 600,
          "conditions": [
              {"method": method, "generated_count": count}
              for method, count in CONDITIONS
          ],
          "seeds": list(SEEDS),
          "summary": summary_rows,
          "paired_comparisons": comparisons,
      }, indent=2, ensure_ascii=False),
      encoding="utf-8",
  )

  def mean_std(row, metric):
    return f'{row[f"{metric}_mean"]:.4f} +/- {row[f"{metric}_std"]:.4f}'

  report = [
      "# 600条真实正样本的平衡数据增强实验",
      "",
      "固定600条真实正样本，比较普通WaveGAN与PSD+Crest WaveGAN；每组唯一负样本数等于正样本总数。",
      "每组使用5个分类器种子、50 epochs和同一测试集。",
      "",
      "| 方法 | 合成正样本 | 正样本总数 | 负样本 | Precision | Recall | 正类F1 | Macro-F1 | AUROC | AUPRC | Specificity |",
      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for row in summary_rows:
    report.append(
        f'| {row["method_label"]} | {row["generated_count"]} | '
        f'{row["positive_count"]} | {row["negative_count"]} | '
        f'{mean_std(row, "positive_precision")} | {mean_std(row, "positive_recall")} | '
        f'{mean_std(row, "positive_f1")} | {mean_std(row, "macro_f1")} | '
        f'{mean_std(row, "auroc")} | {mean_std(row, "auprc")} | '
        f'{mean_std(row, "specificity")} |'
    )
  report.extend([
      "",
      "## 配对Macro-F1差值",
      "",
      "| 对比 | 合成正样本 | 平均差值 | Bootstrap 95% CI |",
      "| --- | ---: | ---: | ---: |",
  ])
  for comparison in comparisons:
    report.append(
        f'| {comparison["comparison"]} | {comparison["generated_count"]} | '
        f'{comparison["mean_difference"]:.4f} | '
        f'[{comparison["ci95_low"]:.4f}, {comparison["ci95_high"]:.4f}] |'
    )
  (output_dir / "EXPERIMENT_REPORT.md").write_text(
      "\n".join(report) + "\n", encoding="utf-8"
  )

  figure, axes = plt.subplots(1, 3, figsize=(12, 4))
  baseline = summary_rows[0]
  for method in METHODS:
    method_rows = [baseline] + [
        row for row in summary_rows if row["method"] == method
    ]
    for axis, metric, title in zip(
        axes, ("macro_f1", "auroc", "auprc"), ("Macro-F1", "AUROC", "AUPRC")
    ):
      axis.errorbar(
          GENERATED_COUNTS,
          [row[f"{metric}_mean"] for row in method_rows],
          yerr=[row[f"{metric}_std"] for row in method_rows],
          marker="o",
          capsize=4,
          label=METHOD_LABELS[method],
      )
      axis.set_title(title)
      axis.set_xlabel("Generated positive samples")
      axis.grid(alpha=0.25)
      axis.legend()
  figure.tight_layout()
  figure.savefig(output_dir / "balanced_augmentation_curve.png", dpi=180)
  plt.close(figure)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument(
      "--output-dir", type=Path, default=SCRIPT_DIR / "results_balanced_600_gpu"
  )
  parser.add_argument("--smoke-test", action="store_true")
  args = parser.parse_args()

  conditions = (("raw_wavegan", 1197),) if args.smoke_test else CONDITIONS
  seeds = (369,) if args.smoke_test else SEEDS
  epochs = 1 if args.smoke_test else args.epochs
  output_root = args.output_dir / "smoke" if args.smoke_test else args.output_dir
  total = len(conditions) * len(seeds)
  completed = 0
  rows = []
  progress(completed, total, "starting")
  for method, generated_count in conditions:
    for seed in seeds:
      method_dir = Path() if method in ("baseline", "joint_wavegan") else Path(method)
      run_dir = output_root / method_dir / f"g{generated_count}" / f"seed_{seed}"
      metrics_path = run_dir / "metrics.json"
      if metrics_path.exists():
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        expected = (method, generated_count, 600 + generated_count, seed, epochs)
        actual = (
            result["method"], result["train_generated_positive"],
            result["train_negative"], result["seed"], result["epochs"],
        )
        if actual != expected:
          raise ValueError(
              f"Existing result {metrics_path} has configuration {actual}; expected {expected}"
          )
      else:
        result = train_one_experiment(
            method, generated_count, seed, run_dir, args.device, epochs, 600
        )
      rows.append(flatten_result(result))
      completed += 1
      progress(completed, total, f"finished G{generated_count} seed {seed}")

  if args.smoke_test:
    print(json.dumps(rows[0], indent=2, ensure_ascii=False))
  else:
    summarize(rows, output_root)
    print(f"Saved {len(CONDITIONS) * len(SEEDS)}-run comparison to {output_root.resolve()}")


if __name__ == "__main__":
  main()
