"""Evaluate balanced augmentation classifiers on an all-positive OOD dataset."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_balanced_600_experiment import (
    CONDITIONS,
    GENERATED_COUNTS,
    METHOD_LABELS,
    METHODS,
    SEEDS,
    progress,
)
from transformer_classifier import train_one_experiment


SCRIPT_DIR = Path(__file__).resolve().parent
OOD_ROOT = Path(r"D:\Project_Github\雷洲湾标注\data\初筛_576kHz")


def flatten_result(result):
  ood = result["ood_detection"]
  return {
      "method": result["method"],
      "generated_count": result["train_generated_positive"],
      "seed": result["seed"],
      "epochs": result["epochs"],
      "train_real_positive": result["train_real_positive"],
      "train_negative": result["train_negative"],
      **ood,
  }


def paired_bootstrap(rows, left_method, left_count, right_method, right_count,
                     seed_offset, iterations=10000):
  left = {
      row["seed"]: row["detection_rate"] for row in rows
      if row["method"] == left_method and row["generated_count"] == left_count
  }
  right = {
      row["seed"]: row["detection_rate"] for row in rows
      if row["method"] == right_method and row["generated_count"] == right_count
  }
  differences = np.array([left[seed] - right[seed] for seed in sorted(left)])
  generator = np.random.default_rng(20260930 + seed_offset)
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
    rates = np.array([row["detection_rate"] for row in group])
    probabilities = np.array([row["mean_positive_probability"] for row in group])
    summary_rows.append({
        "method": method,
        "method_label": METHOD_LABELS[method],
        "generated_count": generated_count,
        "positive_count": 600 + generated_count,
        "negative_count": 600 + generated_count,
        "runs": len(group),
        "detection_rate_mean": float(rates.mean()),
        "detection_rate_std": float(rates.std(ddof=1)),
        "miss_rate_mean": float(1.0 - rates.mean()),
        "mean_positive_probability": float(probabilities.mean()),
        "mean_positive_probability_std": float(probabilities.std(ddof=1)),
    })

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
          "ood_positive_count": 153,
          "threshold": 0.5,
          "summary": summary_rows,
          "paired_comparisons": comparisons,
      }, indent=2, ensure_ascii=False),
      encoding="utf-8",
  )

  report = [
      "# 域外全正样本检出率实验",
      "",
      "域外测试集包含153条正样本，分类阈值固定为0.5。",
      "每组使用5个分类器种子；训练集唯一正负样本数严格相等。",
      "",
      "| 方法 | 合成正样本 | 训练正样本 | 训练负样本 | 检出率 | 漏检率 | 平均正类概率 |",
      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for row in summary_rows:
    report.append(
        f'| {row["method_label"]} | {row["generated_count"]} | '
        f'{row["positive_count"]} | {row["negative_count"]} | '
        f'{row["detection_rate_mean"]:.4f} +/- {row["detection_rate_std"]:.4f} | '
        f'{row["miss_rate_mean"]:.4f} | '
        f'{row["mean_positive_probability"]:.4f} +/- '
        f'{row["mean_positive_probability_std"]:.4f} |'
    )
  report.extend([
      "",
      "## 配对检出率差值",
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

  baseline = summary_rows[0]
  figure, axis = plt.subplots(figsize=(7, 4.5))
  for method in METHODS:
    method_rows = [baseline] + [
        row for row in summary_rows if row["method"] == method
    ]
    axis.errorbar(
        GENERATED_COUNTS,
        [row["detection_rate_mean"] for row in method_rows],
        yerr=[row["detection_rate_std"] for row in method_rows],
        marker="o",
        capsize=4,
        label=METHOD_LABELS[method],
    )
  axis.set_xlabel("Generated positive samples")
  axis.set_ylabel("OOD detection rate")
  axis.set_ylim(0.0, 1.02)
  axis.grid(alpha=0.25)
  axis.legend()
  figure.tight_layout()
  figure.savefig(output_dir / "ood_detection_rate.png", dpi=180)
  plt.close(figure)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument(
      "--output-dir", type=Path,
      default=SCRIPT_DIR / "results_ood_initial_screen_gpu",
  )
  parser.add_argument("--smoke-test", action="store_true")
  args = parser.parse_args()

  ood_paths = sorted(OOD_ROOT.glob("*.wav"))
  if len(ood_paths) != 153:
    raise ValueError(f"Expected 153 OOD WAV files, found {len(ood_paths)}")
  conditions = (("joint_wavegan", 1197),) if args.smoke_test else CONDITIONS
  seeds = (369,) if args.smoke_test else SEEDS
  epochs = 1 if args.smoke_test else args.epochs
  output_root = args.output_dir / "smoke" if args.smoke_test else args.output_dir
  checkpoint_root = output_root / "checkpoints"
  total = len(conditions) * len(seeds)
  completed = 0
  rows = []
  progress(completed, total, "starting")
  for method, generated_count in conditions:
    for seed in seeds:
      run_dir = output_root / "runs" / method / f"g{generated_count}" / f"seed_{seed}"
      checkpoint_path = (
          checkpoint_root / method / f"g{generated_count}" / f"seed_{seed}.pt"
      )
      metrics_path = run_dir / "metrics.json"
      if metrics_path.exists() and checkpoint_path.exists():
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        expected = (method, generated_count, 600 + generated_count, seed, epochs, 153)
        actual = (
            result["method"], result["train_generated_positive"],
            result["train_negative"], result["seed"], result["epochs"],
            result["ood_detection"]["positive_count"],
        )
        if actual != expected:
          raise ValueError(
              f"Existing result {metrics_path} has configuration {actual}; expected {expected}"
          )
      else:
        result = train_one_experiment(
            method, generated_count, seed, run_dir, args.device, epochs, 600,
            ood_paths, checkpoint_path,
        )
      rows.append(flatten_result(result))
      completed += 1
      progress(completed, total, f"finished {method} G{generated_count} seed {seed}")

  if args.smoke_test:
    print(json.dumps(rows[0], indent=2, ensure_ascii=False))
  else:
    summarize(rows, output_root)
    print(f"Saved {len(CONDITIONS) * len(SEEDS)} OOD runs to {output_root.resolve()}")


if __name__ == "__main__":
  main()
