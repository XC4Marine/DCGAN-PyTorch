"""Fine-tune 128-point WaveGANs at 180 kHz and compare OOD augmentation."""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, resample_poly
from scipy.stats import wasserstein_distance
import torch
from torch import autograd, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from prepare_experiment import CHECKPOINTS
from transformer_classifier import (
    ClassicTransformerClassifier, NEGATIVE_SELECTION_SEED, highpass_waveform,
    learning_rate_for_epoch, read_manifest, resolve_device, set_seed,
)


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from wavegan_crest.pytorch_wavegan import WaveGANGenerator, WaveGANDiscriminator, weights_init
from wavegan_crest.crest_loss import CrestFactorLoss
from wavegan_PSDLoss.global_spectral_loss import GlobalSpectralLoss


OOD_ROOT = Path(r"D:\Project_Github\雷洲湾标注\data\初筛")
SAMPLE_RATE = 180000
WAVEFORM_LENGTH = 40
RATIOS = (0.1, 0.3, 0.5, 0.7, 0.8, 1.0)
CLASSIFIER_SEEDS = (369, 370, 371, 372, 373)


def write_json(path, value):
  path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def progress(done, total, message):
  filled = int(30 * done / total)
  print(f"[{'#' * filled}{'-' * (30 - filled)}] {done}/{total} {message}", flush=True)


def split_ood(paths, seed):
  groups = {}
  for path in paths:
    groups.setdefault(path.stem.split("_click_")[0], []).append(path)
  rng = random.Random(seed)
  train, test = [], []
  for group in sorted(groups):
    members = groups[group]
    rng.shuffle(members)
    middle = len(members) // 2
    train.extend(members[:middle])
    test.extend(members[middle:])
  assert len(train) == 76 and len(test) == 77
  assert not set(train).intersection(test)
  return train, test


def read_waveform(path):
  sample_rate, waveform = wavfile.read(str(path))
  if waveform.ndim == 2:
    waveform = waveform.mean(axis=1)
  if np.issubdtype(waveform.dtype, np.integer):
    waveform = waveform.astype(np.float32) / float(abs(np.iinfo(waveform.dtype).min))
  else:
    waveform = waveform.astype(np.float32, copy=True)
  if sample_rate != SAMPLE_RATE:
    divisor = math.gcd(sample_rate, SAMPLE_RATE)
    waveform = resample_poly(waveform, SAMPLE_RATE // divisor, sample_rate // divisor)
  return waveform.astype(np.float32, copy=False)


def load_ood(paths):
  waveforms = []
  for path in paths:
    waveform = read_waveform(path)
    waveform /= np.max(np.abs(waveform))
    waveforms.append(np.pad(waveform, (0, WAVEFORM_LENGTH - waveform.size)))
  output = np.stack(waveforms)
  assert output.shape == (len(paths), WAVEFORM_LENGTH) and np.isfinite(output).all()
  return output


class PolyphaseDownsample(nn.Module):
  """Learnable 5/16 resampling, initially equal to SciPy's lowpass FIR."""

  def __init__(self):
    super().__init__()
    # On the 5x grid: y[5*t+p] = sum_j x[16*t+j] * h[160+16*p-5*j].
    taps = firwin(321, 1.0 / 16, window=("kaiser", 5.0)) * 5
    offsets = np.arange(-32, 45)
    indices = 160 + 16 * np.arange(5)[:, None] - 5 * offsets[None, :]
    weights = np.zeros((5, 77), dtype=np.float32)
    valid = (indices >= 0) & (indices < taps.size)
    weights[valid] = taps[indices[valid]]
    self.weight = nn.Parameter(torch.from_numpy(weights).unsqueeze(1))

  def forward(self, waveforms):
    phases = F.conv1d(F.pad(waveforms, (32, 44)), self.weight, stride=16)
    return phases.transpose(1, 2).reshape(waveforms.size(0), 1, WAVEFORM_LENGTH)


class AdaptedGenerator(nn.Module):
  def __init__(self, checkpoint):
    super().__init__()
    config = checkpoint["args"]
    self.generator = WaveGANGenerator(
        config["latent_dim"], dim=config["model_dim"], kernel_len=config["kernel_len"])
    self.generator.load_state_dict(checkpoint["generator"])
    self.generator.requires_grad_(False)
    self.generator.upconv_1.requires_grad_(True)
    self.downsample = PolyphaseDownsample()

  def forward(self, latent):
    return self.downsample(self.generator(latent))


def gradient_penalty(discriminator, real, fake):
  alpha = torch.rand(real.size(0), 1, 1, device=real.device)
  mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
  scores = discriminator(mixed)
  gradients = autograd.grad(scores, mixed, torch.ones_like(scores), create_graph=True)[0]
  return ((gradients.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def finetune(method, real, args, device, output_dir, done, total):
  set_seed(args.finetune_seed)
  checkpoint = torch.load(CHECKPOINTS[method], map_location="cpu", weights_only=False)
  config = checkpoint["args"]
  generator = AdaptedGenerator(checkpoint).to(device)
  discriminator = WaveGANDiscriminator(
      dim=config["model_dim"], kernel_len=config["kernel_len"], phase_shuffle_radius=1)
  discriminator.output = nn.Linear(config["model_dim"] * 2 * 2, 1)
  discriminator.apply(weights_init)
  discriminator.to(device)
  generator_learning_rate = config["learning_rate"] / 5
  discriminator_learning_rate = config["learning_rate"]
  g_optimizer = torch.optim.Adam(
      [p for p in generator.parameters() if p.requires_grad], lr=generator_learning_rate, betas=(0.5, 0.9))
  d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=discriminator_learning_rate, betas=(0.5, 0.9))
  crest_loss = CrestFactorLoss().to(device)
  psd_loss = GlobalSpectralLoss(n_fft=40, hop_length=10).to(device)
  loader = DataLoader(torch.from_numpy(real).unsqueeze(1), batch_size=config["batch_size"], shuffle=True)
  step = 0
  with (output_dir / "finetuning.jsonl").open("w", encoding="utf-8") as handle:
    for epoch in range(args.finetune_epochs):
      for real_batch in loader:
        real_batch = real_batch.to(device)
        batch_size = real_batch.size(0)
        discriminator.requires_grad_(True)
        with torch.no_grad():
          fake = generator(torch.randn(batch_size, config["latent_dim"], device=device))
        d_optimizer.zero_grad()
        d_loss = (discriminator(fake).mean() - discriminator(real_batch).mean()
                  + config["gradient_penalty"] * gradient_penalty(discriminator, real_batch, fake))
        assert torch.isfinite(d_loss).item()
        d_loss.backward()
        d_optimizer.step()
        record = {"epoch": epoch + 1, "step": step, "d_loss": d_loss.item()}
        if step % config["n_critic"] == 0:
          discriminator.requires_grad_(False)
          g_optimizer.zero_grad()
          fake = generator(torch.randn(batch_size, config["latent_dim"], device=device))
          adversarial = -discriminator(fake).mean()
          g_loss = adversarial
          record["adversarial_loss"] = adversarial.item()
          if config["crest_loss_weight"]:
            crest = crest_loss(real_batch, fake)
            g_loss = g_loss + config["crest_loss_weight"] * crest
            record["crest_loss"] = crest.item()
          if config["psd_loss_weight"]:
            psd = psd_loss(real_batch, fake)
            g_loss = g_loss + config["psd_loss_weight"] * psd
            record["psd_loss"] = psd.item()
          assert torch.isfinite(g_loss).item()
          g_loss.backward()
          g_optimizer.step()
          record["g_loss"] = g_loss.item()
        handle.write(json.dumps(record) + "\n")
        step += 1
      done += 1
      progress(done, total, f"{method} fine-tune {epoch + 1}/{args.finetune_epochs}")
  torch.save({
      "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
      "generator_optimizer": g_optimizer.state_dict(), "discriminator_optimizer": d_optimizer.state_dict(),
      "source_checkpoint": str(CHECKPOINTS[method]), "source_args": config,
      "epochs": args.finetune_epochs, "seed": args.finetune_seed,
      "generator_learning_rate": generator_learning_rate,
      "discriminator_learning_rate": discriminator_learning_rate,
      "sample_rate": SAMPLE_RATE, "waveform_length": WAVEFORM_LENGTH,
      "psd_n_fft": 40, "psd_hop_length": 10,
  }, output_dir / "finetuned.pt")
  set_seed(args.generation_seed)
  generator.eval()
  with torch.inference_mode():
    generated = generator(torch.randn(100, config["latent_dim"], device=device))[:, 0].cpu().numpy()
  assert generated.shape == (100, WAVEFORM_LENGTH) and np.isfinite(generated).all()
  np.save(output_dir / "generated_100.npy", generated)
  wav_dir = output_dir / "generated_wav"
  wav_dir.mkdir(exist_ok=True)
  for index, waveform in enumerate(generated, 1):
    wavfile.write(str(wav_dir / f"generated_{index:03d}.wav"), SAMPLE_RATE, waveform)
  return generated, done


def waveform_features(waveforms):
  power = np.abs(np.fft.rfft(waveforms, axis=1)) ** 2 / WAVEFORM_LENGTH
  frequencies = np.fft.rfftfreq(WAVEFORM_LENGTH, 1 / SAMPLE_RATE)
  total = np.maximum(power.sum(axis=1), 1e-12)
  centroid = (power * frequencies).sum(axis=1) / total
  peak = np.abs(waveforms).max(axis=1)
  rms = np.sqrt(np.mean(waveforms ** 2, axis=1))
  features = {
      "peak_amplitude": peak,
      "peak_to_peak": np.ptp(waveforms, axis=1),
      "rms": rms,
      "crest_factor": peak / np.maximum(rms, 1e-12),
      "peak_location_samples": np.argmax(np.abs(waveforms), axis=1),
      "dominant_frequency_hz": frequencies[np.argmax(power[:, 1:], axis=1) + 1],
      "spectral_centroid_hz": centroid,
      "spectral_bandwidth_hz": np.sqrt((power * (frequencies - centroid[:, None]) ** 2).sum(axis=1) / total),
  }
  return features, power.mean(axis=0)


def distribution_metrics(real, generated):
  real_features, real_psd = waveform_features(real)
  generated_features, generated_psd = waveform_features(generated)
  # Match shape after per-waveform peak normalization; retain raw amplitude in distribution metrics.
  real_normalized = real / np.maximum(np.abs(real).max(axis=1, keepdims=True), 1e-12)
  generated_normalized = generated / np.maximum(np.abs(generated).max(axis=1, keepdims=True), 1e-12)
  distances = np.linalg.norm(generated_normalized[:, None, :] - real_normalized[None, :, :], axis=2)
  real_distances = np.linalg.norm(real_normalized[:, None, :] - real_normalized[None, :, :], axis=2)
  np.fill_diagonal(real_distances, np.inf)
  return {
      "real_count": len(real), "generated_count": len(generated),
      "feature_wasserstein_distance": {
          name: float(wasserstein_distance(real_features[name], generated_features[name]))
          for name in real_features},
      "mean_psd_absolute_error": float(np.abs(real_psd - generated_psd).mean()),
      "mean_log10_psd_absolute_error": float(np.abs(np.log10(real_psd + 1e-12)
                                                        - np.log10(generated_psd + 1e-12)).mean()),
      "shape_similarity": {
          "metric": "nearest_neighbor_l2_over_40_samples; lower_is_closer",
          "normalization": "per_waveform_peak; no_alignment_or_sign_flip",
          "generated_to_test_mean": float(distances.min(axis=1).mean()),
          "generated_to_test_std": float(distances.min(axis=1).std()),
          "test_to_generated_mean": float(distances.min(axis=0).mean()),
          "test_to_test_excluding_self_mean": float(real_distances.min(axis=1).mean()),
      },
  }


def classifier_waveform(waveform):
  # As in the existing classifier: 20 kHz highpass, peak-centered crop/pad, peak normalization.
  waveform = highpass_waveform(waveform, SAMPLE_RATE)
  start = int(np.argmax(np.abs(waveform))) - WAVEFORM_LENGTH // 2
  source_start = max(0, start)
  source_stop = min(waveform.size, start + WAVEFORM_LENGTH)
  output = np.zeros(WAVEFORM_LENGTH, dtype=np.float32)
  offset = source_start - start
  output[offset:offset + source_stop - source_start] = waveform[source_start:source_stop]
  maximum = np.abs(output).max()
  if maximum > 0:
    output /= maximum
  assert np.isfinite(output).all()
  return output


def train_classifier(method, ratio, count, seed, real, generated, negatives, test, test_paths,
                     args, device, output_dir, done, total, source_checkpoint=None,
                     base_learning_rate=1e-4):
  set_seed(seed)
  positive = np.concatenate((real, generated[:count]))
  training = np.concatenate((positive, negatives[:len(positive)]))
  labels = torch.tensor([1] * len(positive) + [0] * len(positive), dtype=torch.long)
  loader = DataLoader(TensorDataset(torch.from_numpy(training).unsqueeze(1), labels),
                      batch_size=64, shuffle=True)
  model = ClassicTransformerClassifier(waveform_length=WAVEFORM_LENGTH).to(device)
  if source_checkpoint is not None:
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(True)
  optimizer = torch.optim.AdamW(model.parameters(), lr=base_learning_rate, weight_decay=1e-4)
  with (output_dir / "training.jsonl").open("w", encoding="utf-8") as handle:
    for epoch in range(args.classifier_epochs):
      learning_rate = learning_rate_for_epoch(epoch, args.classifier_epochs, base_learning_rate)
      for group in optimizer.param_groups:
        group["lr"] = learning_rate
      model.train()
      losses = []
      for waveforms, batch_labels in loader:
        optimizer.zero_grad()
        loss = F.cross_entropy(model(waveforms.to(device)), batch_labels.to(device))
        assert torch.isfinite(loss).item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
      handle.write(json.dumps({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                               "learning_rate": learning_rate}) + "\n")
      done += 1
      progress(done, total, f"{method} X={count} seed={seed} Transformer {epoch + 1}/{args.classifier_epochs}")
  model.eval()
  with torch.inference_mode():
    probabilities = model(torch.from_numpy(test).unsqueeze(1).to(device)).softmax(1)[:, 1].cpu().numpy()
  detected = int((probabilities >= 0.5).sum())
  result = {
      "method": method, "ratio": ratio, "generated_count": count, "seed": seed,
      "train_real_positive": len(real), "train_negative": len(positive),
      "test_positive": len(test), "detected_count": detected,
      "detection_rate": detected / len(test), "miss_rate": 1 - detected / len(test),
      "mean_positive_probability": float(probabilities.mean()), "threshold": 0.5,
      "source_checkpoint": str(source_checkpoint) if source_checkpoint is not None else None,
      "base_learning_rate": base_learning_rate,
  }
  write_json(output_dir / "metrics.json", result)
  write_json(output_dir / "predictions.json", [
      {"path": str(path), "positive_probability": float(p), "prediction": int(p >= 0.5)}
      for path, p in zip(test_paths, probabilities)])
  torch.save({"model_state_dict": model.state_dict(), "waveform_length": WAVEFORM_LENGTH,
              "sample_rate": SAMPLE_RATE, **result}, output_dir / "classifier.pt")
  return result, done


def summarize(rows, distributions, output_dir):
  summaries = []
  for method in CHECKPOINTS:
    for ratio in RATIOS:
      group = [row for row in rows if row["method"] == method and row["ratio"] == ratio]
      rates = np.array([row["detection_rate"] for row in group])
      summaries.append({
          "method": method, "ratio": ratio, "generated_count": group[0]["generated_count"],
          "runs": len(group), "detection_rate_mean": float(rates.mean()),
          "detection_rate_std": float(rates.std(ddof=1)) if len(group) > 1 else 0.0,
          "mean_positive_probability": float(np.mean([row["mean_positive_probability"] for row in group])),
      })
  for name, records in (("runs", rows), ("summary", summaries)):
    with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(records[0]))
      writer.writeheader()
      writer.writerows(records)
  write_json(output_dir / "summary.json", {"distribution_metrics": distributions, "classification": summaries})
  report = [
      "# 域外微调比较", "",
      "76 条域外正样本用于微调和分类器训练，77 条仅用于最终测试。分类阈值固定为 0.5。", "",
      "每种方法固定生成 100 条；六档增强使用同一生成池的前 X 条。", "",
      "相似度采用峰值归一化后、40 点波形的最近邻 L2 距离，越小越接近。", "",
      "| 方法 | 生成→测试 L2 | 测试→生成 L2 | 测试→测试 L2（排除自身） | 平均 log10 PSD 误差 |",
      "| --- | ---: | ---: | ---: | ---: |",
  ]
  for method, metrics in distributions.items():
    similarity = metrics["shape_similarity"]
    report.append(
        f'| {method} | {similarity["generated_to_test_mean"]:.6f} | '
        f'{similarity["test_to_generated_mean"]:.6f} | '
        f'{similarity["test_to_test_excluding_self_mean"]:.6f} | '
        f'{metrics["mean_log10_psd_absolute_error"]:.6f} |')
  report.extend([
      "", "| 方法 | 生成比例 | X | 检出率均值 | 种子间标准差 |",
      "| --- | ---: | ---: | ---: | ---: |",
  ])
  for row in summaries:
    report.append(
        f'| {row["method"]} | {row["ratio"]:.0%} | {row["generated_count"]} | '
        f'{row["detection_rate_mean"]:.4f} | {row["detection_rate_std"]:.4f} |')
  report.extend(["", "标准差来自 Transformer 的随机种子；每种生成方法只进行一次固定种子的微调。"])
  (output_dir / "EXPERIMENT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-dir", type=Path, default=OOD_ROOT)
  parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results_ood_finetuning")
  parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
  parser.add_argument("--finetune-epochs", type=int, default=200)
  parser.add_argument("--classifier-epochs", type=int, default=50)
  parser.add_argument("--split-seed", type=int, default=20260929)
  parser.add_argument("--finetune-seed", type=int, default=374)
  parser.add_argument("--generation-seed", type=int, default=369)
  parser.add_argument("--classifier-seeds", type=int, nargs="+", default=CLASSIFIER_SEEDS)
  args = parser.parse_args()
  args.output_dir = args.output_dir.resolve()
  args.data_dir = args.data_dir.resolve()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  train_paths, test_paths = split_ood(sorted(args.data_dir.glob("*.wav")), args.split_seed)
  negative_paths = read_manifest(SCRIPT_DIR / "manifests" / "train_negative_1797.txt")
  random.Random(NEGATIVE_SELECTION_SEED).shuffle(negative_paths)
  negative_paths = negative_paths[:152]
  write_json(args.output_dir / "split.json", {
      "split_seed": args.split_seed, "train_positive": train_paths, "test_positive": test_paths,
      "negative_pool": negative_paths,
  })
  write_json(args.output_dir / "configuration.json", {
      **vars(args), "checkpoints": CHECKPOINTS, "ratios": RATIOS,
      "generated_counts": [int(len(train_paths) * ratio + 0.5) for ratio in RATIOS],
      "sample_rate": SAMPLE_RATE, "waveform_length": WAVEFORM_LENGTH,
      "generated_pool_count": 100, "generator_learning_rate_scale": 0.2,
      "discriminator_learning_rate_scale": 1.0,
      "trainable_generator_layers": ["upconv_1", "downsample"],
      "phase_shuffle_radius": 1, "psd_n_fft": 40, "psd_hop_length": 10,
      "classifier_highpass_hz": 20000,
  })
  real, test = load_ood(train_paths), load_ood(test_paths)
  classifier_real = np.stack([classifier_waveform(x) for x in real])
  classifier_test = np.stack([classifier_waveform(x) for x in test])
  negatives = np.stack([classifier_waveform(read_waveform(path)) for path in negative_paths])
  device = resolve_device(args.device)
  total = len(CHECKPOINTS) * (args.finetune_epochs + len(RATIOS) * len(args.classifier_seeds) * args.classifier_epochs)
  done, rows, distributions = 0, [], {}
  progress(done, total, "starting")
  for method in CHECKPOINTS:
    method_dir = args.output_dir / method
    method_dir.mkdir(exist_ok=True)
    generated, done = finetune(method, real, args, device, method_dir, done, total)
    distributions[method] = distribution_metrics(test, generated)
    write_json(method_dir / "distribution_metrics.json", distributions[method])
    classifier_generated = np.stack([classifier_waveform(x) for x in generated])
    for ratio in RATIOS:
      count = int(len(real) * ratio + 0.5)
      for seed in args.classifier_seeds:
        run_dir = method_dir / f"g{count}" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result, done = train_classifier(
            method, ratio, count, seed, classifier_real, classifier_generated, negatives,
            classifier_test, test_paths, args, device, run_dir, done, total)
        rows.append(result)
  summarize(rows, distributions, args.output_dir)
  progress(done, total, f"complete: {args.output_dir}")


if __name__ == "__main__":
  main()
