"""Train and evaluate one classic Transformer augmentation condition."""

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt
from scipy.stats import rankdata
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MANIFEST_DIR = SCRIPT_DIR / "manifests"
TARGET_SAMPLE_RATE = 576000
WAVEFORM_LENGTH = 128
NEGATIVE_COUNT = 1797
HIGHPASS_CUTOFF_HZ = 20000
HIGHPASS_ORDER = 8
REAL_SELECTION_SEED = 20260929
NEGATIVE_SELECTION_SEED = 20260929


def read_manifest(path):
  return [PROJECT_ROOT / line for line in path.read_text(encoding="utf-8").splitlines() if line]


def highpass_waveform(waveform, sample_rate):
  sos = butter(
      HIGHPASS_ORDER,
      HIGHPASS_CUTOFF_HZ,
      btype="highpass",
      fs=sample_rate,
      output="sos",
  )
  return sosfiltfilt(sos, waveform).astype(np.float32, copy=False)


def preprocess_waveform(path, source):
  sample_rate, waveform = wavfile.read(path)
  if waveform.ndim == 2:
    waveform = waveform.mean(axis=1)
  if np.issubdtype(waveform.dtype, np.integer):
    waveform = waveform.astype(np.float32) / float(abs(np.iinfo(waveform.dtype).min))
  else:
    waveform = waveform.astype(np.float32, copy=True)

  if source != "other_species" and sample_rate != TARGET_SAMPLE_RATE:
    divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
    waveform = resample_poly(
        waveform, TARGET_SAMPLE_RATE // divisor, sample_rate // divisor
    ).astype(np.float32, copy=False)
    sample_rate = TARGET_SAMPLE_RATE

  waveform = highpass_waveform(waveform, sample_rate)

  peak = int(np.argmax(np.abs(waveform)))
  start = peak - WAVEFORM_LENGTH // 2
  stop = start + WAVEFORM_LENGTH
  output = np.zeros(WAVEFORM_LENGTH, dtype=np.float32)
  source_start = max(0, start)
  source_stop = min(waveform.size, stop)
  destination_start = source_start - start
  output[destination_start:destination_start + source_stop - source_start] = waveform[source_start:source_stop]
  maximum = float(np.max(np.abs(output)))
  if maximum > 0:
    output /= maximum
  if not np.isfinite(output).all():
    raise ValueError(f"Non-finite waveform after preprocessing: {path}")
  return output


def negative_source(path):
  return "background_transient" if path.name.startswith("FalseClick_") else "other_species"


class WaveformDataset(Dataset):
  def __init__(self, records):
    self.records = records
    self.waveforms = torch.from_numpy(
        np.stack([preprocess_waveform(path, source) for path, _, source in records])
    ).unsqueeze(1)
    self.labels = torch.tensor([label for _, label, _ in records], dtype=torch.long)
    self.sources = [source for _, _, source in records]
    if self.waveforms.shape[1:] != (1, WAVEFORM_LENGTH):
      raise ValueError(f"Unexpected waveform tensor shape: {tuple(self.waveforms.shape)}")

  def __len__(self):
    return len(self.records)

  def __getitem__(self, index):
    return self.waveforms[index], self.labels[index], index


class ExactBalancedSampler(Sampler):
  def __init__(self, positive_indices, negative_indices, seed):
    self.positive_indices = torch.tensor(positive_indices, dtype=torch.long)
    self.negative_indices = torch.tensor(negative_indices, dtype=torch.long)
    self.seed = seed
    self.epoch = 0

  def set_epoch(self, epoch):
    self.epoch = epoch

  def __len__(self):
    return 2 * len(self.negative_indices)

  def __iter__(self):
    generator = torch.Generator().manual_seed(self.seed + self.epoch)
    negative = self.negative_indices[torch.randperm(len(self.negative_indices), generator=generator)]
    if len(self.positive_indices) >= len(self.negative_indices):
      order = torch.randperm(len(self.positive_indices), generator=generator)[:len(self.negative_indices)]
      positive = self.positive_indices[order]
    else:
      order = torch.randint(
          len(self.positive_indices), (len(self.negative_indices),), generator=generator
      )
      positive = self.positive_indices[order]
    combined = torch.cat((positive, negative))
    combined = combined[torch.randperm(len(combined), generator=generator)]
    return iter(combined.tolist())


class SinusoidalPositionEncoding(nn.Module):
  def __init__(self, d_model, max_length):
    super().__init__()
    positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    encoding = torch.zeros(max_length, d_model)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

  def forward(self, inputs):
    return inputs + self.encoding[:, :inputs.size(1)]


class ClassicTransformerClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    d_model = 128
    self.input_projection = nn.Linear(1, d_model)
    self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
    self.position = SinusoidalPositionEncoding(d_model, WAVEFORM_LENGTH + 1)
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=4,
        dim_feedforward=512,
        dropout=0.1,
        activation="relu",
        batch_first=True,
        norm_first=False,
    )
    self.encoder = nn.TransformerEncoder(layer, num_layers=4, norm=nn.LayerNorm(d_model))
    self.classifier = nn.Linear(d_model, 2)
    self._reset_parameters()
    nn.init.normal_(self.class_token, mean=0.0, std=0.02)

  def _reset_parameters(self):
    for name, parameter in self.named_parameters():
      if parameter.dim() > 1:
        nn.init.xavier_uniform_(parameter)
      elif name.endswith("bias"):
        nn.init.zeros_(parameter)

  def forward(self, waveforms):
    tokens = self.input_projection(waveforms.transpose(1, 2))
    class_token = self.class_token.expand(waveforms.size(0), -1, -1)
    encoded = self.encoder(self.position(torch.cat((class_token, tokens), dim=1)))
    return self.classifier(encoded[:, 0])


def binary_metrics(labels, probabilities, predictions, sources):
  labels = np.asarray(labels, dtype=np.int64)
  probabilities = np.asarray(probabilities, dtype=np.float64)
  predictions = np.asarray(predictions, dtype=np.int64)
  tp = int(np.sum((labels == 1) & (predictions == 1)))
  tn = int(np.sum((labels == 0) & (predictions == 0)))
  fp = int(np.sum((labels == 0) & (predictions == 1)))
  fn = int(np.sum((labels == 1) & (predictions == 0)))

  positive_precision = tp / (tp + fp) if tp + fp else 0.0
  positive_recall = tp / (tp + fn)
  positive_f1 = (
      2 * positive_precision * positive_recall / (positive_precision + positive_recall)
      if positive_precision + positive_recall else 0.0
  )
  negative_precision = tn / (tn + fn) if tn + fn else 0.0
  specificity = tn / (tn + fp)
  negative_f1 = (
      2 * negative_precision * specificity / (negative_precision + specificity)
      if negative_precision + specificity else 0.0
  )

  positive_count = int(np.sum(labels == 1))
  negative_count = int(np.sum(labels == 0))
  ranks = rankdata(probabilities, method="average")
  auroc = (
      (float(ranks[labels == 1].sum()) - positive_count * (positive_count + 1) / 2)
      / (positive_count * negative_count)
  )
  order = np.argsort(-probabilities, kind="stable")
  sorted_labels = labels[order]
  cumulative_true = np.cumsum(sorted_labels == 1)
  precision_curve = cumulative_true / np.arange(1, len(labels) + 1)
  average_precision = float(precision_curve[sorted_labels == 1].sum() / positive_count)

  subgroup_specificity = {}
  sources = np.asarray(sources)
  for source in ("background_transient", "other_species"):
    mask = sources == source
    subgroup_specificity[source] = float(np.mean(predictions[mask] == 0))

  return {
      "macro_f1": (positive_f1 + negative_f1) / 2,
      "balanced_accuracy": (positive_recall + specificity) / 2,
      "auroc": auroc,
      "auprc": average_precision,
      "positive_precision": positive_precision,
      "positive_recall": positive_recall,
      "positive_f1": positive_f1,
      "specificity": specificity,
      "background_specificity": subgroup_specificity["background_transient"],
      "other_species_specificity": subgroup_specificity["other_species"],
      "tp": tp,
      "tn": tn,
      "fp": fp,
      "fn": fn,
  }


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def resolve_device(device_name):
  if device_name == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return torch.device(device_name)


def learning_rate_for_epoch(epoch, epochs, base_learning_rate=1e-4, warmup_epochs=5):
  if epoch < warmup_epochs:
    return base_learning_rate * (epoch + 1) / warmup_epochs
  progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1)
  factor = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
  return base_learning_rate * factor


def train_one_experiment(method, generated_count, seed, output_dir, device_name="auto", epochs=50,
                         real_count=600, ood_positive_paths=None, checkpoint_path=None):
  set_seed(seed)
  device = resolve_device(device_name)
  real_positive_pool = read_manifest(MANIFEST_DIR / "real_positive_600.txt")
  random.Random(REAL_SELECTION_SEED).shuffle(real_positive_pool)
  real_positive = real_positive_pool[:real_count]
  train_negative_pool = read_manifest(MANIFEST_DIR / "train_negative_1797.txt")
  test_positive = read_manifest(MANIFEST_DIR / "test_positive_155.txt")
  test_negative = read_manifest(MANIFEST_DIR / "test_negative_203.txt")
  if method == "baseline":
    if generated_count != 0:
      raise ValueError("The baseline condition cannot contain generated samples")
    generated = []
  else:
    generated = read_manifest(MANIFEST_DIR / f"generated_{method}_1200.txt")[:generated_count]
  positive_count = real_count + generated_count
  random.Random(NEGATIVE_SELECTION_SEED).shuffle(train_negative_pool)
  train_negative = train_negative_pool[:positive_count]
  if len(real_positive_pool) != 600 or len(real_positive) != real_count:
    raise ValueError("Real-positive manifest does not support the requested subset")
  if len(train_negative_pool) != NEGATIVE_COUNT:
    raise ValueError("Training-negative manifest does not contain 1797 files")
  if len(train_negative) != positive_count:
    raise ValueError("Not enough negatives for a balanced unique training set")
  if len(generated) != generated_count:
    raise ValueError(f"Requested {generated_count} generated files, found {len(generated)}")

  training_records = (
      [(path, 1, "real_positive") for path in real_positive]
      + [(path, 1, "generated_positive") for path in generated]
      + [(path, 0, negative_source(path)) for path in train_negative]
  )
  test_records = (
      [(path, 1, "positive") for path in test_positive]
      + [(path, 0, negative_source(path)) for path in test_negative]
  )
  train_dataset = WaveformDataset(training_records)
  test_dataset = WaveformDataset(test_records)
  positive_indices = list(range(real_count + generated_count))
  negative_indices = list(range(real_count + generated_count, len(train_dataset)))
  sampler = ExactBalancedSampler(positive_indices, negative_indices, seed)
  train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=0)
  test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

  model = ClassicTransformerClassifier().to(device)
  test_shape = model(torch.zeros(2, 1, WAVEFORM_LENGTH, device=device)).shape
  if tuple(test_shape) != (2, 2):
    raise ValueError(f"Unexpected classifier output shape: {tuple(test_shape)}")
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
  loss_function = nn.CrossEntropyLoss()
  epoch_records = []
  for epoch in range(epochs):
    sampler.set_epoch(epoch)
    learning_rate = learning_rate_for_epoch(epoch, epochs)
    for group in optimizer.param_groups:
      group["lr"] = learning_rate
    model.train()
    losses = []
    for waveforms, labels, _ in train_loader:
      waveforms = waveforms.to(device, non_blocking=True)
      labels = labels.to(device, non_blocking=True)
      optimizer.zero_grad()
      loss = loss_function(model(waveforms), labels)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      losses.append(loss.item())
    record = {
        "epoch": epoch + 1,
        "loss": float(np.mean(losses)),
        "learning_rate": learning_rate,
    }
    epoch_records.append(record)
    print(
        f"  method={method:13s} generated={generated_count:4d} seed={seed} "
        f"epoch={epoch + 1:02d}/{epochs} loss={record['loss']:.6f}",
        flush=True,
    )

  model.eval()
  all_labels = []
  all_probabilities = []
  all_predictions = []
  all_indices = []
  with torch.inference_mode():
    for waveforms, labels, indices in test_loader:
      probabilities = torch.softmax(model(waveforms.to(device)), dim=1)[:, 1].cpu()
      predictions = (probabilities >= 0.5).to(torch.long)
      all_labels.extend(labels.tolist())
      all_probabilities.extend(probabilities.tolist())
      all_predictions.extend(predictions.tolist())
      all_indices.extend(indices.tolist())
  sources = [test_dataset.sources[index] for index in all_indices]
  metrics = binary_metrics(all_labels, all_probabilities, all_predictions, sources)
  ood_detection = None
  ood_rows = []
  if ood_positive_paths is not None:
    ood_dataset = WaveformDataset([
        (path, 1, "positive") for path in ood_positive_paths
    ])
    ood_loader = DataLoader(ood_dataset, batch_size=128, shuffle=False, num_workers=0)
    with torch.inference_mode():
      for waveforms, _, indices in ood_loader:
        probabilities = torch.softmax(model(waveforms.to(device)), dim=1)[:, 1].cpu()
        predictions = (probabilities >= 0.5).to(torch.long)
        for index, probability, prediction in zip(
            indices.tolist(), probabilities.tolist(), predictions.tolist()
        ):
          ood_rows.append({
              "file": ood_dataset.records[index][0].name,
              "probability_positive": probability,
              "prediction": prediction,
          })
    detected = sum(row["prediction"] for row in ood_rows)
    ood_detection = {
        "positive_count": len(ood_rows),
        "detected_count": detected,
        "missed_count": len(ood_rows) - detected,
        "detection_rate": detected / len(ood_rows),
        "miss_rate": 1.0 - detected / len(ood_rows),
        "mean_positive_probability": float(np.mean([
            row["probability_positive"] for row in ood_rows
        ])),
    }
  result = {
      "method": method,
      "generated_count": generated_count,
      "seed": seed,
      "epochs": epochs,
      "device": str(device),
      "train_real_positive": real_count,
      "train_generated_positive": generated_count,
      "train_negative": len(train_negative),
      "balanced_samples_per_epoch": 2 * len(train_negative),
      "test_positive": len(test_positive),
      "test_negative": len(test_negative),
      "preprocessing": {
          "other_species_resampled": False,
          "other_sources_sample_rate_hz": TARGET_SAMPLE_RATE,
          "highpass_cutoff_hz": HIGHPASS_CUTOFF_HZ,
          "highpass_order": HIGHPASS_ORDER,
          "highpass_zero_phase": True,
          "waveform_length": WAVEFORM_LENGTH,
          "normalization": "instance_peak",
      },
      "metrics": metrics,
  }
  if ood_detection is not None:
    result["ood_detection"] = ood_detection
  output_dir.mkdir(parents=True, exist_ok=True)
  if checkpoint_path is not None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "method": method,
        "generated_count": generated_count,
        "real_count": real_count,
        "seed": seed,
        "epochs": epochs,
        "model_state_dict": {
            name: parameter.detach().cpu()
            for name, parameter in model.state_dict().items()
        },
    }, checkpoint_path)
    result["checkpoint"] = str(checkpoint_path.resolve())
  (output_dir / "metrics.json").write_text(
      json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
  )
  with (output_dir / "training.jsonl").open("w", encoding="utf-8") as handle:
    for record in epoch_records:
      handle.write(json.dumps(record, ensure_ascii=False) + "\n")
  with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
    for index, label, probability, prediction, source in zip(
        all_indices, all_labels, all_probabilities, all_predictions, sources
    ):
      path = test_dataset.records[index][0]
      handle.write(json.dumps({
          "file": path.name,
          "label": label,
          "probability_positive": probability,
          "prediction": prediction,
          "source": source,
       }, ensure_ascii=False) + "\n")
  if ood_rows:
    with (output_dir / "ood_predictions.jsonl").open("w", encoding="utf-8") as handle:
      for row in ood_rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--method", choices=("baseline", "raw_wavegan", "joint_wavegan"), required=True
  )
  parser.add_argument("--generated-count", type=int, choices=(0, 300, 600, 900, 1200), required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--real-count", type=int, default=600)
  args = parser.parse_args()
  result = train_one_experiment(
      args.method, args.generated_count, args.seed, args.output_dir, args.device, args.epochs,
      args.real_count,
  )
  print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
