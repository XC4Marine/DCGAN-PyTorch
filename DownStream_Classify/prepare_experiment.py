"""Prepare the leakage-free WaveGAN augmentation experiment.

This script rebuilds the downstream split around Voyage_Ori_15, selects the
fixed 600-real-positive subset, and generates matched nested pools of 1,200
raw and PSD+Crest WaveGAN samples. Run it once before
``run_balanced_600_experiment.py``.
"""

import argparse
import json
import math
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from scipy.io import wavfile
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_ROOT = SCRIPT_DIR / "data" / "tobeClassify"
POSITIVE_SOURCE = PROJECT_ROOT / "data" / "wav" / "click_waveform"
LOCATE_ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate")
WAVEGAN_SPLIT = PROJECT_ROOT / "wavegan_PSD_crest" / "voyage_split_full" / "split" / "split.json"
CHECKPOINTS = {
    "raw_wavegan": (
        PROJECT_ROOT / "wavegan_PSD_crest" / "voyage_split_full" / "runs"
        / "raw_wavegan" / "seed_374" / "checkpoints" / "wavegan_epoch_0200.pt"
    ),
    "joint_wavegan": (
        PROJECT_ROOT / "wavegan_PSD_crest" / "voyage_split_full" / "runs"
        / "joint_best" / "seed_374" / "checkpoints" / "wavegan_epoch_0200.pt"
    ),
}
MANIFEST_DIR = SCRIPT_DIR / "manifests"
GENERATED_DIRS = {
    method: SCRIPT_DIR / "data" / "generated" / f"{method}_seed374"
    for method in CHECKPOINTS
}
TEST_VOYAGE = "Voyage_Ori_15"
TEST_NEGATIVE_VOYAGES = {"Voyage_Ori_08", "Voyage_Ori_15"}
SELECTION_SEED = 20260929


def progress(done, total, message):
  width = 30
  filled = int(width * done / total)
  bar = "#" * filled + "-" * (width - filled)
  print(f"\r[{bar}] {done}/{total} {message}", end="", flush=True)
  if done == total:
    print()


def pulse_train_from_name(name):
  parts = Path(name).stem.split("_Pulse_")
  if len(parts) != 2 or not parts[0].startswith("PulseTrain_"):
    raise ValueError(f"Cannot extract PulseTrain from {name}")
  return parts[0]


def build_voyage_maps(locate_root):
  pulse_to_voyage = {}
  recording_to_voyage = {}
  for voyage_dir in sorted(locate_root.glob("Voyage_Ori_*")):
    if not voyage_dir.is_dir():
      continue
    voyage = voyage_dir.name
    for pulse_dir in voyage_dir.glob("PulseTrain_*"):
      if pulse_dir.name in pulse_to_voyage:
        raise ValueError(f"Duplicate PulseTrain mapping: {pulse_dir.name}")
      pulse_to_voyage[pulse_dir.name] = voyage

    suffix = voyage.removeprefix("Voyage_Ori_")
    bounds = suffix.split("-")
    start = int(bounds[0])
    end = int(bounds[-1])
    for recording in range(start, end + 1):
      recording_to_voyage[recording] = voyage
  return pulse_to_voyage, recording_to_voyage


def positive_voyage(path, pulse_to_voyage):
  pulse_train = pulse_train_from_name(path.name)
  if pulse_train not in pulse_to_voyage:
    raise ValueError(f"No voyage mapping for {pulse_train}")
  return pulse_to_voyage[pulse_train]


def false_click_recording(path):
  prefix = "FalseClick_Ori_Recording_"
  if not path.name.startswith(prefix):
    return None
  return int(path.name[len(prefix):].split("_", 1)[0])


def other_audio_group(path):
  marker = "_click_"
  if marker not in path.stem:
    raise ValueError(f"Cannot extract source-audio group from {path.name}")
  return path.stem.split(marker, 1)[0]


def select_test_other_species(paths, count, seed):
  rng = random.Random(seed)
  by_audio = {}
  for path in paths:
    by_audio.setdefault(other_audio_group(path), []).append(path)
  for group_paths in by_audio.values():
    rng.shuffle(group_paths)

  selected = []
  groups = sorted(by_audio)
  while len(selected) < count:
    for group in groups:
      if by_audio[group] and len(selected) < count:
        selected.append(by_audio[group].pop())
  return set(selected)


def move_to(path, destination_dir):
  destination = destination_dir / path.name
  if path.parent == destination_dir:
    return
  if destination.exists():
    raise FileExistsError(f"Destination already exists: {destination}")
  shutil.move(str(path), str(destination))


def rebuild_split(pulse_to_voyage, recording_to_voyage):
  train_positive = DATASET_ROOT / "train" / "positive"
  train_negative = DATASET_ROOT / "train" / "negative"
  test_positive = DATASET_ROOT / "test" / "positive"
  test_negative = DATASET_ROOT / "test" / "negative"
  for directory in (train_positive, train_negative, test_positive, test_negative):
    directory.mkdir(parents=True, exist_ok=True)

  current_positive = list(train_positive.glob("*.wav")) + list(test_positive.glob("*.wav"))
  for path in current_positive:
    destination = test_positive if positive_voyage(path, pulse_to_voyage) == TEST_VOYAGE else train_positive
    move_to(path, destination)

  voyage_15_sources = [
      path for path in POSITIVE_SOURCE.glob("*.wav")
      if positive_voyage(path, pulse_to_voyage) == TEST_VOYAGE
  ]
  if len(voyage_15_sources) != 155:
    raise ValueError(f"Expected 155 Voyage_Ori_15 positives, found {len(voyage_15_sources)}")
  for source in voyage_15_sources:
    destination = test_positive / source.name
    if not destination.exists():
      shutil.copy2(source, destination)

  current_negative = list(train_negative.glob("*.wav")) + list(test_negative.glob("*.wav"))
  other_paths = [path for path in current_negative if false_click_recording(path) is None]
  test_other_paths = select_test_other_species(other_paths, 15, SELECTION_SEED)
  for path in current_negative:
    recording = false_click_recording(path)
    if recording is not None:
      is_test = recording_to_voyage[recording] in TEST_NEGATIVE_VOYAGES
    else:
      is_test = path in test_other_paths
    move_to(path, test_negative if is_test else train_negative)

  train_positive_paths = sorted(train_positive.glob("*.wav"))
  test_positive_paths = sorted(test_positive.glob("*.wav"))
  train_negative_paths = sorted(train_negative.glob("*.wav"))
  test_negative_paths = sorted(test_negative.glob("*.wav"))
  if len(train_positive_paths) != 2000:
    raise ValueError(f"Expected 2000 positive-pool files, found {len(train_positive_paths)}")
  if len(test_positive_paths) != 155:
    raise ValueError(f"Expected 155 test positives, found {len(test_positive_paths)}")
  if len(train_negative_paths) != 1797:
    raise ValueError(f"Expected 1797 train negatives, found {len(train_negative_paths)}")
  if len(test_negative_paths) != 203:
    raise ValueError(f"Expected 203 test negatives, found {len(test_negative_paths)}")

  train_voyages = {positive_voyage(path, pulse_to_voyage) for path in train_positive_paths}
  test_voyages = {positive_voyage(path, pulse_to_voyage) for path in test_positive_paths}
  for path in train_negative_paths:
    recording = false_click_recording(path)
    if recording is not None:
      train_voyages.add(recording_to_voyage[recording])
  for path in test_negative_paths:
    recording = false_click_recording(path)
    if recording is not None:
      test_voyages.add(recording_to_voyage[recording])
  if train_voyages & test_voyages:
    raise ValueError(f"Voyage leakage: {sorted(train_voyages & test_voyages)}")

  train_audio_groups = {pulse_train_from_name(path.name) for path in train_positive_paths}
  test_audio_groups = {pulse_train_from_name(path.name) for path in test_positive_paths}
  for path in train_negative_paths:
    recording = false_click_recording(path)
    if recording is not None:
      train_audio_groups.add(f"recording_{recording:02d}")
  for path in test_negative_paths:
    recording = false_click_recording(path)
    if recording is not None:
      test_audio_groups.add(f"recording_{recording:02d}")
  if train_audio_groups & test_audio_groups:
    raise ValueError(f"Source-audio leakage: {sorted(train_audio_groups & test_audio_groups)}")

  split = json.loads(WAVEGAN_SPLIT.read_text(encoding="utf-8"))
  if not TEST_NEGATIVE_VOYAGES.issubset(split["test_voyages"]):
    raise ValueError("Downstream test voyages are not independent of the WaveGAN training split")

  return train_positive_paths, train_negative_paths, test_positive_paths, test_negative_paths


def proportional_quotas(grouped_paths, total):
  population = sum(len(paths) for paths in grouped_paths.values())
  raw = {group: total * len(paths) / population for group, paths in grouped_paths.items()}
  quotas = {group: math.floor(value) for group, value in raw.items()}
  remaining = total - sum(quotas.values())
  order = sorted(raw, key=lambda group: (-(raw[group] - quotas[group]), group))
  for group in order[:remaining]:
    quotas[group] += 1
  return quotas


def select_real_positives(paths, pulse_to_voyage, count, seed):
  rng = random.Random(seed)
  by_voyage = {}
  for path in paths:
    voyage = positive_voyage(path, pulse_to_voyage)
    by_voyage.setdefault(voyage, []).append(path)
  quotas = proportional_quotas(by_voyage, count)

  selected = []
  for voyage in sorted(by_voyage):
    by_pulse = {}
    for path in by_voyage[voyage]:
      by_pulse.setdefault(pulse_train_from_name(path.name), []).append(path)
    pulse_names = sorted(by_pulse)
    rng.shuffle(pulse_names)
    for pulse in pulse_names:
      rng.shuffle(by_pulse[pulse])

    voyage_selection = []
    while len(voyage_selection) < quotas[voyage]:
      made_progress = False
      for pulse in pulse_names:
        if by_pulse[pulse] and len(voyage_selection) < quotas[voyage]:
          voyage_selection.append(by_pulse[pulse].pop())
          made_progress = True
      if not made_progress:
        raise ValueError(f"Not enough samples in {voyage}")
    selected.extend(voyage_selection)
  if len(selected) != count:
    raise ValueError(f"Expected {count} selected positives, found {len(selected)}")
  return sorted(selected), quotas


def relative_path(path):
  return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def write_manifest(paths, destination):
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text("\n".join(relative_path(path) for path in paths) + "\n", encoding="utf-8")


def generate_waveforms(method, checkpoint_path, count, seed, batch_size, device_name):
  sys.path.insert(0, str(PROJECT_ROOT / "wavegan"))
  from pytorch_wavegan import WaveGANGenerator

  device = torch.device(
      "cuda" if device_name == "auto" and torch.cuda.is_available()
      else "cpu" if device_name == "auto"
      else device_name
  )
  checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
  arguments = checkpoint["args"]
  generator = WaveGANGenerator(
      latent_dim=arguments["latent_dim"],
      dim=arguments["model_dim"],
      kernel_len=arguments["kernel_len"],
  ).to(device)
  generator.load_state_dict(checkpoint["generator"])
  generator.eval()

  generated_dir = GENERATED_DIRS[method]
  generated_dir.mkdir(parents=True, exist_ok=True)
  latent_generator = torch.Generator(device="cpu").manual_seed(seed)
  latent = torch.randn(count, arguments["latent_dim"], generator=latent_generator)
  generated_paths = []
  done = 0
  with torch.inference_mode():
    for start in range(0, count, batch_size):
      waveforms = generator(latent[start:start + batch_size].to(device)).cpu().numpy()[:, 0]
      for waveform in waveforms:
        if not np.isfinite(waveform).all() or waveform.shape != (128,):
          raise ValueError("WaveGAN produced an invalid waveform")
        done += 1
        output = generated_dir / f"{method}_seed374_latent_{done:04d}.wav"
        wavfile.write(output, int(arguments["sample_rate"]), waveform.astype(np.float32))
        generated_paths.append(output)
      progress(done, count, f"generating {method} positives")
  return generated_paths


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
  parser.add_argument("--generation-batch-size", type=int, default=64)
  parser.add_argument("--skip-generation", action="store_true")
  args = parser.parse_args()

  pulse_to_voyage, recording_to_voyage = build_voyage_maps(LOCATE_ROOT)
  train_positive, train_negative, test_positive, test_negative = rebuild_split(
      pulse_to_voyage, recording_to_voyage
  )
  real_positive, quotas = select_real_positives(
      train_positive, pulse_to_voyage, 600, SELECTION_SEED
  )

  write_manifest(real_positive, MANIFEST_DIR / "real_positive_600.txt")
  write_manifest(train_negative, MANIFEST_DIR / "train_negative_1797.txt")
  write_manifest(test_positive, MANIFEST_DIR / "test_positive_155.txt")
  write_manifest(test_negative, MANIFEST_DIR / "test_negative_203.txt")

  generated_paths = {method: [] for method in CHECKPOINTS}
  if not args.skip_generation:
    for method, checkpoint in CHECKPOINTS.items():
      generated_paths[method] = generate_waveforms(
          method, checkpoint, 1200, SELECTION_SEED, args.generation_batch_size, args.device
      )
      write_manifest(
          generated_paths[method], MANIFEST_DIR / f"generated_{method}_1200.txt"
      )

  test_background = sum(false_click_recording(path) is not None for path in test_negative)
  metadata = {
      "selection_seed": SELECTION_SEED,
      "latent_seed": SELECTION_SEED,
      "test_voyage": TEST_VOYAGE,
      "test_negative_voyages": sorted(TEST_NEGATIVE_VOYAGES),
      "other_species_split": "sample_level_stratified_round_robin",
      "nested_generated_counts": [300, 600, 900, 1200],
      "wavegan_checkpoints": {
          method: relative_path(checkpoint) for method, checkpoint in CHECKPOINTS.items()
      },
      "real_positive_quotas_by_voyage": quotas,
      "counts": {
          "positive_pool": len(train_positive),
          "selected_real_positive": len(real_positive),
          "train_negative": len(train_negative),
          "test_positive": len(test_positive),
          "test_negative": len(test_negative),
          "test_negative_background": test_background,
          "test_negative_other_species": len(test_negative) - test_background,
          "generated_raw_wavegan": len(generated_paths["raw_wavegan"]),
          "generated_joint_wavegan": len(generated_paths["joint_wavegan"]),
      },
  }
  MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
  (MANIFEST_DIR / "experiment.json").write_text(
      json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
  )
  print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
