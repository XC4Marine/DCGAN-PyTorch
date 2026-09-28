"""Refresh existing metrics files after adding -3 dB bandwidth distance."""

import json
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EVALUATOR = SCRIPT_DIR / 'evaluate_wavegan_torch.py'


def refresh_items():
  items = [(PROJECT_ROOT / 'wavegan' / 'train_click_wavegan_torch_gpu' / 'wavegan_epoch_0200.pt',
            SCRIPT_DIR / 'controls' / 'raw_wavegan')]
  for family, source_root in (
      ('crest_only', PROJECT_ROOT / 'wavegan_crest' / 'checkpoints'),
      ('psd_only', PROJECT_ROOT / 'wavegan_PSDLoss' / 'checkpoints')):
    for metrics_path in sorted((SCRIPT_DIR / 'controls' / family).glob('weight_*/metrics.json')):
      items.append((source_root / metrics_path.parent.name / 'wavegan_epoch_0200.pt', metrics_path.parent))
  for metrics_path in sorted((SCRIPT_DIR / 'runs').glob('crest_*/seed_*/metrics.json')):
    items.append((metrics_path.parent / 'checkpoints' / 'wavegan_epoch_0200.pt', metrics_path.parent))
  return items


def needs_refresh(destination):
  metrics_path = destination / 'metrics.json'
  with metrics_path.open(encoding='utf-8') as handle:
    metrics = json.load(handle)
  return 'minus_3db_bandwidth_hz' not in metrics['feature_wasserstein_distance']


def main():
  pending = [(checkpoint, destination) for checkpoint, destination in refresh_items()
             if needs_refresh(destination)]
  for index, (checkpoint, destination) in enumerate(pending, 1):
    with (destination / 'evaluation.log').open('w', encoding='utf-8') as log_file:
      subprocess.run([
          sys.executable, str(EVALUATOR), '--device', 'cuda', '--checkpoint', str(checkpoint),
          '--output-dir', str(destination / 'images'), '--metrics-output', str(destination / 'metrics.json'),
          '--evaluation-seed', '369'], check=True, stdout=log_file, stderr=subprocess.STDOUT)
    filled = round(30 * index / len(pending))
    print('Overall progress: [{}] {:6.2f}% | re-evaluated {}/{}'.format(
        '#' * filled + '-' * (30 - filled), 100 * index / len(pending), index, len(pending)), flush=True)


if __name__ == '__main__':
  main()
