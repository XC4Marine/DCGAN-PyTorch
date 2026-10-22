"""Train Ori10 WaveGAN without Phase Shuffle, then export final checkpoint waveforms."""

import argparse
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints' / 'ori_file_10_no_phase_shuffle'
EXPORT_DIR = Path('D:/Project_Github/VGGish_Evaluation/WaveGANdata_no_phase_shuffle')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  parser.add_argument('--seed', type=int, default=369)
  parser.add_argument('--num-workers', type=int, default=0)
  args = parser.parse_args()

  CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
  subprocess.run([
      sys.executable, str(SCRIPT_DIR / 'train_wavegan_torch.py'),
      '--data-dir', str(PROJECT_ROOT / 'data' / 'wav'),
      '--metadata-file', str(PROJECT_ROOT / 'data' / 'ClickTrains.csv'),
      '--ori-file-num', '10',
      '--output-dir', str(CHECKPOINT_DIR),
      '--max-update-steps', '200000',
      '--checkpoint-every', '50',
      '--keep-last-checkpoints', '5',
      '--batch-size', str(args.batch_size),
      '--phase-shuffle-radius', '0',
      '--device', args.device,
      '--seed', str(args.seed),
      '--num-workers', str(args.num_workers),
  ], cwd=str(SCRIPT_DIR), check=True)
  subprocess.run([
      sys.executable, str(SCRIPT_DIR / 'export_last_five_checkpoints.py'),
      '--checkpoint-dir', str(CHECKPOINT_DIR),
      '--output-dir', str(EXPORT_DIR),
      '--batch-size', str(args.batch_size),
      '--device', args.device,
  ], cwd=str(SCRIPT_DIR), check=True)


if __name__ == '__main__':
  main()
