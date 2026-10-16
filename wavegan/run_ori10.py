"""Train baseline WaveGAN on Ori_file_num=10 for 200,000 update steps."""

import argparse
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  parser.add_argument('--seed', type=int, default=369)
  parser.add_argument('--num-workers', type=int, default=0)
  args = parser.parse_args()

  checkpoint_dir = SCRIPT_DIR / 'checkpoints' / 'ori_file_10_full'
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  subprocess.run([
      sys.executable, str(SCRIPT_DIR / 'train_wavegan_torch.py'),
      '--data-dir', str(PROJECT_ROOT / 'data' / 'wav'),
      '--metadata-file', str(PROJECT_ROOT / 'data' / 'ClickTrains.csv'),
      '--ori-file-num', '10',
      '--output-dir', str(checkpoint_dir),
      '--max-update-steps', '200000',
      '--checkpoint-every', '50',
      '--batch-size', str(args.batch_size),
      '--device', args.device,
      '--seed', str(args.seed),
      '--num-workers', str(args.num_workers),
  ], cwd=str(SCRIPT_DIR), check=True)


if __name__ == '__main__':
  main()
