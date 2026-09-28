"""Run the fixed voyage-disjoint 80/20 experiment for fifteen training seeds."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIGURATIONS = (
    ('raw_wavegan', 0.0, 0.0),
    ('crest_only_peak', 1.0, 0.0),
    ('psd_only_log_psd', 0.0, 0.4),
    ('joint_best', 0.6, 0.15),
)
SEEDS = tuple(range(369, 384))


def progress(done, total, label):
  width = 30
  filled = round(width * done / total)
  print('Overall progress: [{}] {:6.2f}% | {}'.format(
      '#' * filled + '-' * (width - filled), 100 * done / total, label), flush=True)


def run_logged(command, log_path):
  with log_path.open('w', encoding='utf-8') as handle:
    handle.write('Command: {}\n\n'.format(' '.join(map(str, command))))
    process = subprocess.Popen(command, cwd=str(SCRIPT_DIR), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    for line in process.stdout:
      handle.write(line)
      print(line, end='', flush=True)
  if process.wait():
    raise subprocess.CalledProcessError(process.returncode, command)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  parser.add_argument('--split-dir', type=Path, default=SCRIPT_DIR / 'voyage_split_full' / 'split')
  parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR / 'voyage_split_full' / 'runs')
  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--evaluation-batch-size', type=int, default=256)
  parser.add_argument('--evaluation-seed', type=int, default=369)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  args = parser.parse_args()

  train_list = args.split_dir / 'train_files.txt'
  test_list = args.split_dir / 'test_files.txt'
  total = len(CONFIGURATIONS) * len(SEEDS)
  completed = 0
  for name, crest_weight, psd_weight in CONFIGURATIONS:
    for seed in SEEDS:
      run_dir = args.output_dir / name / 'seed_{}'.format(seed)
      checkpoint = run_dir / 'checkpoints' / 'wavegan_epoch_{:04d}.pt'.format(args.epochs)
      metrics_path = run_dir / 'metrics.json'
      if checkpoint.is_file() and metrics_path.is_file():
        completed += 1
        progress(completed, total, 'skip {} seed {}'.format(name, seed))
        continue
      run_dir.mkdir(parents=True, exist_ok=True)
      train_command = [
          sys.executable, str(SCRIPT_DIR / 'train_wavegan_torch.py'),
          '--data-dir', str(args.data_dir), '--file-list', str(train_list),
          '--output-dir', str(run_dir / 'checkpoints'), '--log-file', str(run_dir / 'epochs.jsonl'),
          '--epochs', str(args.epochs), '--batch-size', str(args.batch_size), '--device', args.device,
          '--crest-loss-weight', '{:g}'.format(crest_weight), '--psd-loss-weight', '{:g}'.format(psd_weight),
          '--seed', str(seed),
      ]
      run_logged(train_command, run_dir / 'training.log')
      evaluation_command = [
          sys.executable, str(SCRIPT_DIR / 'evaluate_wavegan_torch.py'),
          '--checkpoint', str(checkpoint), '--data-dir', str(args.data_dir), '--file-list', str(test_list),
          '--output-dir', str(run_dir / 'images'), '--metrics-output', str(metrics_path),
          '--batch-size', str(args.evaluation_batch_size), '--device', args.device,
          '--evaluation-seed', str(args.evaluation_seed),
      ]
      run_logged(evaluation_command, run_dir / 'evaluation.log')
      completed += 1
      progress(completed, total, 'complete {} seed {}'.format(name, seed))

  with (args.output_dir.parent / 'experiment_config.json').open('w', encoding='utf-8') as handle:
    json.dump({'configurations': CONFIGURATIONS, 'seeds': SEEDS, 'split_dir': str(args.split_dir.resolve())},
              handle, indent=2, ensure_ascii=False)
  subprocess.run([sys.executable, str(SCRIPT_DIR / 'summarize_voyage_split_experiment.py')], check=True)


if __name__ == '__main__':
  main()
