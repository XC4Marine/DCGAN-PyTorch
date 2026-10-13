"""Run matched Raw WaveGAN and Hilbert-loss experiments at four training batch sizes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def progress(done, total, label):  ##打印进度条
  filled = round(30 * done / total)
  print('Overall progress: [{}] {:6.2f}% | {}'.format(
      '#' * filled + '-' * (30 - filled), 100 * done / total, label), flush=True)


def run_logged(command, log_path):
  with log_path.open('a', encoding='utf-8') as handle:
    handle.write('Command: {}\n'.format(subprocess.list2cmdline(command)))
    handle.flush()
    process = subprocess.Popen(command, cwd=str(SCRIPT_DIR), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    for line in process.stdout:
      handle.write(line)
      handle.flush()
      print(line, end='', flush=True)
  if process.wait():
    raise subprocess.CalledProcessError(process.returncode, command)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  parser.add_argument('--split-dir', type=Path, default=PROJECT_ROOT / 'wavegan_PSD_crest' /
                      'voyage_split_full' / 'split')
  parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR / 'batch_size_experiment')
  parser.add_argument('--batch-sizes', type=int, nargs='+', default=[16, 32, 64, 128])
  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--seed', type=int, default=369)
  parser.add_argument('--hilbert-loss-weight', type=float, default=1.0)
  parser.add_argument('--checkpoint-every', type=int, default=50)
  parser.add_argument('--evaluation-batch-size', type=int, default=256)
  parser.add_argument('--evaluation-seed', type=int, default=369)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='cuda')
  args = parser.parse_args()
  for name in ('data_dir', 'split_dir', 'output_dir'):
    setattr(args, name, getattr(args, name).resolve())
  args.output_dir.mkdir(parents=True, exist_ok=True)
  config = json.loads(json.dumps(vars(args), default=str))
  config_path = args.output_dir / 'experiment_config.json'
  if config_path.is_file() and json.loads(config_path.read_text(encoding='utf-8')) != config:
    raise ValueError('Configuration changed; use a different --output-dir for this experiment.')
  config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
  total = len(args.batch_sizes) * 4
  completed = 0
  progress(completed, total, 'starting training and evaluation')
  for batch_size in args.batch_sizes:
    for method, weight in (('raw_wavegan', 0.0), ('hilbert', args.hilbert_loss_weight)):
      run_dir = args.output_dir / 'runs' / 'batch_{}'.format(batch_size) / method / 'seed_{}'.format(args.seed)
      checkpoint_dir = run_dir / 'checkpoints'
      checkpoint = checkpoint_dir / 'wavegan_epoch_{:04d}.pt'.format(args.epochs)
      metrics_path = run_dir / 'metrics.json'
      run_dir.mkdir(parents=True, exist_ok=True)
      label = 'batch {} | {} | seed {}'.format(batch_size, method, args.seed)
      if not checkpoint.is_file():
        command = [
            sys.executable, '-X', 'utf8', '-u', str(SCRIPT_DIR / 'train_wavegan_torch.py'),
            '--data-dir', str(args.data_dir), '--file-list', str(args.split_dir / 'train_files.txt'),
            '--output-dir', str(checkpoint_dir), '--log-file', str(run_dir / 'epochs.jsonl'),
            '--epochs', str(args.epochs), '--batch-size', str(batch_size), '--seed', str(args.seed),
            '--hilbert-loss-weight', str(weight), '--checkpoint-every', str(args.checkpoint_every),
            '--device', args.device,
        ]
        checkpoints = sorted(checkpoint_dir.glob('wavegan_epoch_*.pt'))
        if checkpoints:
          command.extend(['--resume', str(checkpoints[-1])])
        progress(completed, total, 'train ' + label)
        run_logged(command, run_dir / 'training.log')
      completed += 1
      progress(completed, total, 'checkpoint ready | ' + label)
      if not metrics_path.is_file():
        command = [
            sys.executable, '-X', 'utf8', '-u', str(SCRIPT_DIR / 'evaluate_wavegan_torch.py'),
            '--checkpoint', str(checkpoint), '--data-dir', str(args.data_dir),
            '--file-list', str(args.split_dir / 'test_files.txt'),
            '--output-dir', str(run_dir / 'images'), '--metrics-output', str(metrics_path),
            '--batch-size', str(args.evaluation_batch_size), '--evaluation-seed', str(args.evaluation_seed),
            '--device', args.device,
        ]
        run_logged(command, run_dir / 'evaluation.log')
      completed += 1
      progress(completed, total, 'evaluated | ' + label)
  subprocess.run([
      sys.executable, '-X', 'utf8', str(SCRIPT_DIR / 'summarize_batch_size_experiment.py'),
      '--experiment-dir', str(args.output_dir)], check=True)


if __name__ == '__main__':
  main()
