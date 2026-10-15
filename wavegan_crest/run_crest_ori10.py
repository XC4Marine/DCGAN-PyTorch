"""Run the full Crest-loss experiment for Ori_file_num=10 and weight=0.2."""

import argparse
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def run_and_tee(command, transcript_path):
  """Run training while mirroring its output to the console and a transcript."""
  with transcript_path.open('w', encoding='utf-8') as log_handle:
    log_handle.write('Command: {}\n\n'.format(' '.join(map(str, command))))
    process = subprocess.Popen(
        command, cwd=str(SCRIPT_DIR), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    assert process.stdout is not None
    for line in process.stdout:
      print(line, end='', flush=True)
      log_handle.write(line)
    return_code = process.wait()
  if return_code:
    raise subprocess.CalledProcessError(return_code, command)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  parser.add_argument('--seed', type=int, default=369)
  parser.add_argument('--num-workers', type=int, default=0)
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  parser.add_argument('--metadata-file', type=Path, default=PROJECT_ROOT / 'data' / 'ClickTrains.csv')
  args = parser.parse_args()

  experiment_name = 'ori_file_10_alpha_0.2_full'
  checkpoint_dir = SCRIPT_DIR / 'checkpoints' / experiment_name
  log_dir = SCRIPT_DIR / 'log' / experiment_name
  image_dir = SCRIPT_DIR / 'images' / experiment_name
  for directory in (checkpoint_dir, log_dir, image_dir):
    directory.mkdir(parents=True, exist_ok=True)

  epoch_log = log_dir / 'epochs.jsonl'
  transcript = log_dir / 'training.log'
  command = [
      sys.executable, str(SCRIPT_DIR / 'train_wavegan_torch.py'),
      '--data-dir', str(args.data_dir),
      '--metadata-file', str(args.metadata_file),
      '--ori-file-num', '10',
      '--output-dir', str(checkpoint_dir),
      '--max-update-steps', '200000',
      '--batch-size', str(args.batch_size),
      '--device', args.device,
      '--seed', str(args.seed),
      '--num-workers', str(args.num_workers),
      '--checkpoint-every', '50',
      '--crest-loss-weight', '0.2',
      '--log-file', str(epoch_log),
  ]
  print('Starting full Crest experiment: Ori_file_num=10, weight=0.2, '
        'max steps=200000, early stopping disabled.', flush=True)
  run_and_tee(command, transcript)

  plot_command = [
      sys.executable, str(SCRIPT_DIR / 'plot_crest_training_losses.py'),
      '--log-file', str(epoch_log),
      '--output', str(image_dir / 'training_losses.png'),
      '--title', 'WaveGAN Crest Loss: Ori file 10, weight 0.2',
  ]
  subprocess.run(plot_command, cwd=str(SCRIPT_DIR), check=True)
  print('Experiment complete.', flush=True)
  print('Checkpoints: {}'.format(checkpoint_dir.resolve()), flush=True)
  print('Logs: {}'.format(log_dir.resolve()), flush=True)
  print('Plot: {}'.format((image_dir / 'training_losses.png').resolve()), flush=True)


if __name__ == '__main__':
  main()
