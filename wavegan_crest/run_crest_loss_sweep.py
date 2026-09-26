"""Run CrestFactorLoss weights 0.1 through 1.0 and save logs, checkpoints, and plots."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_WEIGHTS = tuple(round(index / 10, 1) for index in range(1, 11))


def weight_name(weight):
  return '{:.1f}'.format(weight)


def progress_bar(completed, total, label):
  """Print the sweep-level progress bar requested for long-running experiments."""
  width = 30
  filled = round(width * completed / total)
  bar = '#' * filled + '-' * (width - filled)
  print('Overall progress: [{}] {:6.2f}% | {}'.format(
      bar, 100.0 * completed / total, label), flush=True)


def run_training(command, log_path, run_index, run_count, epochs, weight):
  """Run training, save its transcript, and report sweep-level progress."""
  epoch_pattern = re.compile(r'Epoch (\d+)/(\d+) \| batch (\d+)/(\d+)')
  last_reported_epoch = 0
  with log_path.open('w', encoding='utf-8') as log_handle:
    log_handle.write('Command: {}\n\n'.format(' '.join(map(str, command))))
    process = subprocess.Popen(
        command, cwd=str(SCRIPT_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace')
    assert process.stdout is not None
    for line in process.stdout:
      log_handle.write(line)
      match = epoch_pattern.search(line)
      if match is None:
        continue
      epoch, total_epochs, batch, total_batches = map(int, match.groups())
      if batch == total_batches and (epoch == 1 or epoch == total_epochs or epoch - last_reported_epoch >= 10):
        last_reported_epoch = epoch
        completed = run_index * epochs + epoch
        progress_bar(completed, run_count * epochs, 'crest weight {} epoch {}/{}'.format(
            weight_name(weight), epoch, total_epochs))
    return_code = process.wait()
  if return_code:
    raise subprocess.CalledProcessError(return_code, command)


def load_epoch_records(path):
  """Return epoch summaries from a JSONL log without its configuration record."""
  records = []
  with path.open(encoding='utf-8') as handle:
    for line in handle:
      record = json.loads(line)
      if record.get('type') != 'configuration':
        records.append(record)
  if not records:
    raise ValueError('No epoch summaries found in {}.'.format(path))
  return records


def plot_sweep_summary(results, output_path):
  """Compare D, total G, and unweighted Crest losses across all weights."""
  weights = [result['weight'] for result in results]
  metrics = [result['records'][-1] for result in results]
  panels = [
      ('Final discriminator loss', 'mean_discriminator_loss', '#1769aa'),
      ('Final generator total loss', 'mean_generator_total_loss', '#e76f51'),
      ('Final Crest loss (unweighted)', 'mean_generator_crest_loss', '#2a9d8f'),
  ]
  figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
  for axis, (title, key, color) in zip(axes, panels):
    axis.plot(weights, [metric[key] for metric in metrics], marker='o', color=color)
    axis.set_title(title)
    axis.set_xlabel('Crest-loss weight')
    axis.set_ylabel('Loss')
    axis.grid(alpha=0.25)
  figure.suptitle('WaveGAN Crest Loss Weight Sweep', fontsize=15)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  parser.add_argument('--seed', type=int, default=369)
  parser.add_argument('--checkpoint-every', type=int, default=200,
                      help='Default retains only the final checkpoint from each weight.')
  parser.add_argument('--weights', type=float, nargs='+', default=DEFAULT_WEIGHTS)
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  args = parser.parse_args()

  if args.epochs < 1:
    raise ValueError('--epochs must be positive.')
  if not args.weights or any(weight <= 0 for weight in args.weights):
    raise ValueError('--weights must contain positive values; 0.0 is intentionally excluded.')

  log_dir = SCRIPT_DIR / 'log'
  image_dir = SCRIPT_DIR / 'images'
  checkpoint_dir = SCRIPT_DIR / 'checkpoints'
  for directory in (log_dir, image_dir, checkpoint_dir):
    directory.mkdir(parents=True, exist_ok=True)

  results = []
  total_runs = len(args.weights)
  progress_bar(0, total_runs * args.epochs, 'starting crest-loss weight sweep')
  for index, weight in enumerate(args.weights):
    name = weight_name(weight)
    run_dir = checkpoint_dir / 'alpha_{}'.format(name)
    run_image_dir = image_dir / 'alpha_{}'.format(name)
    train_log = log_dir / 'alpha_{}_training.log'.format(name)
    epoch_log = log_dir / 'alpha_{}_epochs.jsonl'.format(name)
    loss_figure = run_image_dir / 'training_losses.png'
    print('\nStarting crest-loss weight={} ({}/{})'.format(name, index + 1, total_runs), flush=True)

    train_command = [
        sys.executable, str(SCRIPT_DIR / 'train_wavegan_torch.py'),
        '--data-dir', str(args.data_dir), '--output-dir', str(run_dir),
        '--epochs', str(args.epochs), '--batch-size', str(args.batch_size),
        '--device', args.device, '--seed', str(args.seed),
        '--checkpoint-every', str(args.checkpoint_every),
        '--crest-loss-weight', name, '--log-file', str(epoch_log),
    ]
    run_training(train_command, train_log, index, total_runs, args.epochs, weight)

    plot_command = [
        sys.executable, str(SCRIPT_DIR / 'plot_crest_training_losses.py'),
        '--log-file', str(epoch_log), '--output', str(loss_figure),
        '--title', 'WaveGAN Crest Loss: weight {}'.format(name),
    ]
    subprocess.run(plot_command, cwd=str(SCRIPT_DIR), check=True)
    records = load_epoch_records(epoch_log)
    results.append({'weight': weight, 'records': records})
    progress_bar((index + 1) * args.epochs, total_runs * args.epochs,
                 'crest weight {} complete'.format(name))

  summary_path = image_dir / 'crest_loss_weight_sweep_summary.png'
  plot_sweep_summary(results, summary_path)
  summary_json = log_dir / 'crest_loss_weight_sweep_summary.json'
  with summary_json.open('w', encoding='utf-8') as handle:
    json.dump(results, handle, indent=2, ensure_ascii=False)
  evaluation_command = [
      sys.executable, str(SCRIPT_DIR / 'evaluate_crest_loss_sweep.py'),
      '--checkpoint-root', str(checkpoint_dir), '--checkpoint-prefix', 'alpha',
      '--checkpoint-epoch', str(args.epochs), '--weights',
      *[weight_name(weight) for weight in args.weights], '--data-dir', str(args.data_dir),
      '--device', args.device,
  ]
  subprocess.run(evaluation_command, cwd=str(SCRIPT_DIR), check=True)
  print('\nCompleted all {} Crest Loss weights.'.format(total_runs))
  print('Logs: {}'.format(log_dir.resolve()))
  print('Images: {}'.format(image_dir.resolve()))
  print('Checkpoints: {}'.format(checkpoint_dir.resolve()))


if __name__ == '__main__':
  main()
