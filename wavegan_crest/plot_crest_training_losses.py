"""Plot discriminator, generator, and crest losses from one Crest WaveGAN run."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_epoch_records(log_path):
  """Load epoch summaries, ignoring the first configuration JSONL record."""
  records = []
  with log_path.open(encoding='utf-8') as handle:
    for line in handle:
      record = json.loads(line)
      if record.get('type') != 'configuration':
        records.append(record)
  if not records:
    raise ValueError('No epoch summaries found in {}.'.format(log_path))
  return records


def plot_records(records, output_path, title):
  """Render one three-panel loss figure from epoch-level JSONL records."""
  epochs = [record['epoch'] for record in records]
  series = [
      ('Discriminator loss', 'mean_discriminator_loss', '#1769aa'),
      ('Generator total loss', 'mean_generator_total_loss', '#e76f51'),
      ('Crest loss (unweighted)', 'mean_generator_crest_loss', '#2a9d8f'),
  ]
  figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
  for axis, (label, key, color) in zip(axes, series):
    axis.plot(epochs, [record[key] for record in records], color=color, linewidth=1.4)
    axis.set_title(label)
    axis.set_ylabel('Loss')
    axis.grid(alpha=0.25)
  axes[-1].set_xlabel('Epoch')
  figure.suptitle(title, fontsize=15)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--log-file', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--title', default='WaveGAN Crest Loss Training Curves')
  args = parser.parse_args()

  records = load_epoch_records(args.log_file)
  plot_records(records, args.output, args.title)
  print('Saved {} epoch loss records to {}.'.format(len(records), args.output.resolve()))


if __name__ == '__main__':
  main()
