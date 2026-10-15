"""Export the Ori_file_num=10 real WAVs and samples from the last five checkpoints."""

import argparse
from pathlib import Path
import shutil

import numpy as np
from scipy.io import wavfile
import torch

from pytorch_wavegan import WaveGANGenerator
from train_wavegan_torch import ClickWaveformDataset, load_train_ids


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def progress(completed, total, label):
  filled = round(30 * completed / total)
  print('Overall progress: [{}] {:6.2f}% | {}'.format(
      '#' * filled + '-' * (30 - filled), 100 * completed / total, label), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint-dir', type=Path, default=SCRIPT_DIR / 'checkpoints' /
                      'ori_file_10_alpha_0.2_full')
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  parser.add_argument('--metadata-file', type=Path, default=PROJECT_ROOT / 'data' / 'ClickTrains.csv')
  parser.add_argument('--output-dir', type=Path,
                      default=Path('D:/Project_Github/VGGish_Evaluation/Crestdata'))
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  args = parser.parse_args()

  checkpoints = sorted(
      args.checkpoint_dir.glob('wavegan_epoch_*.pt'),
      key=lambda path: int(path.stem.rsplit('_', 1)[1]))[-5:]
  if len(checkpoints) != 5:
    raise ValueError('Expected at least five epoch checkpoints in {}.'.format(args.checkpoint_dir))

  train_ids = load_train_ids(args.metadata_file, 10)
  source_paths = ClickWaveformDataset(args.data_dir, 576000, train_ids).paths
  count = len(source_paths)
  total = count * 6
  completed = 0
  original_dir = args.output_dir / 'original'
  original_dir.mkdir(parents=True, exist_ok=True)
  progress(completed, total, 'copy original WAVs')
  for source in source_paths:
    shutil.copy2(source, original_dir / source.name)
    completed += 1
    if completed % 256 == 0 or completed == count:
      progress(completed, total, 'copy original WAVs')

  device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                        else 'cpu' if args.device == 'auto' else args.device)
  for checkpoint_path in checkpoints:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['args']
    generator = WaveGANGenerator(
        config['latent_dim'], dim=config['model_dim'], kernel_len=config['kernel_len']).to(device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    random_source = torch.Generator(device=device).manual_seed(config['seed'])
    epoch = checkpoint['epoch']
    group_dir = args.output_dir / 'epoch_{:04d}'.format(epoch)
    group_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
      for start in range(0, count, args.batch_size):
        batch_count = min(args.batch_size, count - start)
        noise = torch.randn(batch_count, config['latent_dim'],
                            device=device, generator=random_source)
        waveforms = generator(noise)[:, 0].cpu().numpy()
        for index, waveform in enumerate(waveforms, start + 1):
          encoded = np.clip(waveform * 32767.0, -32767.0, 32767.0).astype(np.int16)
          wavfile.write(str(group_dir / 'generated_{:05d}.wav'.format(index)),
                        config['sample_rate'], encoded)
        completed += batch_count
        progress(completed, total, 'export epoch {}'.format(epoch))

  print('Exported {} WAVs per group to {}.'.format(count, args.output_dir.resolve()), flush=True)


if __name__ == '__main__':
  main()
