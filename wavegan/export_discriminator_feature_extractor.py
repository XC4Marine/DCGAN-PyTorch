"""Export the final WaveGAN discriminator as a portable feature extractor."""

import json
from pathlib import Path
import shutil

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints' / 'ori_file_10_full'
OUTPUT_DIR = SCRIPT_DIR / 'discriminator_feature_extractor'


def main():
  checkpoint_path = max(
      CHECKPOINT_DIR.glob('wavegan_step_*.pt'),
      key=lambda path: int(path.stem.rsplit('_', 1)[1]))
  checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
  config = checkpoint['args']
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  torch.save(checkpoint['discriminator'], OUTPUT_DIR / 'discriminator_state_dict.pt')
  with (OUTPUT_DIR / 'config.json').open('w', encoding='utf-8') as handle:
    json.dump({
        'checkpoint': checkpoint_path.name,
        'global_step': checkpoint['global_step'],
        'channels': 1,
        'model_dim': config['model_dim'],
        'kernel_len': config['kernel_len'],
        'phase_shuffle_radius': 2,
        'sample_rate': config['sample_rate'],
        'waveform_length': 128,
        'embedding_dim': config['model_dim'] * 2 * 8,
    }, handle, indent=2)
  shutil.copy2(SCRIPT_DIR / 'pytorch_wavegan.py', OUTPUT_DIR / 'pytorch_wavegan.py')
  print('Exported discriminator from {} to {}.'.format(
      checkpoint_path.name, OUTPUT_DIR.resolve()))


if __name__ == '__main__':
  main()
