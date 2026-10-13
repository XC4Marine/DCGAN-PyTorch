"""Train WaveGAN with a real-batch mean Hilbert-envelope MSE loss."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
from scipy.io import wavfile
import torch
from torch import autograd
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from pytorch_wavegan import WaveGANDiscriminator, WaveGANGenerator, weights_init
from hilbert_loss import hilbert_envelope_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def decode_waveform(path, expected_sample_rate):
  """Read, mono-mix, normalize, and right-pad one waveform to 128 samples."""
  sample_rate, waveform = wavfile.read(str(path))
  if sample_rate != expected_sample_rate:
    raise ValueError('{} has {} Hz; expected {} Hz.'.format(path, sample_rate, expected_sample_rate))
  if waveform.ndim == 2:
    waveform = waveform.mean(axis=1)
  if waveform.ndim != 1:
    raise ValueError('{} has unsupported waveform shape {}.'.format(path, waveform.shape))

  if np.issubdtype(waveform.dtype, np.integer):
    waveform = waveform.astype(np.float32) / float(abs(np.iinfo(waveform.dtype).min))
  elif np.issubdtype(waveform.dtype, np.floating):
    waveform = waveform.astype(np.float32, copy=True)
  else:
    raise ValueError('{} has unsupported sample type {}.'.format(path, waveform.dtype))

  maximum = np.max(np.abs(waveform))
  if maximum > 0:
    waveform /= maximum
  if waveform.size > 128:
    raise ValueError('{} has {} samples; this 128-point model requires shorter inputs.'.format(path, waveform.size))
  return np.pad(waveform, (0, 128 - waveform.size)).astype(np.float32, copy=False)


class ClickWaveformDataset(Dataset):
  """One normalized and zero-padded 128-point example per WAV file."""

  def __init__(self, data_dir, sample_rate, file_list=None):
    self.paths = (sorted(path for path in Path(data_dir).rglob('*') if path.suffix.lower() == '.wav')
                  if file_list is None else
                  [Path(line) for line in file_list.read_text(encoding='utf-8').splitlines()])
    if not self.paths:
      raise FileNotFoundError('No WAV files found under {}.'.format(data_dir))
    self.sample_rate = sample_rate

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, index):
    waveform = decode_waveform(self.paths[index], self.sample_rate)
    return torch.from_numpy(waveform).unsqueeze(0)


def gradient_penalty(discriminator, real_waveforms, fake_waveforms):
  batch_size = real_waveforms.size(0)
  interpolation = torch.rand(batch_size, 1, 1, device=real_waveforms.device)
  mixed = interpolation * real_waveforms + (1.0 - interpolation) * fake_waveforms
  mixed.requires_grad_(True)
  scores = discriminator(mixed)
  gradients = autograd.grad(
      outputs=scores,
      inputs=mixed,
      grad_outputs=torch.ones_like(scores),
      create_graph=True,
      retain_graph=True,
      only_inputs=True)[0]
  slopes = gradients.reshape(batch_size, -1).norm(2, dim=1)
  return ((slopes - 1.0) ** 2).mean()


def save_preview(generator, fixed_noise, output_dir, epoch, sample_rate):
  generator.eval()
  with torch.no_grad():
    waveform = generator(fixed_noise[:1])[0, 0].detach().cpu().numpy()
  generator.train()
  encoded = np.clip(waveform * 32767.0, -32767.0, 32767.0).astype(np.int16)
  wavfile.write(str(output_dir / 'preview_epoch_{:04d}.wav'.format(epoch)), sample_rate, encoded)


def save_checkpoint(path, epoch, generator, discriminator, generator_optimizer, discriminator_optimizer,
                    args, global_step, fixed_noise):
  torch.save({
      'epoch': epoch,
      'generator': generator.state_dict(),
      'discriminator': discriminator.state_dict(),
      'generator_optimizer': generator_optimizer.state_dict(),
      'discriminator_optimizer': discriminator_optimizer.state_dict(),
      'args': vars(args),
      'global_step': global_step,
      'fixed_noise': fixed_noise,
      'torch_rng_state': torch.get_rng_state(),
      'cuda_rng_states': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
  }, str(path))


def plot_training_losses(log_file, output_path):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  records = [json.loads(line) for line in log_file.read_text(encoding='utf-8').splitlines()]
  records = [record for record in records if record.get('type') != 'configuration']
  figure, axes = plt.subplots(1, 2, figsize=(11, 4))
  for key, label in (('mean_discriminator_loss', 'D'),
                     ('mean_generator_adversarial_loss', 'G adversarial'),
                     ('mean_generator_total_loss', 'G total')):
    axes[0].plot([record['epoch'] for record in records], [record[key] for record in records], label=label)
  for key, label in (('mean_generator_hilbert_loss', 'Envelope MSE'),
                     ('mean_generator_weighted_hilbert_loss', 'Weighted envelope MSE')):
    axes[1].plot([record['epoch'] for record in records], [record[key] for record in records], label=label)
  for axis in axes:
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Loss')
    axis.grid(alpha=0.25)
    axis.legend()
  figure.tight_layout()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path, dpi=180)
  plt.close(figure)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data' / 'wav')
  parser.add_argument('--file-list', type=Path, default=PROJECT_ROOT / 'wavegan_PSD_crest' /
                      'voyage_split_full' / 'split' / 'train_files.txt')
  parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR / 'runs' / 'single' / 'checkpoints')
  parser.add_argument('--log-file', type=Path)
  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--latent-dim', type=int, default=100)
  parser.add_argument('--model-dim', type=int, default=64)
  parser.add_argument('--kernel-len', type=int, default=25)
  parser.add_argument('--sample-rate', type=int, default=576000)
  parser.add_argument('--n-critic', type=int, default=5)
  parser.add_argument('--gradient-penalty', type=float, default=10.0)
  parser.add_argument('--learning-rate', type=float, default=1e-4)
  parser.add_argument('--hilbert-loss-weight', type=float, default=1.0)
  parser.add_argument('--num-workers', type=int, default=0)
  parser.add_argument('--seed', type=int, default=369)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  parser.add_argument('--resume', type=Path)
  parser.add_argument('--checkpoint-every', type=int, default=50)
  args = parser.parse_args()
  args.data_dir = args.data_dir.resolve()
  args.file_list = args.file_list.resolve()
  args.output_dir = args.output_dir.resolve()
  args.log_file = (args.log_file or args.output_dir.parent / 'epochs.jsonl').resolve()

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

  if args.device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested but PyTorch cannot access a CUDA device.')
  if args.device == 'auto':
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
  else:
    device_name = args.device
  device = torch.device(device_name)
  if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print('Using CUDA device: {}'.format(torch.cuda.get_device_name(device)))
  else:
    print('Using CPU. Install a CUDA-enabled PyTorch build to train on the RTX 4060.')

  dataset = ClickWaveformDataset(args.data_dir, args.sample_rate, args.file_list)
  loader = DataLoader(
      dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
      num_workers=args.num_workers, pin_memory=device.type == 'cuda')
  if len(loader) == 0:
    raise ValueError('Batch size {} exceeds dataset size {}.'.format(args.batch_size, len(dataset)))
  print('Loaded {} click waveforms from {}.'.format(len(dataset), args.data_dir.resolve()))

  generator = WaveGANGenerator(args.latent_dim, dim=args.model_dim, kernel_len=args.kernel_len).to(device)
  discriminator = WaveGANDiscriminator(dim=args.model_dim, kernel_len=args.kernel_len).to(device)
  generator.apply(weights_init)
  discriminator.apply(weights_init)
  generator_optimizer = Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
  discriminator_optimizer = Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
  start_epoch = 1
  global_step = 0
  fixed_noise = torch.randn(16, args.latent_dim, device=device)

  if args.resume:
    checkpoint = torch.load(str(args.resume), map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
    discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    global_step = checkpoint['global_step']
    fixed_noise = checkpoint['fixed_noise'].to(device)
    torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
    if device.type == 'cuda':
      torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng_states']])
    print('Resumed from {} at epoch {}.'.format(args.resume, start_epoch))

  args.output_dir.mkdir(parents=True, exist_ok=True)
  args.log_file.parent.mkdir(parents=True, exist_ok=True)
  if args.resume and args.log_file.is_file():
    records = [json.loads(line) for line in args.log_file.read_text(encoding='utf-8').splitlines()]
    records = [record for record in records if record.get('epoch', 0) < start_epoch]
    args.log_file.write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records),
                             encoding='utf-8')
  else:
    args.log_file.write_text(json.dumps({'type': 'configuration', **vars(args)}, default=str,
                                        ensure_ascii=False) + '\n', encoding='utf-8')
  for epoch in range(start_epoch, args.epochs + 1):
    generator.train()
    discriminator.train()
    generator_loss = None
    discriminator_losses, adversarial_losses, hilbert_losses, total_losses = [], [], [], []
    for batch_index, real_waveforms in enumerate(loader, 1):
      real_waveforms = real_waveforms.to(device, non_blocking=True)
      batch_size = real_waveforms.size(0)

      noise = torch.randn(batch_size, args.latent_dim, device=device)
      fake_waveforms = generator(noise).detach()
      discriminator_optimizer.zero_grad()
      discriminator_loss = (
          discriminator(fake_waveforms).mean() - discriminator(real_waveforms).mean()
          + args.gradient_penalty * gradient_penalty(discriminator, real_waveforms, fake_waveforms))
      discriminator_loss.backward()
      discriminator_optimizer.step()
      discriminator_losses.append(discriminator_loss.item())

      if global_step % args.n_critic == 0:
        generator_optimizer.zero_grad()
        generated_waveforms = generator(torch.randn(batch_size, args.latent_dim, device=device))
        adversarial_loss = -discriminator(generated_waveforms).mean()
        hilbert_loss = hilbert_envelope_loss(real_waveforms, generated_waveforms)
        generator_loss = adversarial_loss + args.hilbert_loss_weight * hilbert_loss
        generator_loss.backward()
        generator_optimizer.step()
        adversarial_losses.append(adversarial_loss.item())
        hilbert_losses.append(hilbert_loss.item())
        total_losses.append(generator_loss.item())

      global_step += 1
      if batch_index % 50 == 0 or batch_index == len(loader):
        generator_value = float('nan') if generator_loss is None else generator_loss.item()
        fraction = ((epoch - 1) * len(loader) + batch_index) / (args.epochs * len(loader))
        filled = round(30 * fraction)
        hilbert_value = float('nan') if not hilbert_losses else hilbert_losses[-1]
        print(
            'Training progress: [{}] {:6.2f}% | Epoch {}/{} | batch {}/{} | D={:.5f} | G={:.5f} | H={:.6f}'.format(
                '#' * filled + '-' * (30 - filled), fraction * 100, epoch, args.epochs,
                batch_index, len(loader), discriminator_loss.item(), generator_value, hilbert_value),
            flush=True)

    record = {
        'epoch': epoch,
        'mean_discriminator_loss': float(np.mean(discriminator_losses)),
        'mean_generator_adversarial_loss': float(np.mean(adversarial_losses)),
        'mean_generator_hilbert_loss': float(np.mean(hilbert_losses)),
        'mean_generator_weighted_hilbert_loss': args.hilbert_loss_weight * float(np.mean(hilbert_losses)),
        'mean_generator_total_loss': float(np.mean(total_losses)),
        'generator_update_count': len(total_losses),
        'global_step': global_step,
    }
    with args.log_file.open('a', encoding='utf-8') as handle:
      handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
      save_checkpoint(
          args.output_dir / 'wavegan_epoch_{:04d}.pt'.format(epoch), epoch,
          generator, discriminator, generator_optimizer, discriminator_optimizer, args, global_step, fixed_noise)
      save_preview(generator, fixed_noise, args.output_dir, epoch, args.sample_rate)
  plot_training_losses(args.log_file, args.output_dir.parent / 'images' / 'training_losses.png')


if __name__ == '__main__':
  main()
