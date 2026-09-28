"""Train WaveGAN-GP with CrestFactorLoss and GlobalSpectralLoss."""

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
from scipy.io import wavfile
import torch
from torch import autograd
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'wavegan_crest'))
from crest_loss import CrestFactorLoss
from pytorch_wavegan import WaveGANDiscriminator, WaveGANGenerator, weights_init
sys.path.insert(0, str(PROJECT_ROOT / 'wavegan_PSDLoss'))
from global_spectral_loss import GlobalSpectralLoss


def decode_waveform(path, expected_sample_rate):
  sample_rate, waveform = wavfile.read(str(path))
  if sample_rate != expected_sample_rate:
    raise ValueError('{} has {} Hz; expected {} Hz.'.format(path, sample_rate, expected_sample_rate))
  if waveform.ndim == 2:
    waveform = waveform.mean(axis=1)
  if np.issubdtype(waveform.dtype, np.integer):
    waveform = waveform.astype(np.float32) / float(abs(np.iinfo(waveform.dtype).min))
  else:
    waveform = waveform.astype(np.float32, copy=True)
  maximum = np.max(np.abs(waveform))
  if maximum > 0:
    waveform /= maximum
  if waveform.size > 128:
    raise ValueError('{} has {} samples; this model requires at most 128.'.format(path, waveform.size))
  return np.pad(waveform, (0, 128 - waveform.size)).astype(np.float32, copy=False)


class ClickWaveformDataset(Dataset):
  def __init__(self, data_dir, sample_rate, file_list=None):
    self.paths = (sorted(Path(data_dir).rglob('*.wav')) if file_list is None else
                  [Path(line) for line in file_list.read_text(encoding='utf-8').splitlines()])
    self.sample_rate = sample_rate

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, index):
    return torch.from_numpy(decode_waveform(self.paths[index], self.sample_rate)).unsqueeze(0)


def gradient_penalty(discriminator, real_waveforms, fake_waveforms):
  batch_size = real_waveforms.size(0)
  interpolation = torch.rand(batch_size, 1, 1, device=real_waveforms.device)
  mixed = interpolation * real_waveforms + (1.0 - interpolation) * fake_waveforms
  mixed.requires_grad_(True)
  scores = discriminator(mixed)
  gradients = autograd.grad(
      outputs=scores, inputs=mixed, grad_outputs=torch.ones_like(scores),
      create_graph=True, retain_graph=True, only_inputs=True)[0]
  slopes = gradients.reshape(batch_size, -1).norm(2, dim=1)
  return ((slopes - 1.0) ** 2).mean()


def save_checkpoint(path, epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, args):
  torch.save({
      'epoch': epoch,
      'generator': generator.state_dict(),
      'discriminator': discriminator.state_dict(),
      'generator_optimizer': generator_optimizer.state_dict(),
      'discriminator_optimizer': discriminator_optimizer.state_dict(),
      'args': vars(args),
  }, str(path))


def save_preview(generator, fixed_noise, path, sample_rate):
  generator.eval()
  with torch.no_grad():
    waveform = generator(fixed_noise[:1])[0, 0].cpu().numpy()
  generator.train()
  wavfile.write(str(path), sample_rate, np.clip(waveform * 32767, -32767, 32767).astype(np.int16))


def write_epoch_log(log_file, epoch, args, discriminator_losses, adversarial_losses,
                    crest_losses, psd_losses, total_losses):
  def average(values):
    return float(np.mean(values)) if values else None
  record = {
      'epoch': epoch,
      'mean_discriminator_loss': average(discriminator_losses),
      'mean_generator_adversarial_loss': average(adversarial_losses),
      'mean_generator_crest_loss': average(crest_losses),
      'mean_generator_psd_loss': average(psd_losses),
      'mean_generator_weighted_crest_loss': average(
          [args.crest_loss_weight * value for value in crest_losses]),
      'mean_generator_weighted_psd_loss': average(
          [args.psd_loss_weight * value for value in psd_losses]),
      'mean_generator_total_loss': average(total_losses),
      'generator_update_count': len(total_losses),
  }
  with log_file.open('a', encoding='utf-8') as handle:
    handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def plot_training_losses(log_file, output_path):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  records = [json.loads(line) for line in log_file.read_text(encoding='utf-8').splitlines()]
  records = [record for record in records if record.get('type') != 'configuration']
  figure, axis = plt.subplots(figsize=(9, 5))
  for key, label in (
      ('mean_discriminator_loss', 'D'),
      ('mean_generator_adversarial_loss', 'G adversarial'),
      ('mean_generator_crest_loss', 'Crest'),
      ('mean_generator_psd_loss', 'PSD'),
      ('mean_generator_total_loss', 'G total')):
    axis.plot([record['epoch'] for record in records], [record[key] for record in records], label=label)
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
  parser.add_argument('--file-list', type=Path, default=None)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--log-file', type=Path, required=True)
  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--latent-dim', type=int, default=100)
  parser.add_argument('--model-dim', type=int, default=64)
  parser.add_argument('--kernel-len', type=int, default=25)
  parser.add_argument('--sample-rate', type=int, default=576000)
  parser.add_argument('--n-critic', type=int, default=5)
  parser.add_argument('--gradient-penalty', type=float, default=10.0)
  parser.add_argument('--learning-rate', type=float, default=1e-4)
  parser.add_argument('--crest-loss-weight', type=float, required=True)
  parser.add_argument('--psd-loss-weight', type=float, required=True)
  parser.add_argument('--psd-n-fft', type=int, default=128)
  parser.add_argument('--psd-hop-length', type=int, default=32)
  parser.add_argument('--seed', type=int, default=369)
  parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
  args = parser.parse_args()

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
  device_name = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device
  device = torch.device(device_name)
  if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

  dataset = ClickWaveformDataset(args.data_dir, args.sample_rate, args.file_list)
  loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                      pin_memory=device.type == 'cuda')
  generator = WaveGANGenerator(args.latent_dim, dim=args.model_dim, kernel_len=args.kernel_len).to(device)
  discriminator = WaveGANDiscriminator(dim=args.model_dim, kernel_len=args.kernel_len).to(device)
  generator.apply(weights_init)
  discriminator.apply(weights_init)
  crest_loss_function = CrestFactorLoss().to(device)
  psd_loss_function = GlobalSpectralLoss(args.psd_n_fft, args.psd_hop_length).to(device)
  generator_optimizer = Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
  discriminator_optimizer = Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

  args.output_dir.mkdir(parents=True, exist_ok=True)
  args.log_file.parent.mkdir(parents=True, exist_ok=True)
  with args.log_file.open('w', encoding='utf-8') as handle:
    handle.write(json.dumps({'type': 'configuration', **vars(args)}, default=str, ensure_ascii=False) + '\n')
  fixed_noise = torch.randn(16, args.latent_dim, device=device)
  global_step = 0
  for epoch in range(1, args.epochs + 1):
    discriminator_losses, adversarial_losses, crest_losses, psd_losses, total_losses = [], [], [], [], []
    generator_loss = None
    for batch_index, real_waveforms in enumerate(loader, 1):
      real_waveforms = real_waveforms.to(device, non_blocking=True)
      batch_size = real_waveforms.size(0)
      fake_waveforms = generator(torch.randn(batch_size, args.latent_dim, device=device)).detach()
      discriminator_optimizer.zero_grad()
      discriminator_loss = (discriminator(fake_waveforms).mean() - discriminator(real_waveforms).mean() +
                            args.gradient_penalty * gradient_penalty(discriminator, real_waveforms, fake_waveforms))
      discriminator_loss.backward()
      discriminator_optimizer.step()
      discriminator_losses.append(discriminator_loss.item())

      if global_step % args.n_critic == 0:
        generator_optimizer.zero_grad()
        generated_waveforms = generator(torch.randn(batch_size, args.latent_dim, device=device))
        adversarial_loss = -discriminator(generated_waveforms).mean()
        crest_loss = crest_loss_function(real_waveforms, generated_waveforms)
        psd_loss = psd_loss_function(real_waveforms, generated_waveforms)
        generator_loss = (adversarial_loss + args.crest_loss_weight * crest_loss +
                          args.psd_loss_weight * psd_loss)
        generator_loss.backward()
        generator_optimizer.step()
        adversarial_losses.append(adversarial_loss.item())
        crest_losses.append(crest_loss.item())
        psd_losses.append(psd_loss.item())
        total_losses.append(generator_loss.item())
      global_step += 1
      if batch_index % 50 == 0 or batch_index == len(loader):
        generator_value = float('nan') if generator_loss is None else generator_loss.item()
        print('Epoch {}/{} | batch {}/{} | D={:.5f} | G={:.5f}'.format(
            epoch, args.epochs, batch_index, len(loader), discriminator_loss.item(), generator_value), flush=True)

    write_epoch_log(args.log_file, epoch, args, discriminator_losses, adversarial_losses,
                    crest_losses, psd_losses, total_losses)
  checkpoint = args.output_dir / 'wavegan_epoch_{:04d}.pt'.format(args.epochs)
  save_checkpoint(checkpoint, args.epochs, generator, discriminator, generator_optimizer,
                  discriminator_optimizer, args)
  save_preview(generator, fixed_noise, args.output_dir / 'preview_epoch_{:04d}.wav'.format(args.epochs),
               args.sample_rate)
  plot_training_losses(args.log_file, args.output_dir.parent / 'images' / 'training_losses.png')


if __name__ == '__main__':
  main()
