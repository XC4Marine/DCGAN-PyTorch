# WaveGAN Batch PSD Loss

This experiment matches the real and generated batches by their average log
power spectral density (PSD), without pairing individual real and generated
waveforms.  The WaveGAN architecture and evaluation metrics are shared with
`wavegan_improve` so that the results are directly comparable.

For a quick smoke test:

```powershell
python .\train_wavegan_torch.py --epochs 1 --batch-size 8 --device cpu --psd-loss-weight 0.1 --output-dir .\smoke_run
```

Run the full weight sweep (0.1 through 1.0):

```powershell
python .\run_psd_loss_sweep.py --device cuda
```

The training objective is:

```text
L_G = L_WGAN + lambda_psd * mean(abs(log(mean(PSD_fake)) - log(mean(PSD_real))))
```

`n_fft=128` is the waveform length; `hop_length=32` is configurable but has no
effect for the default one-frame, `center=False` transform.  Power is computed
as `abs(STFT)**2`, not magnitude.
