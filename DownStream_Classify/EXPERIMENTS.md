# Downstream classification experiments

## Retained experiments

### Balanced augmentation comparison

- Entry point: `run_balanced_600_experiment.py`
- Output: `results_balanced_600_gpu/`
- Report: `results_balanced_600_gpu/EXPERIMENT_REPORT.md`
- Design: 600 real positive samples, matched unique negatives, and 150 to 1,197 generated positives from Raw WaveGAN or PSD+Crest WaveGAN.
- Formal evidence: 17 conditions, 5 classifier seeds, 85 CUDA runs.

### Out-of-domain positive detection

- Entry point: `run_ood_detection_experiment.py`
- Output: `results_ood_initial_screen_gpu/`
- Checkpoints: `results_ood_initial_screen_gpu/checkpoints/`
- Test data: 153 positive WAV files from the external initial-screen dataset.
- Primary metric: detection rate at a fixed threshold of 0.5.

## Shared code and data

- `prepare_experiment.py`: prepares manifests and generated waveform pools.
- `transformer_classifier.py`: preprocessing, Transformer classifier, training, evaluation, OOD inference, and checkpoint saving.
- `manifests/`: fixed training and test file lists.
- `data/`: prepared real and generated waveform files.

## Output layout

- `results_balanced_600_gpu/logs/`: archived logs for the retained balanced augmentation experiment.
- `results_ood_initial_screen_gpu.stdout.log`: live OOD experiment output.
- `results_ood_initial_screen_gpu.stderr.log`: live OOD error output.

Smoke-test outputs and experiments that did not support the retained augmentation claim are not part of the formal result package.
