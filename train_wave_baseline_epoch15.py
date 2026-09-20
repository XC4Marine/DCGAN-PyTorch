"""Select a waveform DCGAN by held-out physical-feature validation metrics.

No reconstruction, feature-matching, or other auxiliary loss is used here: each
candidate uses only its adversarial objective (LSGAN or BCE-with-logits GAN).
"""

from __future__ import annotations

import csv
from pathlib import Path
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dcgan_wave import Discriminator1D, Generator1D, weights_init
from utils_wave import WavSegmentDataset, discover_wav_paths


seed = 369
params = {
    "bsize": 256,
    "channels": 1,
    "signal_length": 128,
    "nz": 100,
    "nepochs": 15,
    "beta1": 0.5,
    "eval_every": 3,
    # Splitting files, not segments, prevents one recording leaking across sets.
    "train_file_ratio": 0.80,
    "validation_file_ratio": 0.10,
    "data_root": "data/wav",
    "run_name": "baseline_epoch15_validation_selection",
}

# All candidates use only the adversarial loss.  They differ in the requested
# hyperparameters, and are compared solely by the fixed validation set.
CANDIDATES = (
    {
        "name": "lsgan_weak_discriminator",
        "adversarial_loss": "lsgan",
        "ngf": 64,
        "ndf": 32,
        "lr_d": 1e-5,
        "lr_g": 5e-5,
    },
    {
        "name": "lsgan_balanced_learning_rate",
        "adversarial_loss": "lsgan",
        "ngf": 64,
        "ndf": 32,
        "lr_d": 5e-5,
        "lr_g": 5e-5,
    },
    {
        "name": "bce_balanced_learning_rate",
        "adversarial_loss": "bce_logits",
        "ngf": 64,
        "ndf": 32,
        "lr_d": 5e-5,
        "lr_g": 5e-5,
    },
)


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def split_wav_paths(paths: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """Make reproducible train/validation/test partitions at recording level."""
    if len(paths) < 3:
        raise ValueError("At least three WAV files are required for train/validation/test splitting.")

    shuffled_paths = list(paths)
    random.Random(seed).shuffle(shuffled_paths)
    train_end = int(len(shuffled_paths) * params["train_file_ratio"])
    validation_end = train_end + int(len(shuffled_paths) * params["validation_file_ratio"])
    train_paths = shuffled_paths[:train_end]
    validation_paths = shuffled_paths[train_end:validation_end]
    test_paths = shuffled_paths[validation_end:]
    if not train_paths or not validation_paths or not test_paths:
        raise ValueError("The configured file split produced an empty partition.")
    return train_paths, validation_paths, test_paths


def dataset_to_numpy(dataset: WavSegmentDataset) -> np.ndarray:
    """Materialize a held-out dataset once; it is never passed to either network."""
    return torch.stack([dataset[index] for index in range(len(dataset))]).numpy()


def waveform_features(waveforms: np.ndarray) -> dict[str, np.ndarray]:
    """Per-waveform physical descriptors used only for model selection."""
    signals = waveforms[:, 0, :].astype(np.float64, copy=False)
    absolute = np.abs(signals)
    peak_amplitude = absolute.max(axis=1)
    threshold = 0.5 * peak_amplitude[:, None]
    # Width is the duration above half of each waveform's own peak amplitude.
    pulse_width = (absolute >= threshold).sum(axis=1).astype(np.float64)
    return {
        "sample_amplitude": signals.ravel(),
        "peak_amplitude": peak_amplitude,
        "peak_to_peak": np.ptp(signals, axis=1),
        "rms": np.sqrt(np.mean(np.square(signals), axis=1)),
        "pulse_width_half_peak": pulse_width,
    }


def feature_distances(
    real_features: dict[str, np.ndarray], generated_features: dict[str, np.ndarray]
) -> dict[str, float]:
    """Standardize Wasserstein distances so no high-scale feature dominates."""
    distances = {}
    for name, real_values in real_features.items():
        raw_distance = wasserstein_distance(real_values, generated_features[name])
        scale = max(float(np.std(real_values)), 1e-6)
        distances[name] = float(raw_distance / scale)
    distances["selection_score"] = float(np.mean(list(distances.values())))
    return distances


def generate_evaluation_dataset(
    generator: Generator1D, latent_vectors: torch.Tensor, batch_size: int, device: torch.device
) -> np.ndarray:
    """Generate exactly one held-out-set-sized, reproducible fake dataset."""
    batches = []
    generator.eval()
    with torch.no_grad():
        for start in range(0, len(latent_vectors), batch_size):
            noise = latent_vectors[start:start + batch_size].to(device)
            batches.append(generator(noise).cpu().numpy())
    generator.train()
    return np.concatenate(batches, axis=0)


def show_training_progress(
    candidate_name: str,
    candidate_index: int,
    candidate_count: int,
    epoch: int,
    total_epochs: int,
    batch: int,
    total_batches: int,
) -> None:
    completed = candidate_index * total_epochs * total_batches + (epoch - 1) * total_batches + batch
    total = candidate_count * total_epochs * total_batches
    width = 30
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\rOverall training [{bar}] {completed}/{total} batches "
        f"(candidate {candidate_index + 1}/{candidate_count}: {candidate_name})",
        end="",
        flush=True,
    )


def adversarial_criterion(name: str) -> nn.Module:
    if name == "lsgan":
        return nn.MSELoss()
    if name == "bce_logits":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown adversarial loss: {name}")


def train_candidate(
    candidate: dict,
    candidate_index: int,
    train_loader: DataLoader,
    validation_features: dict[str, np.ndarray],
    validation_noise: torch.Tensor,
    device: torch.device,
    run_dir: Path,
    results: list[dict[str, object]],
) -> tuple[Path, float, list[float], list[float]]:
    """Train one configuration and return its best validation checkpoint."""
    set_seed(seed)  # Common initialization and training RNG for fair comparison.
    net_g = Generator1D(params["nz"], params["channels"], candidate["ngf"]).to(device)
    net_d = Discriminator1D(params["channels"], candidate["ndf"]).to(device)
    net_g.apply(weights_init)
    net_d.apply(weights_init)
    criterion = adversarial_criterion(candidate["adversarial_loss"])
    optimizer_d = optim.Adam(net_d.parameters(), lr=candidate["lr_d"], betas=(params["beta1"], 0.999))
    optimizer_g = optim.Adam(net_g.parameters(), lr=candidate["lr_g"], betas=(params["beta1"], 0.999))
    candidate_dir = run_dir / candidate["name"]
    candidate_dir.mkdir(parents=True, exist_ok=True)
    best_path = candidate_dir / "best_validation_model.pth"
    best_score = float("inf")
    d_losses, g_losses = [], []

    print(f"\nTraining {candidate['name']} using only {candidate['adversarial_loss']} adversarial loss.")
    for epoch in range(1, params["nepochs"] + 1):
        for batch_index, real_waveforms in enumerate(train_loader, start=1):
            real_waveforms = real_waveforms.to(device)
            batch_size = real_waveforms.size(0)
            real_labels = torch.ones(batch_size, dtype=torch.float32, device=device)
            fake_labels = torch.zeros(batch_size, dtype=torch.float32, device=device)

            optimizer_d.zero_grad(set_to_none=True)
            loss_d_real = criterion(net_d(real_waveforms), real_labels)
            noise = torch.randn(batch_size, params["nz"], device=device)
            fake_waveforms = net_g(noise)
            loss_d_fake = criterion(net_d(fake_waveforms.detach()), fake_labels)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            loss_g = criterion(net_d(fake_waveforms), real_labels)
            loss_g.backward()
            optimizer_g.step()
            d_losses.append(float(loss_d.item()))
            g_losses.append(float(loss_g.item()))
            show_training_progress(
                candidate["name"],
                candidate_index,
                len(CANDIDATES),
                epoch,
                params["nepochs"],
                batch_index,
                len(train_loader),
            )
        print()

        if epoch % params["eval_every"] == 0 or epoch == params["nepochs"]:
            generated = generate_evaluation_dataset(net_g, validation_noise, params["bsize"], device)
            distances = feature_distances(validation_features, waveform_features(generated))
            row = {"candidate": candidate["name"], "epoch": epoch, **candidate, **distances}
            results.append(row)
            print(
                f"  validation epoch {epoch:02d}: selection_score={distances['selection_score']:.6f}; "
                + ", ".join(f"{name}={value:.4f}" for name, value in distances.items() if name != "selection_score")
            )
            checkpoint = {
                "generator": net_g.state_dict(),
                "discriminator": net_d.state_dict(),
                "params": params,
                "candidate": candidate,
                "epoch": epoch,
                "validation_distances": distances,
            }
            torch.save(checkpoint, candidate_dir / f"model_epoch_{epoch}.pth")
            if distances["selection_score"] < best_score:
                best_score = distances["selection_score"]
                torch.save(checkpoint, best_path)

    return best_path, best_score, d_losses, g_losses


def save_selection_results(results: list[dict[str, object]], output_path: Path) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    if params["signal_length"] != 128:
        raise ValueError("Generator1D and Discriminator1D require signal_length=128.")
    if not 0 < params["train_file_ratio"] < 1 or not 0 < params["validation_file_ratio"] < 1:
        raise ValueError("File split ratios must lie between zero and one.")
    if params["train_file_ratio"] + params["validation_file_ratio"] >= 1:
        raise ValueError("Train and validation file ratios must leave files for test.")

    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wav_paths = discover_wav_paths(params["data_root"])
    train_paths, validation_paths, test_paths = split_wav_paths(wav_paths)
    train_dataset = WavSegmentDataset(segment_length=params["signal_length"], wav_paths=train_paths)
    validation_dataset = WavSegmentDataset(segment_length=params["signal_length"], wav_paths=validation_paths)
    test_dataset = WavSegmentDataset(segment_length=params["signal_length"], wav_paths=test_paths)
    train_loader = DataLoader(train_dataset, batch_size=params["bsize"], shuffle=True, num_workers=0)
    validation_waveforms = dataset_to_numpy(validation_dataset)
    test_waveforms = dataset_to_numpy(test_dataset)
    validation_features = waveform_features(validation_waveforms)
    # Fixed latent vectors make every epoch and candidate evaluation comparable.
    evaluation_generator = torch.Generator().manual_seed(seed + 1)
    validation_noise = torch.randn(len(validation_dataset), params["nz"], generator=evaluation_generator)
    test_noise = torch.randn(len(test_dataset), params["nz"], generator=evaluation_generator)

    run_dir = Path("model") / params["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Using {device}; file-level split: {len(train_paths)} train / {len(validation_paths)} validation / "
        f"{len(test_paths)} test WAV files."
    )
    print(
        f"Segments: {len(train_dataset)} train / {len(validation_dataset)} validation / "
        f"{len(test_dataset)} test.  Validation and test data are never used for backpropagation."
    )

    results: list[dict[str, object]] = []
    trained_candidates = []
    for candidate_index, candidate in enumerate(CANDIDATES):
        best_path, best_score, d_losses, g_losses = train_candidate(
            candidate,
            candidate_index,
            train_loader,
            validation_features,
            validation_noise,
            device,
            run_dir,
            results,
        )
        trained_candidates.append((best_score, candidate, best_path, d_losses, g_losses))
    save_selection_results(results, run_dir / "validation_selection.csv")

    best_score, best_candidate, best_path, d_losses, g_losses = min(trained_candidates, key=lambda item: item[0])
    selected_path = run_dir / "best_model_by_validation.pth"
    shutil.copy2(best_path, selected_path)
    selected = torch.load(selected_path, map_location=device, weights_only=False)
    selected_generator = Generator1D(params["nz"], params["channels"], best_candidate["ngf"]).to(device)
    selected_generator.load_state_dict(selected["generator"])
    generated_test_waveforms = generate_evaluation_dataset(selected_generator, test_noise, params["bsize"], device)
    test_distances = feature_distances(waveform_features(test_waveforms), waveform_features(generated_test_waveforms))
    with (run_dir / "final_test_metrics.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["candidate", "selected_epoch", *test_distances.keys()])
        writer.writeheader()
        writer.writerow({"candidate": best_candidate["name"], "selected_epoch": selected["epoch"], **test_distances})

    print(f"\nSelected {best_candidate['name']} at epoch {selected['epoch']} (validation={best_score:.6f}).")
    print(
        "Final untouched test score="
        f"{test_distances['selection_score']:.6f}; "
        + ", ".join(f"{name}={value:.4f}" for name, value in test_distances.items() if name != "selection_score")
    )
    print(f"Saved selected checkpoint to {selected_path.resolve()}")

    plt.figure(figsize=(9, 4))
    plt.plot(d_losses, label="D")
    plt.plot(g_losses, label="G")
    plt.xlabel("Iteration")
    plt.ylabel("Adversarial loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "selected_candidate_losses.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
