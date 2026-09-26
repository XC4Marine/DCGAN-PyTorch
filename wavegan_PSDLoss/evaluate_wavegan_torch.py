"""Run the shared WaveGAN evaluation so PSD-loss results use identical metrics."""

from pathlib import Path
import runpy
import sys


SOURCE_DIR = Path(__file__).resolve().parents[1] / 'wavegan_improve'
sys.path.insert(0, str(SOURCE_DIR))
runpy.run_path(str(SOURCE_DIR / 'evaluate_wavegan_torch.py'), run_name='__main__')
