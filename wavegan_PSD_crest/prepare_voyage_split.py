"""Create the fixed voyage-disjoint 80/20 click split."""

import argparse
import json
from pathlib import Path
import re


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'data' / 'wav'
LOCATE_DIR = Path(r'D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate')
TEST_VOYAGES = ('Voyage_Ori_02', 'Voyage_Ori_08', 'Voyage_Ori_15')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
  parser.add_argument('--locate-dir', type=Path, default=LOCATE_DIR)
  parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR / 'voyage_split_full' / 'split')
  args = parser.parse_args()

  voyage_by_train = {}
  for voyage_dir in sorted(args.locate_dir.glob('Voyage_Ori_*')):
    for train_dir in voyage_dir.glob('PulseTrain_*'):
      voyage_by_train[train_dir.name] = voyage_dir.name

  selected_paths = sorted(args.data_dir.rglob('*.wav'))
  pattern = re.compile(r'^(PulseTrain_\d+)_Pulse_\d+$')
  train_paths, test_paths = [], []
  voyages = set()
  for path in selected_paths:
    voyage = voyage_by_train[pattern.match(path.stem).group(1)]
    voyages.add(voyage)
    if voyage in TEST_VOYAGES:
      test_paths.append(path.resolve())
    else:
      train_paths.append(path.resolve())

  args.output_dir.mkdir(parents=True, exist_ok=True)
  (args.output_dir / 'train_files.txt').write_text(
      '\n'.join(map(str, train_paths)) + '\n', encoding='utf-8')
  (args.output_dir / 'test_files.txt').write_text(
      '\n'.join(map(str, test_paths)) + '\n', encoding='utf-8')
  metadata = {
      'data_dir': str(args.data_dir.resolve()),
      'selected_click_count': len(selected_paths),
      'test_voyages': TEST_VOYAGES,
      'train_voyages': sorted(voyages - set(TEST_VOYAGES)),
      'train_click_count': len(train_paths),
      'test_click_count': len(test_paths),
      'test_fraction': len(test_paths) / (len(train_paths) + len(test_paths)),
  }
  with (args.output_dir / 'split.json').open('w', encoding='utf-8') as handle:
    json.dump(metadata, handle, indent=2, ensure_ascii=False)
  print('Created full voyage-disjoint split: {} train, {} test.'.format(
      len(train_paths), len(test_paths)))


if __name__ == '__main__':
  main()
