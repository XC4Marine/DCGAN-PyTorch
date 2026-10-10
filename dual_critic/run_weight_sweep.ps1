$python = 'D:\Python_env\wavegan_torch\python.exe'
$train = 'D:\Project_Github\DCGAN-PyTorch\dual_critic\train_dual_critic.py'
$evaluate = 'D:\Project_Github\DCGAN-PyTorch\dual_critic\evaluate_dual_critic.py'
$trainList = 'D:\Project_Github\DCGAN-PyTorch\wavegan_PSD_crest\voyage_split_full\split\train_files.txt'
$testList = 'D:\Project_Github\DCGAN-PyTorch\wavegan_PSD_crest\voyage_split_full\split\test_files.txt'

foreach ($weight in 0.75, 0.25) {
  $name = 'wave_{0:00}_stft_{1:00}' -f (100 * $weight), (100 * (1 - $weight))
  $base = "D:\Project_Github\DCGAN-PyTorch\dual_critic\runs\$name\seed_369"
  New-Item -ItemType Directory -Path $base -Force | Out-Null
  & $python $train --file-list $trainList --output-dir "$base\checkpoints" --log-file "$base\epochs.jsonl" --crest-loss-weight 0.6 --psd-loss-weight 0.15 --waveform-critic-weight $weight --epochs 200 --batch-size 64 --seed 369 *>> "$base\train_console.log"
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
  & $python $evaluate --checkpoint "$base\checkpoints\wavegan_epoch_0200.pt" --file-list $testList --output-dir "$base\images" --metrics-output "$base\metrics.json" --evaluation-seed 369 *>> "$base\evaluate_console.log"
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}
