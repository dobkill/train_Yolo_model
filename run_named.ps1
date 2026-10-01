param(
    [int]$PerClass = 1000,
    [int]$Workers = 16,
    [string]$Output = "dataset_named_1000"
)
$ErrorActionPreference = "Stop"
$entry = Join-Path $PSScriptRoot "code/render_named_dataset.py"
python -X utf8 $entry --per-class $PerClass --workers $Workers --output $Output
if ($LASTEXITCODE -ne 0) { throw "Named dataset generation failed. Completed metadata checkpoints can be resumed." }
python -X utf8 (Join-Path $PSScriptRoot "code/finalize_named_dataset.py") --dataset $Output
if ($LASTEXITCODE -ne 0) { throw "Delivery audit or preview generation failed." }
