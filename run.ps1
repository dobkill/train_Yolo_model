param(
    [int]$Count = 2000,
    [string]$Output = 'dataset',
    [switch]$SkipExtract
)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$pipelineScript = Join-Path $PSScriptRoot 'code\run_pipeline.py'
$pipelineArguments = @($pipelineScript, '--count', "$Count", '--output', $Output)
if ($SkipExtract) { $pipelineArguments += '--skip-extract' }
& python @pipelineArguments
if ($LASTEXITCODE -ne 0) { throw "Pipeline failed with exit code $LASTEXITCODE" }
