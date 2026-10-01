param([string]$Model = '', [string]$Resume = '')
$ErrorActionPreference = 'Stop'
$trainPython = 'D:\APP\miniconda\envs\yolo\python.exe'
$trainArguments = @((Join-Path $PSScriptRoot 'launch.py'))
if ($Model) { $trainArguments += @('--model', $Model) }
if ($Resume) { $trainArguments += @('--resume', $Resume) }
& $trainPython @trainArguments
exit $LASTEXITCODE
