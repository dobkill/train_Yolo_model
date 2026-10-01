param([int]$Port = 8766, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$webPython = 'D:\APP\miniconda\envs\yolo\python.exe'
$webArguments = @((Join-Path $PSScriptRoot 'web/launch_web.py'), '--port', "$Port")
if ($NoBrowser) { $webArguments += '--no-browser' }
& $webPython @webArguments
exit $LASTEXITCODE
