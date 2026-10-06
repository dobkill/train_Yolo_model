param([int]$Port = 8766, [switch]$NoBrowser, [string]$Python = '')
$ErrorActionPreference = 'Stop'
$webPython = $Python
if (-not $webPython) {
    $isolatedPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    if (Test-Path -LiteralPath $isolatedPython) { $webPython = $isolatedPython }
    elseif (Test-Path -LiteralPath 'D:\APP\miniconda\envs\yolo\python.exe') { $webPython = 'D:\APP\miniconda\envs\yolo\python.exe' }
    else { $webPython = (Get-Command python -ErrorAction Stop).Source }
}
if (-not (Test-Path -LiteralPath $webPython)) { throw 'Python executable not found. Use -Python to select an environment.' }
$webArguments = @((Join-Path $PSScriptRoot 'web/launch_web.py'), '--port', "$Port")
if ($NoBrowser) { $webArguments += '--no-browser' }
& $webPython @webArguments
exit $LASTEXITCODE
