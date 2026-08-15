<#
.SYNOPSIS
    PowerShell equivalent of the Makefile, for Windows machines without `make`.

.EXAMPLE
    .\tasks.ps1 install
    .\tasks.ps1 test
    .\tasks.ps1 check
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'venv', 'install', 'install-all', 'install-evallab',
                 'env-check', 'lint', 'format', 'typecheck', 'test', 'test-all',
                 'smoke', 'check', 'docker-build', 'docker-test', 'docker-shell',
                 'clean')]
    [string]$Task = 'help',

    # Path to the sibling Evallab checkout (spec section 2.3).
    [string]$EvallabPath = '../evallab'
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

function Invoke-Py { & $Python @args; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }

function New-Venv {
    if (-not (Test-Path $Python)) {
        Write-Host 'Creating .venv ...' -ForegroundColor Cyan
        py -3 -m venv .venv
    }
    Invoke-Py -m pip install --upgrade pip
}

function Install-Evallab {
    if (Test-Path $EvallabPath) {
        Write-Host "Installing Evallab (editable) from $EvallabPath" -ForegroundColor Cyan
        Invoke-Py -m pip install -e $EvallabPath
    }
    else {
        Write-Warning "Evallab not found at $EvallabPath. Evallab-dependent code will be unavailable."
        Write-Warning "Clone it next to this repo or pass -EvallabPath <path>."
    }
}

switch ($Task) {
    'help' {
@'
EvidenceRoute tasks:

  install          venv + core package + dev tools (one-command setup)
  install-all      full experiment environment (retrieval, generation, analysis)
  install-evallab  install the sibling Evallab checkout as an editable dependency
  env-check        verify imports, config and Evallab availability
  lint             ruff checks
  format           ruff format + autofix
  typecheck        mypy
  test             default suite (no paid API calls)
  test-all         every test, including live-model tests (incurs cost)
  smoke            offline end-to-end smoke experiment
  check            lint + typecheck + test
  docker-build     build the container image
  docker-test      run the default suite inside the container
  docker-shell     interactive shell in the container
  clean            remove caches and build artifacts
'@ | Write-Host
    }
    'venv'            { New-Venv }
    'install'         { New-Venv; Install-Evallab; Invoke-Py -m pip install -e '.[dev]' }
    'install-all'     { New-Venv; Install-Evallab; Invoke-Py -m pip install -e '.[all,dev]' }
    'install-evallab' { Install-Evallab }
    'env-check'       { Invoke-Py -m evidence_route.cli env check }
    'lint'            { Invoke-Py -m ruff check src tests scripts }
    'format'          { Invoke-Py -m ruff format src tests scripts; Invoke-Py -m ruff check --fix src tests scripts }
    'typecheck'       { Invoke-Py -m mypy }
    'test'            { Invoke-Py -m pytest -m 'not llm and not slow' }
    'test-all'        { Invoke-Py -m pytest }
    'smoke'           { Invoke-Py -m pytest -m smoke }
    'check'           { Invoke-Py -m ruff check src tests scripts; Invoke-Py -m mypy; Invoke-Py -m pytest -m 'not llm and not slow' }
    'docker-build'    { docker compose build }
    'docker-test'     { docker compose run --rm app python -m pytest -m 'not llm and not slow' }
    'docker-shell'    { docker compose run --rm app bash }
    'clean' {
        foreach ($p in '.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov', 'build', 'dist') {
            if (Test-Path $p) { Remove-Item -Recurse -Force $p }
        }
        Get-ChildItem -Recurse -Directory -Filter '__pycache__' |
            Where-Object { $_.FullName -notmatch '\\\.venv\\' } |
            Remove-Item -Recurse -Force
    }
}
