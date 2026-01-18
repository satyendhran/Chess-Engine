[CmdletBinding()]
param (
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]]$Files
)

$BlackLineLength = 88
$RuffLineLength  = 88
$IsortProfile    = "black"

function Die {
    param([string]$Message)
    Write-Error $Message
    exit 1
}

function Require-Tool {
    param([string]$Tool)
    if (-not (Get-Command $Tool -ErrorAction SilentlyContinue)) {
        Die "Required tool '$Tool' is not installed or not in PATH"
    }
}

Require-Tool isort
Require-Tool black
Require-Tool ruff

foreach ($file in $Files) {
    if (-not (Test-Path $file)) {
        Die "File not found: $file"
    }
}

Write-Host "Formatting files:"
$Files | ForEach-Object { Write-Host "  $_" }
Write-Host ""

Write-Host "isort"
isort `
    --profile $IsortProfile `
    --line-length $BlackLineLength `
    $Files
if ($LASTEXITCODE -ne 0) { Die "isort failed" }

Write-Host "black"
black `
    --line-length $BlackLineLength `
    $Files
if ($LASTEXITCODE -ne 0) { Die "black failed" }

Write-Host "ruff format"
ruff format `
    --line-length $RuffLineLength `
    $Files
if ($LASTEXITCODE -ne 0) { Die "ruff format failed" }

Write-Host "ruff check"
ruff check `
    $Files --fix
if ($LASTEXITCODE -ne 0) { Die "ruff check failed" }

Write-Host ""
Write-Host "Formatting and linting completed successfully"
