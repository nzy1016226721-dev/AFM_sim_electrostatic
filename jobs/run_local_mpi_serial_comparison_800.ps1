[CmdletBinding()]
param(
    [string]$EnvironmentRoot = "D:\afm_pack_v1\.afm_mpi_local_env",
    [string]$OutputRoot = "outputs/local_mpi_comparison_800"
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $EnvironmentRoot "Scripts/python.exe"
$MpiExec = Join-Path $EnvironmentRoot "Library/bin/mpiexec.exe"
$SerialConfig = Join-Path $PackageRoot "afm_config_nm_local_comparison_serial_800.json"
$MpiConfig = Join-Path $PackageRoot "afm_config_nm_local_comparison_mpi_800.json"
$ResolvedOutput = Join-Path $PackageRoot $OutputRoot

foreach ($Required in @($Python, $MpiExec, $SerialConfig, $MpiConfig)) {
    if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
        throw "Required comparison input is missing: $Required"
    }
}

New-Item -ItemType Directory -Path $ResolvedOutput -Force | Out-Null
$env:PYTHONUNBUFFERED = "1"

& $Python (Join-Path $PackageRoot "run_all.py") $SerialConfig `
    --output-dir (Join-Path $ResolvedOutput "serial") --no-plot
if ($LASTEXITCODE -ne 0) {
    throw "The 800^3 serial reference run failed with exit code $LASTEXITCODE."
}

& $MpiExec -n 8 $Python (Join-Path $PackageRoot "run_mpi.py") $MpiConfig `
    --output-dir (Join-Path $ResolvedOutput "mpi") --no-plot
if ($LASTEXITCODE -ne 0) {
    throw "The 800^3 MPI candidate run failed with exit code $LASTEXITCODE."
}

$SerialNpy = Join-Path $ResolvedOutput `
    "serial/afm_config_nm_local_comparison_serial_800/afm_phi_1_0nm_-1.00V.npy"
$MpiNpy = Join-Path $ResolvedOutput `
    "mpi/afm_config_nm_local_comparison_mpi_800/afm_phi_1_0nm_-1.00V.npy"
$Report = Join-Path $ResolvedOutput "comparison_800.json"

& $Python (Join-Path $PackageRoot "postprocessing/compare_mpi_npy.py") `
    $SerialNpy $MpiNpy --atol 0 --rtol 0 --chunk-planes 1 --report $Report
if ($LASTEXITCODE -ne 0) {
    throw "The 800^3 fields were not bitwise identical; inspect $Report."
}

Write-Host "SUCCESS: serial and MPI 800^3 fields are bitwise identical."
Write-Host "Comparison report: $Report"
