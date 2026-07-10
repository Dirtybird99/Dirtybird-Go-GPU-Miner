param([string]$Compiler = $env:SLANGC)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($Compiler)) {
    $command = Get-Command slangc -ErrorAction SilentlyContinue
    if ($command) {
        $Compiler = $command.Source
    }
}
if ([string]::IsNullOrWhiteSpace($Compiler)) {
    $candidate = Get-ChildItem (Join-Path $env:TEMP 'slang-*\bin\slangc.exe') -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($candidate) {
        $Compiler = $candidate.FullName
    }
}
if ([string]::IsNullOrWhiteSpace($Compiler) -or -not (Test-Path -LiteralPath $Compiler)) {
    throw 'slangc not found; put it on PATH, set SLANGC, or pass -Compiler.'
}

$source = Join-Path $PSScriptRoot 'src'

function Compile-Shader {
    param(
        [string]$File,
        [string]$Entry,
        [string]$Output,
        [string[]]$Define = @()
    )

    $arguments = @(
        (Join-Path $source $File),
        '-entry', $Entry,
        '-stage', 'compute',
        '-target', 'spirv',
        '-profile', 'glsl_460',
        '-O3'
    )
    foreach ($item in $Define) {
        $arguments += "-D$item"
    }
    $arguments += @('-o', (Join-Path $PSScriptRoot $Output))
    & $Compiler @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "slangc failed for $Output"
    }
}

Compile-Shader sort.slang init init_0.spv RADIX_SHIFT=0
Compile-Shader sort.slang scan scan_0.spv RADIX_SHIFT=0
foreach ($shift in 0, 8, 16, 24) {
    Compile-Shader sort.slang histogram "histogram_$shift.spv" "RADIX_SHIFT=$shift"
    Compile-Shader sort.slang scatter "scatter_$shift.spv" "RADIX_SHIFT=$shift"
}

Compile-Shader refine.slang DetectDirect detectdirect.spv
Compile-Shader refine.slang PrepareDirect preparedirect.spv
Compile-Shader refine.slang SortTiny direct_tiny_0.spv BIN=0, MAXLEN=2
foreach ($bin in 1, 2, 3) {
    Compile-Shader refine.slang SortTinyGlobal "global_$bin.spv" "BIN=$bin"
}
Compile-Shader refine.slang SortLarge direct_large_4.spv BIN=4, MAXLEN=128
Compile-Shader refine.slang SortLarge direct_large_5.spv BIN=5, MAXLEN=512
Compile-Shader refine.slang SortLarge direct_large_6.spv BIN=6, MAXLEN=2048
Compile-Shader final_sha.slang main final_sha.spv

Get-ChildItem (Join-Path $PSScriptRoot '*.spv') | Sort-Object Name | Select-Object Name, Length
