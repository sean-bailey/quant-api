# PowerShell script to quantize a model with llama.cpp in all available quantizations
# Clean version - same as quantize-all.ps1 but outputs are placed next to the base model
# rather than in a "quantized" subfolder.
# Automatically clones llama.cpp if not present, or pulls latest if it is.
# Usage: .\quantize-all-clean.ps1 -ModelFolder "path\to\model\folder" [options]

param(
    [Parameter(Mandatory=$true)]
    [string]$ModelFolder,

    [Parameter(Mandatory=$false)]
    [string]$BuildDir = "build\bin\Release",

    [Parameter(Mandatory=$false)]
    [string]$IMatrix = "",

    [Parameter(Mandatory=$false)]
    [int]$Threads = [Environment]::ProcessorCount,

    [Parameter(Mandatory=$false)]
    [switch]$SkipBuild,

    [Parameter(Mandatory=$false)]
    [switch]$ForceRebuild,

    [Parameter(Mandatory=$false)]
    [string]$LlamaCppUrl = "https://github.com/ggml-org/llama.cpp.git"
)

$ErrorActionPreference = "Stop"
$ScriptRoot = $PSScriptRoot
if (-not $ScriptRoot) { $ScriptRoot = Get-Location }

$LlamaCppDir = Join-Path $ScriptRoot "llama.cpp"

# All available quantization types from the source code (tools/quantize/quantize.cpp)
$QuantTypes = @(
    "Q4_0",      # 4.34G, +0.4685 ppl @ Llama-3-8B
    "Q4_1",      # 4.78G, +0.4511 ppl @ Llama-3-8B
    "Q5_0",      # 5.21G, +0.1316 ppl @ Llama-3-8B
    "Q5_1",      # 5.65G, +0.1062 ppl @ Llama-3-8B
    "IQ2_XXS",   # 2.06 bpw quantization
    "IQ2_XS",    # 2.31 bpw quantization
    "IQ2_S",     # 2.5  bpw quantization
    "IQ2_M",     # 2.7  bpw quantization
    "IQ1_S",     # 1.56 bpw quantization
    "IQ1_M",     # 1.75 bpw quantization
    "TQ1_0",     # 1.69 bpw ternarization
    "TQ2_0",     # 2.06 bpw ternarization
    "Q2_K",      # 2.96G, +3.5199 ppl @ Llama-3-8B
    "Q2_K_S",    # 2.96G, +3.1836 ppl @ Llama-3-8B
    "IQ3_XXS",   # 3.06 bpw quantization
    "IQ3_S",     # 3.44 bpw quantization
    "IQ3_M",     # 3.66 bpw quantization mix
    "Q3_K_S",    # 3.41G, +1.6321 ppl @ Llama-3-8B
    "Q3_K_M",    # 3.74G, +0.6569 ppl @ Llama-3-8B
    "Q3_K_L",    # 4.03G, +0.5562 ppl @ Llama-3-8B
    "IQ3_XS",    # 3.3 bpw quantization
    "IQ4_NL",    # 4.50 bpw non-linear quantization
    "IQ4_XS",    # 4.25 bpw non-linear quantization
    "Q4_K_S",    # 4.37G, +0.2689 ppl @ Llama-3-8B
    "Q4_K_M",    # 4.58G, +0.1754 ppl @ Llama-3-8B
    "Q5_K_S",    # 5.21G, +0.1049 ppl @ Llama-3-8B
    "Q5_K_M",    # 5.33G, +0.0569 ppl @ Llama-3-8B
    "Q6_K",      # 6.14G, +0.0217 ppl @ Llama-3-8B
    "Q8_0",      # 7.96G, +0.0026 ppl @ Llama-3-8B
    "F16",       # 14.00G, +0.0020 ppl @ Mistral-7B
    "BF16"       # 14.00G, -0.0050 ppl @ Mistral-7B
)

$RequiresIMatrix = @("IQ1_S", "IQ1_M", "IQ2_S", "IQ2_XXS", "IQ2_XS", "Q2_K_S")

function Write-ColoredOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Test-PathExists {
    param([string]$Path, [string]$Type)
    if (!(Test-Path $Path)) {
        Write-ColoredOutput "Error: $Type not found at: $Path" "Red"
        exit 1
    }
}

function Find-BaseModel {
    param([string]$ModelFolder)

    $BasePatterns = @("*f16*.gguf", "*bf16*.gguf", "*f32*.gguf", "*unquantized*.gguf")
    foreach ($pattern in $BasePatterns) {
        $found = Get-ChildItem -Path $ModelFolder -Filter $pattern | Select-Object -First 1
        if ($found) { return $found.FullName }
    }

    $AllGgufs = Get-ChildItem -Path $ModelFolder -Filter "*.gguf"
    foreach ($gguf in $AllGgufs) {
        $name = $gguf.Name.ToLower()
        $hasQuantSuffix = $false
        foreach ($quant in $QuantTypes) {
            if ($name -match $quant.ToLower()) { $hasQuantSuffix = $true; break }
        }
        if (!$hasQuantSuffix) { return $gguf.FullName }
    }

    $FirstGguf = Get-ChildItem -Path $ModelFolder -Filter "*.gguf" | Select-Object -First 1
    if ($FirstGguf) {
        Write-ColoredOutput "Warning: Could not identify base model, using: $($FirstGguf.Name)" "Yellow"
        return $FirstGguf.FullName
    }
    return $null
}

# Clone or update llama.cpp
if (-not $SkipBuild) {
    if (-not (Test-Path $LlamaCppDir)) {
        Write-ColoredOutput "llama.cpp not found, cloning from $LlamaCppUrl ..." "Yellow"
        git clone $LlamaCppUrl $LlamaCppDir
        if ($LASTEXITCODE -ne 0) { Write-ColoredOutput "Failed to clone llama.cpp!" "Red"; exit 1 }
        Write-ColoredOutput "llama.cpp cloned!" "Green"
        $OldCommit = ""; $NewCommit = "new"
    } else {
        $OldCommit = git -C $LlamaCppDir rev-parse HEAD 2>$null
        Write-ColoredOutput "Pulling latest llama.cpp..." "Yellow"
        git -C $LlamaCppDir pull 2>$null
        $NewCommit = git -C $LlamaCppDir rev-parse HEAD 2>$null
    }

    if ($ForceRebuild) {
        $BuildPath = Join-Path $LlamaCppDir "build"
        if (Test-Path $BuildPath) { Remove-Item -Recurse -Force $BuildPath }
    }

    $QuantizeBin = Join-Path (Join-Path $LlamaCppDir $BuildDir) "llama-quantize.exe"
    if ($ForceRebuild -or $OldCommit -ne $NewCommit -or -not (Test-Path $QuantizeBin)) {
        Write-ColoredOutput "Building llama.cpp..." "Yellow"
        Push-Location $LlamaCppDir
        try {
            cmake -B build
            if ($LASTEXITCODE -ne 0) { Write-ColoredOutput "CMake failed!" "Red"; exit 1 }
            cmake --build build --config Release -j $Threads
            if ($LASTEXITCODE -ne 0) { Write-ColoredOutput "Build failed!" "Red"; exit 1 }
        } finally { Pop-Location }
        Write-ColoredOutput "Build complete!" "Green"
    } else {
        Write-ColoredOutput "llama.cpp up to date, skipping rebuild." "Green"
    }
}

Write-ColoredOutput "Starting llama.cpp quantization for all formats..." "Green"

Test-PathExists $ModelFolder "Model folder"
$QuantizePath = Join-Path (Join-Path $LlamaCppDir $BuildDir) "llama-quantize.exe"
Test-PathExists $QuantizePath "llama-quantize.exe"

if ($IMatrix -ne "" -and !(Test-Path $IMatrix)) {
    Write-ColoredOutput "Warning: Importance matrix file not found at: $IMatrix" "Yellow"
    $IMatrix = ""
}

$BaseModel = Find-BaseModel $ModelFolder
if (!$BaseModel) { Write-ColoredOutput "Error: No GGUF files found in $ModelFolder" "Red"; exit 1 }

Write-ColoredOutput "Base model: $(Split-Path $BaseModel -Leaf)" "Cyan"
Write-ColoredOutput "Build directory: $BuildDir" "Cyan"
Write-ColoredOutput "Threads: $Threads" "Cyan"
if ($IMatrix -ne "") {
    Write-ColoredOutput "Importance matrix: $(Split-Path $IMatrix -Leaf)" "Cyan"
} else {
    Write-ColoredOutput "No importance matrix provided (required for some quantizations)" "Yellow"
}

# Output goes alongside the base model (clean - no subfolder)
$OutputFolder = $ModelFolder

$SuccessCount = 0
$SkippedCount = 0
$FailedCount = 0
$TotalCount = $QuantTypes.Count

Write-ColoredOutput "`nStarting quantization process..." "Green"
Write-ColoredOutput "Total quantization types to process: $TotalCount`n" "Green"

foreach ($QuantType in $QuantTypes) {
    $BaseModelName = [System.IO.Path]::GetFileNameWithoutExtension($BaseModel)
    $OutputModel = Join-Path $OutputFolder "$BaseModelName-$QuantType.gguf"

    Write-ColoredOutput "Processing $QuantType ($($QuantTypes.IndexOf($QuantType) + 1)/$TotalCount)..." "Yellow"

    if (Test-Path $OutputModel) {
        Write-ColoredOutput "  Already exists: $(Split-Path $OutputModel -Leaf)" "Gray"
        $SkippedCount++
        continue
    }

    if ($RequiresIMatrix -contains $QuantType -and $IMatrix -eq "") {
        Write-ColoredOutput "  Skipped: $QuantType requires importance matrix" "Yellow"
        $SkippedCount++
        continue
    }

    $Args = @("`"$BaseModel`"", "`"$OutputModel`"", $QuantType, $Threads)
    if ($IMatrix -ne "") { $Args = @("--imatrix", "`"$IMatrix`"") + $Args }

    try {
        $StartTime = Get-Date
        $Process = Start-Process -FilePath $QuantizePath -ArgumentList $Args -Wait -PassThru -NoNewWindow
        $EndTime = Get-Date
        $Duration = ($EndTime - $StartTime).TotalSeconds

        if ($Process.ExitCode -eq 0) {
            $SizeMB = [math]::Round((Get-Item $OutputModel).Length / 1MB, 2)
            Write-ColoredOutput "  Success: $QuantType (${SizeMB}MB, ${Duration}s)" "Green"
            $SuccessCount++
        } else {
            Write-ColoredOutput "  Failed: $QuantType (exit code: $($Process.ExitCode))" "Red"
            $FailedCount++
        }
    } catch {
        Write-ColoredOutput "  Error: $QuantType - $($_.Exception.Message)" "Red"
        $FailedCount++
    }
}

Write-ColoredOutput "`n=================================================" "Cyan"
Write-ColoredOutput "QUANTIZATION SUMMARY" "Cyan"
Write-ColoredOutput "=================================================" "Cyan"
Write-ColoredOutput "Total quantizations: $TotalCount" "White"
Write-ColoredOutput "Successful: $SuccessCount" "Green"
Write-ColoredOutput "Skipped: $SkippedCount" "Yellow"
Write-ColoredOutput "Failed: $FailedCount" "Red"
Write-ColoredOutput "Output directory: $OutputFolder" "Cyan"

if ($SuccessCount -gt 0) {
    Write-ColoredOutput "`nQuantized models:" "Green"
    Get-ChildItem -Path $OutputFolder -Filter "*.gguf" | Sort-Object Length | ForEach-Object {
        $SizeMB = [math]::Round($_.Length / 1MB, 2)
        Write-ColoredOutput "  $($_.Name) (${SizeMB}MB)" "White"
    }
}

Write-ColoredOutput "`nQuantization complete!" "Green"
