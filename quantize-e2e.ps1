<#
.SYNOPSIS
    End-to-end model quantization script for llama.cpp

.DESCRIPTION
    This script automates the complete workflow of quantizing a HuggingFace model:
    1. Clones llama.cpp if not present, or pulls latest if it is
    2. Rebuilds llama.cpp if needed
    3. Recreates Python venv and installs requirements
    4. Clones the specified HuggingFace model
    5. Detects if model is multimodal/vision and creates mmproj if needed
    6. Converts to GGUF format
    7. Deletes original safetensors (keeping config files)
    8. Creates all quantization variants
    9. Uploads to HuggingFace under your account as <model-name>-gguf

.PARAMETER HfRepo
    The HuggingFace repository to quantize (e.g., "meta-llama/Llama-3.2-1B-Instruct")

.PARAMETER HfUsername
    Your HuggingFace username for uploading the quantized model

.PARAMETER ModelsDir
    Directory to store downloaded models (default: .\models)

.PARAMETER BuildDir
    Directory for llama.cpp build output, relative to the llama.cpp clone (default: build\bin\Release)

.PARAMETER SkipBuild
    Skip cloning/pulling and rebuilding llama.cpp

.PARAMETER ForceRebuild
    Delete the build directory and force a clean rebuild of llama.cpp

.PARAMETER SkipVenv
    Skip recreating the Python virtual environment

.PARAMETER SkipUpload
    Skip uploading to HuggingFace

.PARAMETER IMatrix
    Path to importance matrix file (optional, for IQ quantizations)

.PARAMETER KeepFiles
    Keep local files after successful upload (by default, files are deleted to conserve space)

.PARAMETER HfToken
    HuggingFace token for authentication (alternative to being already logged in)

.PARAMETER LlamaCppUrl
    URL of the llama.cpp repository to clone (default: https://github.com/ggml-org/llama.cpp.git)

.EXAMPLE
    .\quantize-e2e.ps1 -HfRepo "meta-llama/Llama-3.2-1B-Instruct" -HfUsername "your-username"

.EXAMPLE
    .\quantize-e2e.ps1 -HfRepo "meta-llama/Llama-3.2-1B-Instruct" -HfUsername "your-username" -HfToken "hf_xxxxxxxxxxxx"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$HfRepo,

    [Parameter(Mandatory=$false)]
    [string]$HfUsername = "",

    [Parameter(Mandatory=$false)]
    [string]$ModelsDir = "models",

    [Parameter(Mandatory=$false)]
    [string]$BuildDir = "build\bin\Release",

    [Parameter(Mandatory=$false)]
    [switch]$SkipBuild,

    [Parameter(Mandatory=$false)]
    [switch]$ForceRebuild,

    [Parameter(Mandatory=$false)]
    [switch]$SkipVenv,

    [Parameter(Mandatory=$false)]
    [switch]$SkipUpload,

    [Parameter(Mandatory=$false)]
    [string]$IMatrix = "",

    [Parameter(Mandatory=$false)]
    [int]$Threads = [Environment]::ProcessorCount,

    [Parameter(Mandatory=$false)]
    [switch]$KeepFiles,

    [Parameter(Mandatory=$false)]
    [string]$HfToken = "",

    [Parameter(Mandatory=$false)]
    [string]$LlamaCppUrl = "https://github.com/ggml-org/llama.cpp.git"
)

$ErrorActionPreference = "Stop"
$ScriptRoot = $PSScriptRoot
if (-not $ScriptRoot) { $ScriptRoot = Get-Location }

# llama.cpp lives as a subdirectory of this repo
$LlamaCppDir = Join-Path $ScriptRoot "llama.cpp"

# All available quantization types
$QuantTypes = @(
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ2_M",
    "IQ1_S",
    "IQ1_M",
    "TQ1_0",
    "TQ2_0",
    "Q2_K",
    "Q2_K_S",
    "IQ3_XXS",
    "IQ3_S",
    "IQ3_M",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "IQ3_XS",
    "IQ4_NL",
    "IQ4_XS",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
    "F16",
    "BF16"
)

# Quantization types that require importance matrix
$RequiresIMatrix = @("IQ1_S", "IQ1_M", "IQ2_S", "IQ2_XXS", "IQ2_XS", "Q2_K_S")

#region Helper Functions

function Write-ColoredOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host "  $Message" -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
}

function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

function Get-ModelNameFromRepo {
    param([string]$Repo)
    return ($Repo -split "/")[-1]
}

function Test-IsMultimodalModel {
    param([string]$ModelPath)

    Write-ColoredOutput "Checking if model is multimodal/vision..." "Yellow"

    # Check config.json for vision-related keys
    $ConfigPath = Join-Path $ModelPath "config.json"
    if (Test-Path $ConfigPath) {
        $Config = Get-Content $ConfigPath -Raw | ConvertFrom-Json

        # Common indicators of multimodal/vision models
        $VisionIndicators = @(
            "vision_config",
            "visual_config",
            "image_size",
            "vision_encoder",
            "visual_encoder",
            "image_processor",
            "vision_tower",
            "mm_projector",
            "multimodal",
            "image_token_index"
        )

        $ConfigString = $Config | ConvertTo-Json -Depth 10
        foreach ($indicator in $VisionIndicators) {
            if ($ConfigString -match $indicator) {
                Write-ColoredOutput "  Found vision indicator: $indicator" "Green"
                return $true
            }
        }
    }

    # Check README.md for vision/multimodal mentions
    $ReadmePath = Join-Path $ModelPath "README.md"
    if (Test-Path $ReadmePath) {
        $ReadmeContent = Get-Content $ReadmePath -Raw
        $VisionKeywords = @("vision", "multimodal", "VLM", "vision-language", "image", "visual")
        foreach ($keyword in $VisionKeywords) {
            if ($ReadmeContent -match "\b$keyword\b") {
                Write-ColoredOutput "  Found in README: $keyword" "Green"
                return $true
            }
        }
    }

    # Check for preprocessor_config.json (image processors)
    $PreprocessorPath = Join-Path $ModelPath "preprocessor_config.json"
    if (Test-Path $PreprocessorPath) {
        Write-ColoredOutput "  Found preprocessor_config.json" "Green"
        return $true
    }

    Write-ColoredOutput "  Model does not appear to be multimodal" "Gray"
    return $false
}

function Get-OriginalModelCard {
    param([string]$ModelPath)

    $ReadmePath = Join-Path $ModelPath "README.md"
    if (Test-Path $ReadmePath) {
        return Get-Content $ReadmePath -Raw
    }
    return ""
}

function Initialize-HfRepo {
    param(
        [string]$OutputDir,
        [string]$RepoName,
        [string]$FullRepoName
    )

    Write-ColoredOutput "Repository ready for uploads: $FullRepoName" "Green"
    return $true
}

function Push-SingleFile {
    param(
        [string]$OutputDir,
        [string]$FilePath,
        [string]$FullRepoName,
        [string]$CommitMessage,
        [bool]$DeleteAfterPush = $true
    )

    $FileName = Split-Path $FilePath -Leaf

    try {
        Write-ColoredOutput "    Uploading: $FileName" "Yellow"

        & hf upload $FullRepoName $FilePath $FileName --commit-message $CommitMessage

        if ($LASTEXITCODE -eq 0) {
            Write-ColoredOutput "    Uploaded: $FileName" "Green"

            if ($DeleteAfterPush) {
                Remove-Item $FilePath -Force
                Write-ColoredOutput "    Deleted local copy to save space" "Gray"
            }
            return $true
        } else {
            Write-ColoredOutput "    Upload failed for: $FileName (exit code: $LASTEXITCODE)" "Red"
            return $false
        }
    } catch {
        Write-ColoredOutput "    Error uploading $FileName : $($_.Exception.Message)" "Red"
        return $false
    }
}

#endregion

#region Main Script

Write-Host ""
Write-Host "  llama.cpp End-to-End Quantization Script" -ForegroundColor Magenta
Write-Host "  =========================================" -ForegroundColor Magenta
Write-Host ""
Write-ColoredOutput "Repository: $HfRepo" "Cyan"
Write-ColoredOutput "HF Username: $(if ($HfUsername) { $HfUsername } else { '(not set - upload will be skipped)' })" "Cyan"
Write-ColoredOutput "Models Directory: $ModelsDir" "Cyan"
Write-ColoredOutput "Build Directory: $BuildDir" "Cyan"
Write-ColoredOutput "Threads: $Threads" "Cyan"
Write-ColoredOutput "HF Token: $(if ($HfToken) { '(provided)' } else { '(not provided)' })" "Cyan"
Write-ColoredOutput "llama.cpp directory: $LlamaCppDir" "Cyan"

# Extract model name
$ModelName = Get-ModelNameFromRepo $HfRepo
$OutputRepoName = "$ModelName-GGUF"
Write-ColoredOutput "Output Repository: $OutputRepoName" "Cyan"

#region Step 1: Clone or update llama.cpp, then rebuild

if (-not $SkipBuild) {
    Write-Step "Step 1: Cloning/updating llama.cpp and rebuilding"

    if (-not (Test-Path $LlamaCppDir)) {
        Write-ColoredOutput "llama.cpp not found, cloning from $LlamaCppUrl ..." "Yellow"
        git clone $LlamaCppUrl $LlamaCppDir
        if ($LASTEXITCODE -ne 0) {
            Write-ColoredOutput "Failed to clone llama.cpp!" "Red"
            exit 1
        }
        Write-ColoredOutput "llama.cpp cloned successfully!" "Green"
    } else {
        # Check current commit
        $OldCommit = git -C $LlamaCppDir rev-parse HEAD 2>$null
        Write-ColoredOutput "Current llama.cpp commit: $OldCommit" "Gray"

        Write-ColoredOutput "Pulling latest llama.cpp changes..." "Yellow"
        git -C $LlamaCppDir pull
        if ($LASTEXITCODE -ne 0) {
            Write-ColoredOutput "Warning: git pull failed, continuing with current version" "Yellow"
        }

        $NewCommit = git -C $LlamaCppDir rev-parse HEAD 2>$null
        Write-ColoredOutput "New llama.cpp commit: $NewCommit" "Gray"
    }

    # Force rebuild: delete build directory inside llama.cpp
    if ($ForceRebuild) {
        $BuildPath = Join-Path $LlamaCppDir "build"
        if (Test-Path $BuildPath) {
            Write-ColoredOutput "Force rebuild requested, deleting build directory..." "Yellow"
            Remove-Item -Recurse -Force $BuildPath
            Write-ColoredOutput "Build directory deleted" "Green"
        }
    }

    # Rebuild if there are changes, build doesn't exist, or force rebuild requested
    $QuantizePath = Join-Path (Join-Path $LlamaCppDir $BuildDir) "llama-quantize.exe"
    $OldCommit = if ($OldCommit) { $OldCommit } else { "" }
    $NewCommit = if ($NewCommit) { $NewCommit } else { "new" }

    if ($ForceRebuild -or $OldCommit -ne $NewCommit -or -not (Test-Path $QuantizePath)) {
        if ($ForceRebuild) {
            Write-ColoredOutput "Force rebuild requested, rebuilding..." "Yellow"
        } else {
            Write-ColoredOutput "Changes detected or build missing, rebuilding..." "Yellow"
        }

        Push-Location $LlamaCppDir
        try {
            # Configure CMake
            Write-ColoredOutput "Configuring CMake..." "Yellow"
            cmake -B build
            if ($LASTEXITCODE -ne 0) {
                Write-ColoredOutput "CMake configuration failed!" "Red"
                exit 1
            }

            # Build
            Write-ColoredOutput "Building llama.cpp..." "Yellow"
            cmake --build build --config Release -j $Threads
            if ($LASTEXITCODE -ne 0) {
                Write-ColoredOutput "Build failed!" "Red"
                exit 1
            }
        } finally {
            Pop-Location
        }

        Write-ColoredOutput "Build completed successfully!" "Green"
    } else {
        Write-ColoredOutput "No changes detected, skipping rebuild" "Green"
    }
} else {
    Write-Step "Step 1: Skipping build (--SkipBuild)"

    # Still ensure llama.cpp exists for convert script and requirements
    if (-not (Test-Path $LlamaCppDir)) {
        Write-ColoredOutput "llama.cpp not found at $LlamaCppDir, cloning..." "Yellow"
        git clone $LlamaCppUrl $LlamaCppDir
        if ($LASTEXITCODE -ne 0) {
            Write-ColoredOutput "Failed to clone llama.cpp!" "Red"
            exit 1
        }
    }
}

#endregion

#region Step 2: Setup Python venv

if (-not $SkipVenv) {
    Write-Step "Step 2: Setting up Python virtual environment"

    $VenvPath = Join-Path $ScriptRoot "venv"

    # Remove existing venv
    if (Test-Path $VenvPath) {
        Write-ColoredOutput "Removing existing venv..." "Yellow"
        Remove-Item -Recurse -Force $VenvPath
    }

    # Create new venv
    Write-ColoredOutput "Creating new virtual environment..." "Yellow"
    py -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-ColoredOutput "Failed to create venv!" "Red"
        exit 1
    }

    # Activate venv
    Write-ColoredOutput "Activating virtual environment..." "Yellow"
    & "$VenvPath\Scripts\Activate.ps1"

    # Install requirements from llama.cpp
    Write-ColoredOutput "Installing requirements from llama.cpp..." "Yellow"
    pip install --upgrade pip
    pip install -r (Join-Path $LlamaCppDir "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        Write-ColoredOutput "Failed to install requirements!" "Red"
        exit 1
    }

    # Install huggingface_hub for uploads
    pip install huggingface_hub

    Write-ColoredOutput "Python environment setup complete!" "Green"
} else {
    Write-Step "Step 2: Skipping venv setup (--SkipVenv)"

    # Still activate existing venv
    $VenvPath = Join-Path $ScriptRoot "venv"
    if (Test-Path "$VenvPath\Scripts\Activate.ps1") {
        & "$VenvPath\Scripts\Activate.ps1"
    }
}

#endregion

#region Step 3: Clone model from HuggingFace

Write-Step "Step 3: Cloning model from HuggingFace"

# Create models directory if it doesn't exist
$ModelsPath = Join-Path $ScriptRoot $ModelsDir
if (-not (Test-Path $ModelsPath)) {
    New-Item -ItemType Directory -Path $ModelsPath | Out-Null
}

$ModelPath = Join-Path $ModelsPath $ModelName

# Remove existing model directory if it exists
if (Test-Path $ModelPath) {
    Write-ColoredOutput "Removing existing model directory..." "Yellow"
    Remove-Item -Recurse -Force $ModelPath
}

# Clone using git
Write-ColoredOutput "Cloning $HfRepo..." "Yellow"

# Build URL with token if provided
if ($HfToken) {
    $HfUrl = "https://hf_user:$HfToken@huggingface.co/$HfRepo"
    Write-ColoredOutput "Using provided HF token for clone authentication" "Yellow"
} else {
    $HfUrl = "https://huggingface.co/$HfRepo"
}

# Check if git-xet is available and install it
$XetVersion = git-xet --version 2>$null
if ($XetVersion) {
    Write-ColoredOutput "Found $XetVersion" "Cyan"
    git-xet install 2>$null
} else {
    Write-ColoredOutput "Warning: git-xet not found, using regular git clone" "Yellow"
}

Push-Location $ModelsPath
try {
    git clone $HfUrl

    if ($LASTEXITCODE -ne 0) {
        Write-ColoredOutput "Failed to clone repository!" "Red"
        exit 1
    }
} finally {
    Pop-Location
}

Write-ColoredOutput "Model cloned successfully!" "Green"

# Save original model card for attribution
$OriginalModelCard = Get-OriginalModelCard $ModelPath

#endregion

#region Step 4: Check for multimodal/vision model

Write-Step "Step 4: Detecting model type"

$IsMultimodal = Test-IsMultimodalModel $ModelPath

if ($IsMultimodal) {
    Write-ColoredOutput "Model is MULTIMODAL - will create mmproj" "Magenta"
} else {
    Write-ColoredOutput "Model is TEXT-ONLY" "Green"
}

#endregion

#region Step 5: Convert to GGUF

Write-Step "Step 5: Converting to GGUF format"

$ConvertScript = Join-Path $LlamaCppDir "convert_hf_to_gguf.py"
if (-not (Test-Path $ConvertScript)) {
    Write-ColoredOutput "Error: convert_hf_to_gguf.py not found at: $ConvertScript" "Red"
    exit 1
}

# Create output directory for GGUF files
$GgufOutputDir = Join-Path $ModelsPath "$ModelName-gguf"
if (-not (Test-Path $GgufOutputDir)) {
    New-Item -ItemType Directory -Path $GgufOutputDir | Out-Null
}

# Convert main model (bf16/f16)
Write-ColoredOutput "Converting main model to GGUF (BF16)..." "Yellow"
python $ConvertScript $ModelPath --outfile "$GgufOutputDir\" --outtype bf16
if ($LASTEXITCODE -ne 0) {
    Write-ColoredOutput "Failed to convert main model!" "Red"
    exit 1
}

# Find the generated base GGUF file
$BaseGguf = Get-ChildItem -Path $GgufOutputDir -Filter "*.gguf" | Where-Object { $_.Name -notmatch "mmproj" } | Select-Object -First 1
if (-not $BaseGguf) {
    Write-ColoredOutput "Could not find generated GGUF file!" "Red"
    exit 1
}
Write-ColoredOutput "Base model: $($BaseGguf.Name)" "Green"

# Convert mmproj if multimodal
$MmprojFile = $null
if ($IsMultimodal) {
    Write-ColoredOutput "Converting mmproj for vision support..." "Yellow"
    try {
        python $ConvertScript $ModelPath --outfile "$GgufOutputDir\" --mmproj --outtype f16
        if ($LASTEXITCODE -eq 0) {
            $MmprojFile = Get-ChildItem -Path $GgufOutputDir -Filter "*mmproj*.gguf" | Select-Object -First 1
            if ($MmprojFile) {
                Write-ColoredOutput "mmproj created: $($MmprojFile.Name)" "Green"
            }
        }
    } catch {
        Write-ColoredOutput "Warning: mmproj conversion failed - model may not support it" "Yellow"
    }
}

Write-ColoredOutput "GGUF conversion complete!" "Green"

#endregion

#region Step 6: Clean up original model files

Write-Step "Step 6: Cleaning up original model files"

# Files to delete (safetensors, bin, pt files)
$DeletePatterns = @(
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ot",
    "model*.safetensors*"
)

$DeletedSize = 0
$DeletedCount = 0

foreach ($pattern in $DeletePatterns) {
    $FilesToDelete = Get-ChildItem -Path $ModelPath -Filter $pattern -Recurse -ErrorAction SilentlyContinue
    foreach ($file in $FilesToDelete) {
        $DeletedSize += $file.Length
        $DeletedCount++
        Write-ColoredOutput "  Deleting: $($file.Name)" "Gray"
        Remove-Item -Path $file.FullName -Force
    }
}

$DeletedSizeGB = [math]::Round($DeletedSize / 1GB, 2)
Write-ColoredOutput "Deleted $DeletedCount files ($DeletedSizeGB GB)" "Green"

#endregion

#region Step 6.5: Setup HuggingFace repository for incremental uploads

$HfRepoReady = $false
$FullRepoName = ""

if (-not $SkipUpload -and $HfUsername) {
    Write-Step "Step 6.5: Setting up HuggingFace repository"

    $FullRepoName = "$HfUsername/$OutputRepoName"
    Write-ColoredOutput "Preparing repository: $FullRepoName" "Yellow"

    Write-ColoredOutput "Checking HuggingFace authentication..." "Yellow"

    $IsAuthenticated = $false
    $AuthMethod = ""

    if ($HfToken) {
        $env:HF_TOKEN = $HfToken
        Write-ColoredOutput "Using provided HF token for authentication" "Green"
        $IsAuthenticated = $true
        $AuthMethod = "token"
    } else {
        $HfWhoami = hf auth whoami 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColoredOutput "Authenticated as: $HfWhoami" "Green"
            $IsAuthenticated = $true
            $AuthMethod = "login"
        }
    }

    if (-not $IsAuthenticated) {
        Write-ColoredOutput "Please login to HuggingFace first:" "Yellow"
        Write-ColoredOutput "  hf auth login" "Cyan"
        Write-ColoredOutput "Or provide a token with -HfToken parameter" "Cyan"
        Write-ColoredOutput "Will continue without uploading." "Yellow"
    } else {
        Write-ColoredOutput "Creating repository if needed..." "Yellow"
        hf repo create $OutputRepoName --repo-type model --exist-ok 2>$null

        if (Initialize-HfRepo -OutputDir $GgufOutputDir -RepoName $OutputRepoName -FullRepoName $FullRepoName) {
            Write-ColoredOutput "Repository ready for incremental uploads" "Green"
            $HfRepoReady = $true
        } else {
            Write-ColoredOutput "Failed to setup repository - will skip uploads" "Yellow"
        }
    }
} else {
    if (-not $HfUsername) {
        Write-ColoredOutput "No HuggingFace username - skipping upload setup" "Gray"
    }
    if ($SkipUpload) {
        Write-ColoredOutput "Upload disabled - skipping upload setup" "Gray"
    }
}

#endregion

# Track files for README generation (since we delete after upload)
$TrackedFiles = @()

# Track base model
$TrackedFiles += [PSCustomObject]@{
    Name = $BaseGguf.Name
    Size = $BaseGguf.Length
    QuantType = if ($BaseGguf.Name -match "BF16") { "BF16" } else { "F16" }
}

# Track mmproj if it exists
if ($MmprojFile) {
    $TrackedFiles += [PSCustomObject]@{
        Name = $MmprojFile.Name
        Size = $MmprojFile.Length
        QuantType = "Vision Projector"
    }

    if ($HfRepoReady) {
        Write-ColoredOutput "Uploading mmproj file..." "Yellow"
        $PushResult = Push-SingleFile -OutputDir $GgufOutputDir -FilePath $MmprojFile.FullName -FullRepoName $FullRepoName -CommitMessage "Add vision projector (mmproj)" -DeleteAfterPush $true
        if ($PushResult) {
            Write-ColoredOutput "mmproj uploaded and local copy deleted" "Green"
        }
    }
}

#region Step 7: Create all quantizations

Write-Step "Step 7: Creating all quantizations"

$QuantizePath = Join-Path (Join-Path $LlamaCppDir $BuildDir) "llama-quantize.exe"
if (-not (Test-Path $QuantizePath)) {
    Write-ColoredOutput "Error: llama-quantize.exe not found at: $QuantizePath" "Red"
    exit 1
}

$SuccessCount = 0
$SkippedCount = 0
$FailedCount = 0
$UploadedCount = 0
$TotalCount = $QuantTypes.Count

if ($IMatrix -ne "" -and -not (Test-Path $IMatrix)) {
    Write-ColoredOutput "Warning: Importance matrix not found at: $IMatrix" "Yellow"
    $IMatrix = ""
}

foreach ($QuantType in $QuantTypes) {
    $BaseModelName = [System.IO.Path]::GetFileNameWithoutExtension($BaseGguf.FullName)
    $BaseModelName = $BaseModelName -replace "-BF16$|-F16$|-F32$", ""
    $OutputModel = Join-Path $GgufOutputDir "$BaseModelName-$QuantType.gguf"

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

    $QuantArgs = @()

    if ($IMatrix -ne "") {
        $QuantArgs += "--imatrix"
        $QuantArgs += "`"$IMatrix`""
    }

    $QuantArgs += "`"$($BaseGguf.FullName)`""
    $QuantArgs += "`"$OutputModel`""
    $QuantArgs += $QuantType
    $QuantArgs += $Threads

    try {
        $StartTime = Get-Date
        $Process = Start-Process -FilePath $QuantizePath -ArgumentList $QuantArgs -Wait -PassThru -NoNewWindow
        $EndTime = Get-Date
        $Duration = [math]::Round(($EndTime - $StartTime).TotalSeconds, 1)

        if ($Process.ExitCode -eq 0 -and (Test-Path $OutputModel)) {
            $FileInfo = Get-Item $OutputModel
            $SizeMB = [math]::Round($FileInfo.Length / 1MB, 2)
            Write-ColoredOutput "  Success: $QuantType (${SizeMB}MB, ${Duration}s)" "Green"
            $SuccessCount++

            $TrackedFiles += [PSCustomObject]@{
                Name = $FileInfo.Name
                Size = $FileInfo.Length
                QuantType = $QuantType
            }

            if ($HfRepoReady) {
                $PushResult = Push-SingleFile -OutputDir $GgufOutputDir -FilePath $OutputModel -FullRepoName $FullRepoName -CommitMessage "Add $QuantType quantization" -DeleteAfterPush $true
                if ($PushResult) {
                    $UploadedCount++
                } else {
                    Write-ColoredOutput "  Warning: Upload failed, keeping local file" "Yellow"
                }
            }
        } else {
            Write-ColoredOutput "  Failed: $QuantType (exit code: $($Process.ExitCode))" "Red"
            $FailedCount++
        }
    }
    catch {
        Write-ColoredOutput "  Error: $QuantType - $($_.Exception.Message)" "Red"
        $FailedCount++
    }
}

Write-Host ""
Write-ColoredOutput "Quantization Summary:" "Cyan"
Write-ColoredOutput "  Successful: $SuccessCount" "Green"
Write-ColoredOutput "  Skipped: $SkippedCount" "Yellow"
Write-ColoredOutput "  Failed: $FailedCount" "Red"
if ($HfRepoReady) {
    Write-ColoredOutput "  Uploaded: $UploadedCount" "Cyan"
}

#endregion

#region Step 8: Create README for quantized model

Write-Step "Step 8: Creating README for quantized repository"

$SortedTrackedFiles = $TrackedFiles | Sort-Object Size

$ReadmeContent = @"
---
license: other
license_name: same-as-original
license_link: https://huggingface.co/$HfRepo
base_model: $HfRepo
tags:
- gguf
- quantized
- llama.cpp
---

# $OutputRepoName

This repository contains GGUF quantizations of [$HfRepo](https://huggingface.co/$HfRepo).

## Quantizations

| Filename | Size | Description |
|----------|------|-------------|
"@

foreach ($file in $SortedTrackedFiles) {
    $SizeMB = [math]::Round($file.Size / 1MB, 2)
    $SizeGB = [math]::Round($file.Size / 1GB, 2)
    $SizeStr = if ($SizeGB -ge 1) { "${SizeGB} GB" } else { "${SizeMB} MB" }

    $QuantType = $file.QuantType

    $ReadmeContent += "| $($file.Name) | $SizeStr | $QuantType |`n"
}

$ReadmeContent += @"

## Usage

### With llama.cpp

``````bash
# Download a quantization
hf download $HfUsername/$OutputRepoName <filename> --local-dir .

# Run with llama.cpp
./llama-cli -m <filename> -p "Your prompt here"
``````

### With llama-cpp-python

``````python
from llama_cpp import Llama

llm = Llama(model_path="<filename>")
output = llm("Your prompt here", max_tokens=256)
print(output["choices"][0]["text"])
``````
"@

if ($MmprojFile) {
    $ReadmeContent += @"


## Vision Support

This model includes vision capabilities. Use the mmproj file for image understanding:

``````bash
./llama-mtmd-cli -m <model-file> --mmproj $($MmprojFile.Name) --image <image-path> -p "Describe this image"
``````
"@
}

$ReadmeContent += @"


## Quantization Details

These quantizations were created using [llama.cpp](https://github.com/ggml-org/llama.cpp).

- **Base Model:** [$HfRepo](https://huggingface.co/$HfRepo)
- **Quantization Tool:** llama.cpp convert_hf_to_gguf.py + llama-quantize

## Original Model Card

---

$OriginalModelCard
"@

$ReadmePath = Join-Path $GgufOutputDir "README.md"
$ReadmeContent | Out-File -FilePath $ReadmePath -Encoding utf8

Write-ColoredOutput "README.md created!" "Green"

#endregion

#region Step 9: Finalize upload to HuggingFace

$UploadSuccessful = $false

if ($HfRepoReady) {
    Write-Step "Step 9: Finalizing HuggingFace upload"

    Write-ColoredOutput "Pushing README and base model..." "Yellow"

    # Push README
    $ReadmePushResult = Push-SingleFile -OutputDir $GgufOutputDir -FilePath $ReadmePath -FullRepoName $FullRepoName -CommitMessage "Add README" -DeleteAfterPush $false
    if ($ReadmePushResult) {
        Write-ColoredOutput "README uploaded" "Green"
    }

    # Push base model (BF16/F16)
    if (Test-Path $BaseGguf.FullName) {
        Write-ColoredOutput "Uploading base model ($($BaseGguf.Name))..." "Yellow"
        $BasePushResult = Push-SingleFile -OutputDir $GgufOutputDir -FilePath $BaseGguf.FullName -FullRepoName $FullRepoName -CommitMessage "Add $($BaseGguf.Name -replace '.*-(BF16|F16).*','$1') base model" -DeleteAfterPush $true
        if ($BasePushResult) {
            Write-ColoredOutput "Base model uploaded and local copy deleted" "Green"
            $UploadSuccessful = $true
        } else {
            Write-ColoredOutput "Base model upload failed" "Red"
        }
    } else {
        $UploadSuccessful = $true
    }

    if ($UploadSuccessful) {
        Write-ColoredOutput "All uploads complete!" "Green"
        Write-ColoredOutput "Repository: https://huggingface.co/$FullRepoName" "Cyan"
    }

} elseif (-not $SkipUpload -and $HfUsername) {
    Write-Step "Step 9: Uploading to HuggingFace"

    $FullRepoName = "$HfUsername/$OutputRepoName"
    Write-ColoredOutput "Uploading to: $FullRepoName" "Yellow"

    $IsAuthenticated = $false

    if ($HfToken) {
        $env:HF_TOKEN = $HfToken
        Write-ColoredOutput "Using provided HF token for authentication" "Green"
        $IsAuthenticated = $true
    } else {
        $HfWhoami = hf auth whoami 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColoredOutput "Authenticated as: $HfWhoami" "Green"
            $IsAuthenticated = $true
        }
    }

    if (-not $IsAuthenticated) {
        Write-ColoredOutput "Please login to HuggingFace first:" "Yellow"
        Write-ColoredOutput "  hf auth login" "Cyan"
        Write-ColoredOutput "Or provide a token with -HfToken parameter" "Cyan"
        Write-ColoredOutput "Skipping upload." "Yellow"
    } else {
        hf repo create $OutputRepoName --repo-type model --exist-ok 2>$null

        if (Initialize-HfRepo -OutputDir $GgufOutputDir -RepoName $OutputRepoName -FullRepoName $FullRepoName) {
            $AllFilesUploaded = $true

            $FilesToUpload = Get-ChildItem -Path $GgufOutputDir -Filter "*.gguf"
            $FilesToUpload += Get-Item $ReadmePath

            foreach ($FileToUpload in $FilesToUpload) {
                Write-ColoredOutput "  Uploading: $($FileToUpload.Name)..." "Yellow"
                $Result = Push-SingleFile -OutputDir $GgufOutputDir -FilePath $FileToUpload.FullName -FullRepoName $FullRepoName -CommitMessage "Add $($FileToUpload.Name)" -DeleteAfterPush ($FileToUpload.Extension -eq ".gguf")
                if (-not $Result) {
                    $AllFilesUploaded = $false
                    Write-ColoredOutput "  Failed to upload: $($FileToUpload.Name)" "Red"
                }
            }

            if ($AllFilesUploaded) {
                Write-ColoredOutput "Upload complete!" "Green"
                Write-ColoredOutput "Repository: https://huggingface.co/$FullRepoName" "Cyan"
                $UploadSuccessful = $true
            } else {
                Write-ColoredOutput "Some uploads failed, please check manually" "Yellow"
            }
        }
    }
} else {
    Write-Step "Step 9: Skipping upload"
    if (-not $HfUsername) {
        Write-ColoredOutput "No HuggingFace username provided (-HfUsername)" "Yellow"
    }
    if ($SkipUpload) {
        Write-ColoredOutput "Upload skipped via --SkipUpload flag" "Yellow"
    }
}

#endregion

#region Step 10: Cleanup after successful upload

if ($UploadSuccessful -and -not $KeepFiles) {
    Write-Step "Step 10: Cleaning up local files after successful upload"

    $TotalCleanedSize = 0

    if (Test-Path $ModelPath) {
        $ModelPathSize = (Get-ChildItem -Path $ModelPath -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $TotalCleanedSize += $ModelPathSize
    }
    if (Test-Path $GgufOutputDir) {
        $GgufOutputSize = (Get-ChildItem -Path $GgufOutputDir -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $TotalCleanedSize += $GgufOutputSize
    }

    if (Test-Path $ModelPath) {
        Write-ColoredOutput "Deleting original model folder: $ModelPath" "Yellow"
        Remove-Item -Recurse -Force $ModelPath -ErrorAction SilentlyContinue
        if (-not (Test-Path $ModelPath)) {
            Write-ColoredOutput "  Deleted successfully" "Green"
        } else {
            Write-ColoredOutput "  Warning: Could not fully delete folder" "Yellow"
        }
    }

    if (Test-Path $GgufOutputDir) {
        Write-ColoredOutput "Deleting GGUF output folder: $GgufOutputDir" "Yellow"
        Remove-Item -Recurse -Force $GgufOutputDir -ErrorAction SilentlyContinue
        if (-not (Test-Path $GgufOutputDir)) {
            Write-ColoredOutput "  Deleted successfully" "Green"
        } else {
            Write-ColoredOutput "  Warning: Could not fully delete folder" "Yellow"
        }
    }

    $TotalCleanedGB = [math]::Round($TotalCleanedSize / 1GB, 2)
    Write-ColoredOutput "Freed up $TotalCleanedGB GB of disk space" "Green"

} elseif ($UploadSuccessful -and $KeepFiles) {
    Write-Step "Step 10: Keeping local files (-KeepFiles specified)"
    Write-ColoredOutput "Files retained at: $GgufOutputDir" "Cyan"
} elseif (-not $UploadSuccessful -and -not $SkipUpload -and $HfUsername) {
    Write-ColoredOutput "`nLocal files retained (upload was not successful)" "Yellow"
}

#endregion

#region Final Summary

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Green
Write-Host "  QUANTIZATION COMPLETE!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green
Write-Host ""
Write-ColoredOutput "Source Model: $HfRepo" "Cyan"
Write-ColoredOutput "Multimodal: $(if ($IsMultimodal) { 'Yes' } else { 'No' })" "Cyan"

if ($UploadSuccessful) {
    Write-Host ""
    Write-ColoredOutput "HuggingFace Repository: https://huggingface.co/$HfUsername/$OutputRepoName" "Cyan"

    if (-not $KeepFiles) {
        Write-ColoredOutput "Local files: Cleaned up to conserve space" "Green"
    } else {
        Write-ColoredOutput "Local files: Retained at $GgufOutputDir" "Cyan"
    }
} else {
    Write-ColoredOutput "Output Directory: $GgufOutputDir" "Cyan"

    if (Test-Path $GgufOutputDir) {
        Write-Host ""
        Write-ColoredOutput "Generated Files:" "Cyan"
        Get-ChildItem -Path $GgufOutputDir -Filter "*.gguf" | Sort-Object Length | ForEach-Object {
            $SizeGB = [math]::Round($_.Length / 1GB, 2)
            Write-ColoredOutput "  $($_.Name) ($SizeGB GB)" "White"
        }
    }
}

Write-Host ""

#endregion
