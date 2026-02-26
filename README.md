# Quantize-All Scripts for llama.cpp

These scripts automatically quantize a GGUF model using all available quantization formats in llama.cpp.

## Files

- `quantize-e2e.ps1` - **End-to-end script** (full pipeline from HuggingFace to upload)
- `quantize-all.ps1` - PowerShell version (recommended for Windows, requires existing GGUF)
- `quantize-all.bat` - Batch file version (simpler alternative)

---

## End-to-End Script (quantize-e2e.ps1)

The `quantize-e2e.ps1` script provides a complete automated workflow:

1. Pulls latest llama.cpp and rebuilds if updated
2. Creates fresh Python venv and installs requirements
3. Clones model from HuggingFace (supports git-xet for faster downloads)
4. Detects multimodal/vision models and creates mmproj if needed
5. Converts to GGUF format
6. Deletes original safetensors to save space
7. Creates all quantization variants
8. Uploads to your HuggingFace account with proper attribution

### Prerequisites for E2E Script

1. **Git** with git-xet installed (for HuggingFace downloads/uploads)
   - Install: `pip install xet && git xet install`
2. **Python** (accessible via `py` command)
3. **CMake** and Visual Studio Build Tools
4. **CUDA Toolkit** (if building with GPU support)
5. **HuggingFace account** (optional, for uploading)

### E2E Usage

```powershell
# Basic usage - download, convert, and quantize
.\quantize-e2e.ps1 -HfRepo "meta-llama/Llama-3.2-1B-Instruct"

# With upload to your HuggingFace account
.\quantize-e2e.ps1 -HfRepo "meta-llama/Llama-3.2-1B-Instruct" -HfUsername "your-username"

# Skip rebuilding llama.cpp (if already up to date)
.\quantize-e2e.ps1 -HfRepo "Qwen/Qwen2-VL-7B-Instruct" -HfUsername "your-username" -SkipBuild

# Skip venv recreation (if dependencies already installed)
.\quantize-e2e.ps1 -HfRepo "mistralai/Mistral-7B-v0.1" -SkipVenv

# Custom models directory
.\quantize-e2e.ps1 -HfRepo "meta-llama/Llama-3.2-3B" -ModelsDir "D:\Models"

# With importance matrix for better low-bit quantizations
.\quantize-e2e.ps1 -HfRepo "meta-llama/Llama-3.2-1B" -IMatrix "calibration.dat"
```

### E2E Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `-HfRepo` | Yes | - | HuggingFace repository (e.g., "meta-llama/Llama-3.2-1B") |
| `-HfUsername` | No | - | Your HuggingFace username for uploads |
| `-ModelsDir` | No | `models` | Directory to store downloaded models |
| `-BuildDir` | No | `build\bin\Release` | llama.cpp build directory |
| `-SkipBuild` | No | $false | Skip git pull and rebuild |
| `-SkipVenv` | No | $false | Skip Python venv recreation |
| `-SkipUpload` | No | $false | Skip uploading to HuggingFace |
| `-IMatrix` | No | - | Path to importance matrix file |
| `-Threads` | No | All cores | Number of threads for quantization |
| `-KeepFiles` | No | $false | Keep local files after upload (default: delete to save space) |

### Multimodal/Vision Model Detection

The script automatically detects vision-language models by checking:
- `config.json` for vision-related keys (vision_config, image_size, etc.)
- `README.md` for keywords (vision, multimodal, VLM, etc.)
- Presence of `preprocessor_config.json`

When detected, it creates an `mmproj-*.gguf` file for the vision encoder.

### Output Structure (E2E)

```
models/
└── YourModel-gguf/
    ├── README.md                     # Auto-generated with attribution
    ├── YourModel-BF16.gguf           # Base model (kept for reference)
    ├── YourModel-Q4_K_M.gguf
    ├── YourModel-Q5_K_M.gguf
    ├── YourModel-Q8_0.gguf
    ├── mmproj-YourModel-F16.gguf     # (if multimodal)
    └── ...                           # All other quantizations
```

### HuggingFace Upload

To upload, first authenticate:
```powershell
hf auth login
```

The script will:
1. Create a new repo named `<ModelName>-GGUF` under your account
2. Generate a README with model card, usage instructions, and original attribution
3. Upload all GGUF files using git-xet (HuggingFace's storage backend)

**How authentication works:**
- Run `hf auth login` once before using the script
- This stores your HuggingFace token locally (~/.cache/huggingface/token)
- The script uses `hf auth whoami` to verify you're logged in
- Once logged in, git credential helper handles authentication for pushes

**How git-xet works:**
- Run `git xet install` once to set up xet globally
- The script also runs `git xet install` in each repo to enable xet
- Once enabled, regular `git clone` and `git push` commands work seamlessly
- Xet handles large files (like GGUFs) efficiently without separate tracking

### Automatic Cleanup

**By default**, after a successful upload, the script automatically deletes:
- The original cloned model folder (safetensors, configs, etc.)
- The GGUF output folder (all quantized files)

This conserves disk space since the files are now on HuggingFace.

To **keep local files** after upload:
```powershell
.\quantize-e2e.ps1 -HfRepo "meta-llama/Llama-3.2-1B" -HfUsername "you" -KeepFiles
```

**Note:** Files are only deleted if the upload succeeds. If upload fails, files are retained so you can retry or upload manually.

---

## Prerequisites

1. **Built llama.cpp**: Make sure you have built llama.cpp with the quantization tool:
   ```bash
   mkdir build
   cd build
   cmake .. -DGGML_CUDA=ON  # or other backends as needed
   cmake --build . --config Release --target llama-quantize
   ```

2. **Base GGUF Model**: You need a base (typically F16/BF16/F32) GGUF model file in your model folder.

## Usage

### PowerShell Script (Recommended)

```powershell
.\quantize-all.ps1 -ModelFolder "C:\path\to\your\model\folder"
```

**Optional Parameters:**
- `-BuildDir`: Path to build directory (default: `build\bin\Release`)
- `-IMatrix`: Path to importance matrix file for better quality on low-bit quantizations
- `-Threads`: Number of threads to use (default: all CPU cores)

**Examples:**
```powershell
# Basic usage
.\quantize-all.ps1 -ModelFolder "C:\Models\Llama-3.2-3B"

# With custom build directory
.\quantize-all.ps1 -ModelFolder "C:\Models\Llama-3.2-3B" -BuildDir "custom_build\bin"

# With importance matrix for better quality
.\quantize-all.ps1 -ModelFolder "C:\Models\Llama-3.2-3B" -IMatrix "importance_matrix.dat"

# With custom thread count
.\quantize-all.ps1 -ModelFolder "C:\Models\Llama-3.2-3B" -Threads 8
```

### Batch Script

```cmd
quantize-all.bat "C:\path\to\your\model\folder"
```

**Optional Parameters:**
1. Build directory (default: `build\bin\Release`)
2. Importance matrix file

**Examples:**
```cmd
REM Basic usage
quantize-all.bat "C:\Models\Llama-3.2-3B"

REM With custom build directory
quantize-all.bat "C:\Models\Llama-3.2-3B" "custom_build\bin"

REM With importance matrix
quantize-all.bat "C:\Models\Llama-3.2-3B" "build\bin\Release" "importance_matrix.dat"
```

## Quantization Types

The scripts will create the following quantizations:

### Standard Quantizations
| Type | Description | Size Impact | Quality Impact |
|------|-------------|-------------|----------------|
| **Q4_0** | 4-bit quantization | ~4.3GB | +0.47 ppl |
| **Q4_1** | 4-bit quantization (improved) | ~4.8GB | +0.45 ppl |
| **Q5_0** | 5-bit quantization | ~5.2GB | +0.13 ppl |
| **Q5_1** | 5-bit quantization (improved) | ~5.7GB | +0.11 ppl |
| **Q8_0** | 8-bit quantization | ~8.0GB | +0.003 ppl |
| **F16** | Half precision | ~14GB | +0.002 ppl |
| **BF16** | BFloat16 | ~14GB | -0.005 ppl |

### K-Quantizations (Recommended)
| Type | Description | Size Impact | Quality Impact |
|------|-------------|-------------|----------------|
| **Q2_K** | 2-bit k-quantization | ~2.96GB | +3.52 ppl |
| **Q2_K_S** | 2-bit k-quantization (small) | ~2.96GB | +3.18 ppl |
| **Q3_K_S** | 3-bit k-quantization (small) | ~3.41GB | +1.63 ppl |
| **Q3_K_M** | 3-bit k-quantization (medium) | ~3.74GB | +0.66 ppl |
| **Q3_K_L** | 3-bit k-quantization (large) | ~4.03GB | +0.56 ppl |
| **Q4_K_S** | 4-bit k-quantization (small) | ~4.37GB | +0.27 ppl |
| **Q4_K_M** | 4-bit k-quantization (medium) | ~4.58GB | +0.18 ppl |
| **Q5_K_S** | 5-bit k-quantization (small) | ~5.21GB | +0.10 ppl |
| **Q5_K_M** | 5-bit k-quantization (medium) | ~5.33GB | +0.06 ppl |
| **Q6_K** | 6-bit k-quantization | ~6.14GB | +0.02 ppl |

### I-Quantizations (Ultra-Low Bit)
| Type | Description | Size Impact | Requires iMatrix |
|------|-------------|-------------|------------------|
| **IQ1_S** | 1.56 bpw quantization | Ultra small | ✅ Yes |
| **IQ1_M** | 1.75 bpw quantization | Ultra small | ✅ Yes |
| **IQ2_XXS** | 2.06 bpw quantization | Very small | ✅ Yes |
| **IQ2_XS** | 2.31 bpw quantization | Very small | ✅ Yes |
| **IQ2_S** | 2.5 bpw quantization | Very small | ✅ Yes |
| **IQ2_M** | 2.7 bpw quantization | Small | No |
| **IQ3_XXS** | 3.06 bpw quantization | Small | No |
| **IQ3_XS** | 3.3 bpw quantization | Small | No |
| **IQ3_S** | 3.44 bpw quantization | Small | No |
| **IQ3_M** | 3.66 bpw quantization | Medium | No |
| **IQ4_XS** | 4.25 bpw quantization | Medium | No |
| **IQ4_NL** | 4.50 bpw quantization | Medium | No |

### Ternary Quantizations
| Type | Description | Size Impact |
|------|-------------|-------------|
| **TQ1_0** | 1.69 bpw ternarization | Ultra small |
| **TQ2_0** | 2.06 bpw ternarization | Ultra small |

## Output Structure

The scripts will create a `quantized` folder in your model directory:

```
YourModel/
├── model-f16.gguf                    # Your original model
└── quantized/
    ├── model-Q4_0.gguf
    ├── model-Q4_1.gguf
    ├── model-Q4_K_M.gguf
    ├── model-Q5_K_M.gguf
    ├── model-Q8_0.gguf
    └── ...                           # All other quantizations
```

## Creating an Importance Matrix

For the best quality with ultra-low bit quantizations (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, Q2_K_S), you should create an importance matrix first:

```bash
# Generate importance matrix using sample text
./llama-imatrix -m your-model-f16.gguf -f your-calibration-data.txt -o importance_matrix.dat

# Use with script
.\quantize-all.ps1 -ModelFolder "C:\Models\Llama-3.2-3B" -IMatrix "importance_matrix.dat"
```

## Recommendations

### For General Use:
- **Q4_K_M**: Best balance of size and quality
- **Q5_K_M**: Higher quality, slightly larger
- **Q8_0**: Nearly lossless, much larger

### For Resource-Constrained Devices:
- **Q3_K_M**: Good quality at small size
- **IQ3_M**: Even smaller, requires importance matrix for best results
- **Q2_K_S**: Smallest reasonable size

### For Maximum Quality:
- **F16** or **BF16**: Full precision
- **Q8_0**: Minimal quality loss
- **Q6_K**: Good compromise

## Troubleshooting

1. **"llama-quantize.exe not found"**: Make sure you've built llama.cpp and specify the correct build directory.

2. **"No GGUF files found"**: Ensure your model folder contains a .gguf file.

3. **Quantization failed**: Some quantizations may fail on certain models. This is normal - the script will continue with other formats.

4. **Out of memory**: Some quantizations require significant RAM. Try closing other applications or use a machine with more RAM.

5. **Very small quantizations produce poor quality**: Use an importance matrix for IQ1/IQ2 quantizations.

## Performance Notes

- Quantization time varies by format and model size
- I-quantizations typically take longer than K-quantizations
- Using all CPU cores (default) provides the fastest quantization
- The script will skip existing files, so you can resume interrupted runs

## Size Estimates (for 7B parameter models)

- F16/BF16: ~14GB
- Q8_0: ~8GB
- Q6_K: ~6GB
- Q5_K_M: ~5.3GB
- Q4_K_M: ~4.6GB
- Q3_K_M: ~3.7GB
- Q2_K: ~3GB
- IQ2_M: ~2.7GB
- IQ1_M: ~1.75GB

*Note: Actual sizes will vary based on the specific model architecture and content.*