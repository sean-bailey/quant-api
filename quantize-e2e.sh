#!/bin/bash

# End-to-end model quantization script for llama.cpp
#
# This script automates the complete workflow of quantizing a HuggingFace model:
# 1. Clones llama.cpp if not present, or pulls latest if it is
# 2. Rebuilds llama.cpp if needed
# 3. Recreates Python venv and installs requirements
# 4. Clones the specified HuggingFace model
# 5. Detects if model is multimodal/vision and creates mmproj if needed
# 6. Converts to GGUF format
# 7. Deletes original safetensors (keeping config files)
# 8. Creates all quantization variants
# 9. Uploads to HuggingFace under your account as <model-name>-gguf

set -euo pipefail

# Default values
HF_REPO=""
HF_USERNAME=""
MODELS_DIR="models"
BUILD_DIR="build/bin"
SKIP_BUILD=false
FORCE_REBUILD=false
SKIP_VENV=false
SKIP_UPLOAD=false
IMATRIX=""
THREADS=$(nproc)
KEEP_FILES=false
HF_TOKEN=""
LLAMA_CPP_URL="https://github.com/ggml-org/llama.cpp.git"

# All available quantization types
QUANT_TYPES=(
    "Q4_0"
    "Q4_1"
    "Q5_0"
    "Q5_1"
    "IQ2_XXS"
    "IQ2_XS"
    "IQ2_S"
    "IQ2_M"
    "IQ1_S"
    "IQ1_M"
    "TQ1_0"
    "TQ2_0"
    "Q2_K"
    "Q2_K_S"
    "IQ3_XXS"
    "IQ3_S"
    "IQ3_M"
    "Q3_K_S"
    "Q3_K_M"
    "Q3_K_L"
    "IQ3_XS"
    "IQ4_NL"
    "IQ4_XS"
    "Q4_K_S"
    "Q4_K_M"
    "Q5_K_S"
    "Q5_K_M"
    "Q6_K"
    "Q8_0"
    "F16"
    "BF16"
)

# Quantization types that require importance matrix
REQUIRES_IMATRIX=("IQ1_S" "IQ1_M" "IQ2_S" "IQ2_XXS" "IQ2_XS" "Q2_K_S")

# Helper functions
write_colored_output() {
    local message="$1"
    local color="${2:-white}"

    case "$color" in
        red) echo -e "\033[31m$message\033[0m" ;;
        green) echo -e "\033[32m$message\033[0m" ;;
        yellow) echo -e "\033[33m$message\033[0m" ;;
        blue) echo -e "\033[34m$message\033[0m" ;;
        magenta) echo -e "\033[35m$message\033[0m" ;;
        cyan) echo -e "\033[36m$message\033[0m" ;;
        gray) echo -e "\033[37m$message\033[0m" ;;
        *) echo "$message" ;;
    esac
}

write_step() {
    local message="$1"
    echo ""
    echo -e "\033[36m============================================================\033[0m"
    echo -e "\033[36m  $message\033[0m"
    echo -e "\033[36m============================================================\033[0m"
}

test_command() {
    command -v "$1" >/dev/null 2>&1
}

get_model_name_from_repo() {
    basename "$1"
}

test_is_multimodal_model() {
    local model_path="$1"

    write_colored_output "Checking if model is multimodal/vision..." "yellow"

    local config_path="$model_path/config.json"
    if [[ -f "$config_path" ]]; then
        local config_content
        config_content=$(cat "$config_path")

        local vision_indicators=(
            "vision_config"
            "visual_config"
            "image_size"
            "vision_encoder"
            "visual_encoder"
            "image_processor"
            "vision_tower"
            "mm_projector"
            "multimodal"
            "image_token_index"
        )

        for indicator in "${vision_indicators[@]}"; do
            if echo "$config_content" | grep -q "$indicator"; then
                write_colored_output "  Found vision indicator: $indicator" "green"
                return 0
            fi
        done
    fi

    local readme_path="$model_path/README.md"
    if [[ -f "$readme_path" ]]; then
        local readme_content
        readme_content=$(cat "$readme_path")
        local vision_keywords=("vision" "multimodal" "VLM" "vision-language" "image" "visual")

        for keyword in "${vision_keywords[@]}"; do
            if echo "$readme_content" | grep -qi "\b$keyword\b"; then
                write_colored_output "  Found in README: $keyword" "green"
                return 0
            fi
        done
    fi

    local preprocessor_path="$model_path/preprocessor_config.json"
    if [[ -f "$preprocessor_path" ]]; then
        write_colored_output "  Found preprocessor_config.json" "green"
        return 0
    fi

    write_colored_output "  Model does not appear to be multimodal" "gray"
    return 1
}

get_original_model_card() {
    local model_path="$1"
    local readme_path="$model_path/README.md"

    if [[ -f "$readme_path" ]]; then
        cat "$readme_path"
    fi
}

push_single_file() {
    local output_dir="$1"
    local file_path="$2"
    local full_repo_name="$3"
    local commit_message="$4"
    local delete_after_push="${5:-true}"

    local filename
    filename=$(basename "$file_path")

    write_colored_output "    Uploading: $filename" "yellow"

    if hf upload "$full_repo_name" "$file_path" "$filename" --commit-message "$commit_message"; then
        write_colored_output "    Uploaded: $filename" "green"

        if [[ "$delete_after_push" == "true" ]]; then
            rm -f "$file_path"
            write_colored_output "    Deleted local copy to save space" "gray"
        fi
        return 0
    else
        write_colored_output "    Upload failed for: $filename (exit code: $?)" "red"
        return 1
    fi
}

# Parse command line arguments
usage() {
    echo "Usage: $0 --hf-repo <repo> [options]"
    echo ""
    echo "Options:"
    echo "  --hf-repo <repo>         HuggingFace repository to quantize (required)"
    echo "  --hf-username <user>     Your HuggingFace username for uploading"
    echo "  --hf-token <token>       HuggingFace token for authentication"
    echo "  --models-dir <dir>       Directory to store downloaded models (default: models)"
    echo "  --build-dir <dir>        Build output dir relative to llama.cpp (default: build/bin)"
    echo "  --skip-build             Skip cloning/pulling and rebuilding llama.cpp"
    echo "  --force-rebuild          Delete the build directory and force a clean rebuild"
    echo "  --skip-venv              Skip recreating the Python virtual environment"
    echo "  --skip-upload            Skip uploading to HuggingFace"
    echo "  --imatrix <file>         Path to importance matrix file (optional)"
    echo "  --threads <n>            Number of threads to use (default: $(nproc))"
    echo "  --keep-files             Keep local files after successful upload"
    echo "  --llama-cpp-url <url>    URL of llama.cpp repo (default: https://github.com/ggml-org/llama.cpp.git)"
    echo "  --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --hf-repo meta-llama/Llama-3.2-1B-Instruct --hf-username your-username"
    echo "  $0 --hf-repo meta-llama/Llama-3.2-1B-Instruct --hf-username your-username --hf-token hf_xxxxxxxxxxxx"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-repo)
            HF_REPO="$2"
            shift 2
            ;;
        --hf-username)
            HF_USERNAME="$2"
            shift 2
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --skip-venv)
            SKIP_VENV=true
            shift
            ;;
        --skip-upload)
            SKIP_UPLOAD=true
            shift
            ;;
        --imatrix)
            IMATRIX="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --keep-files)
            KEEP_FILES=true
            shift
            ;;
        --llama-cpp-url)
            LLAMA_CPP_URL="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$HF_REPO" ]]; then
    echo "Error: --hf-repo is required"
    usage
    exit 1
fi

# Get script directory
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP_DIR="$SCRIPT_ROOT/llama.cpp"

echo ""
write_colored_output "  llama.cpp End-to-End Quantization Script" "magenta"
write_colored_output "  =========================================" "magenta"
echo ""
write_colored_output "Repository: $HF_REPO" "cyan"
write_colored_output "HF Username: ${HF_USERNAME:-'(not set - upload will be skipped)'}" "cyan"
write_colored_output "Models Directory: $MODELS_DIR" "cyan"
write_colored_output "Build Directory: $BUILD_DIR" "cyan"
write_colored_output "Threads: $THREADS" "cyan"
write_colored_output "llama.cpp directory: $LLAMA_CPP_DIR" "cyan"
if [[ -n "$HF_TOKEN" ]]; then
    write_colored_output "HF Token: (provided)" "cyan"
else
    write_colored_output "HF Token: (not provided)" "cyan"
fi

MODEL_NAME=$(get_model_name_from_repo "$HF_REPO")
OUTPUT_REPO_NAME="$MODEL_NAME-GGUF"
write_colored_output "Output Repository: $OUTPUT_REPO_NAME" "cyan"

# Step 1: Clone or update llama.cpp, then rebuild
if [[ "$SKIP_BUILD" != "true" ]]; then
    write_step "Step 1: Cloning/updating llama.cpp and rebuilding"

    if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
        write_colored_output "llama.cpp not found, cloning from $LLAMA_CPP_URL ..." "yellow"
        git clone "$LLAMA_CPP_URL" "$LLAMA_CPP_DIR"
        write_colored_output "llama.cpp cloned successfully!" "green"
        OLD_COMMIT=""
        NEW_COMMIT=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD 2>/dev/null || echo "new")
    else
        OLD_COMMIT=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
        write_colored_output "Current llama.cpp commit: $OLD_COMMIT" "gray"

        write_colored_output "Pulling latest llama.cpp changes..." "yellow"
        if ! git -C "$LLAMA_CPP_DIR" pull; then
            write_colored_output "Warning: git pull failed, continuing with current version" "yellow"
        fi

        NEW_COMMIT=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
        write_colored_output "New llama.cpp commit: $NEW_COMMIT" "gray"
    fi

    # Force rebuild: delete build directory inside llama.cpp
    if [[ "$FORCE_REBUILD" == "true" ]]; then
        local build_path="$LLAMA_CPP_DIR/build"
        if [[ -d "$build_path" ]]; then
            write_colored_output "Force rebuild requested, deleting build directory..." "yellow"
            rm -rf "$build_path"
            write_colored_output "Build directory deleted" "green"
        fi
    fi

    QUANTIZE_PATH="$LLAMA_CPP_DIR/$BUILD_DIR/llama-quantize"
    if [[ "$FORCE_REBUILD" == "true" ]] || [[ "${OLD_COMMIT:-}" != "$NEW_COMMIT" ]] || [[ ! -f "$QUANTIZE_PATH" ]]; then
        if [[ "$FORCE_REBUILD" == "true" ]]; then
            write_colored_output "Force rebuild requested, rebuilding..." "yellow"
        else
            write_colored_output "Changes detected or build missing, rebuilding..." "yellow"
        fi

        # Configure and build from llama.cpp directory
        write_colored_output "Configuring CMake..." "yellow"
        if ! cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build"; then
            write_colored_output "CMake configuration failed!" "red"
            exit 1
        fi

        write_colored_output "Building llama.cpp..." "yellow"
        if ! cmake --build "$LLAMA_CPP_DIR/build" --config Release -j "$THREADS"; then
            write_colored_output "Build failed!" "red"
            exit 1
        fi

        write_colored_output "Build completed successfully!" "green"
    else
        write_colored_output "No changes detected, skipping rebuild" "green"
    fi
else
    write_step "Step 1: Skipping build (--skip-build)"

    # Still ensure llama.cpp exists for convert script and requirements
    if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
        write_colored_output "llama.cpp not found at $LLAMA_CPP_DIR, cloning..." "yellow"
        git clone "$LLAMA_CPP_URL" "$LLAMA_CPP_DIR"
        write_colored_output "llama.cpp cloned!" "green"
    fi
fi

# Step 2: Setup Python venv
if [[ "$SKIP_VENV" != "true" ]]; then
    write_step "Step 2: Setting up Python virtual environment"

    VENV_PATH="$SCRIPT_ROOT/venv"

    if [[ -d "$VENV_PATH" ]]; then
        write_colored_output "Removing existing venv..." "yellow"
        rm -rf "$VENV_PATH"
    fi

    write_colored_output "Creating new virtual environment..." "yellow"
    if ! python3 -m venv "$VENV_PATH"; then
        write_colored_output "Failed to create venv!" "red"
        exit 1
    fi

    write_colored_output "Activating virtual environment..." "yellow"
    source "$VENV_PATH/bin/activate"

    write_colored_output "Installing requirements from llama.cpp..." "yellow"
    pip install --upgrade pip
    if ! pip install -r "$LLAMA_CPP_DIR/requirements.txt"; then
        write_colored_output "Failed to install requirements!" "red"
        exit 1
    fi

    pip install huggingface_hub

    write_colored_output "Python environment setup complete!" "green"
else
    write_step "Step 2: Skipping venv setup (--skip-venv)"

    VENV_PATH="$SCRIPT_ROOT/venv"
    if [[ -f "$VENV_PATH/bin/activate" ]]; then
        source "$VENV_PATH/bin/activate"
    fi
fi

# Step 3: Clone model from HuggingFace
write_step "Step 3: Cloning model from HuggingFace"

MODELS_PATH="$SCRIPT_ROOT/$MODELS_DIR"
mkdir -p "$MODELS_PATH"

MODEL_PATH="$MODELS_PATH/$MODEL_NAME"

if [[ -d "$MODEL_PATH" ]]; then
    write_colored_output "Removing existing model directory..." "yellow"
    rm -rf "$MODEL_PATH"
fi

write_colored_output "Cloning $HF_REPO..." "yellow"

if [[ -n "$HF_TOKEN" ]]; then
    HF_URL="https://hf_user:$HF_TOKEN@huggingface.co/$HF_REPO"
    write_colored_output "Using provided HF token for clone authentication" "yellow"
else
    HF_URL="https://huggingface.co/$HF_REPO"
fi

if command -v git-lfs >/dev/null 2>&1; then
    write_colored_output "Found git-lfs, will use for large file support" "cyan"
    git lfs install 2>/dev/null || true
else
    write_colored_output "Warning: git-lfs not found, may have issues with large files" "yellow"
fi

cd "$MODELS_PATH"
if ! git clone "$HF_URL"; then
    write_colored_output "Failed to clone repository!" "red"
    exit 1
fi
cd "$SCRIPT_ROOT"

write_colored_output "Model cloned successfully!" "green"

ORIGINAL_MODEL_CARD=$(get_original_model_card "$MODEL_PATH")

# Step 4: Check for multimodal/vision model
write_step "Step 4: Detecting model type"

IS_MULTIMODAL=false
if test_is_multimodal_model "$MODEL_PATH"; then
    IS_MULTIMODAL=true
    write_colored_output "Model is MULTIMODAL - will create mmproj" "magenta"
else
    write_colored_output "Model is TEXT-ONLY" "green"
fi

# Step 5: Convert to GGUF
write_step "Step 5: Converting to GGUF format"

CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
if [[ ! -f "$CONVERT_SCRIPT" ]]; then
    write_colored_output "Error: convert_hf_to_gguf.py not found at: $CONVERT_SCRIPT" "red"
    exit 1
fi

GGUF_OUTPUT_DIR="$MODELS_PATH/$MODEL_NAME-gguf"
mkdir -p "$GGUF_OUTPUT_DIR"

write_colored_output "Converting main model to GGUF (BF16)..." "yellow"
if ! python "$CONVERT_SCRIPT" "$MODEL_PATH" --outfile "$GGUF_OUTPUT_DIR/" --outtype bf16; then
    write_colored_output "Failed to convert main model!" "red"
    exit 1
fi

BASE_GGUF=$(find "$GGUF_OUTPUT_DIR" -name "*.gguf" -not -name "*mmproj*" | head -n1)
if [[ -z "$BASE_GGUF" ]]; then
    write_colored_output "Could not find generated GGUF file!" "red"
    exit 1
fi
write_colored_output "Base model: $(basename "$BASE_GGUF")" "green"

MMPROJ_FILE=""
if [[ "$IS_MULTIMODAL" == "true" ]]; then
    write_colored_output "Converting mmproj for vision support..." "yellow"
    if python "$CONVERT_SCRIPT" "$MODEL_PATH" --outfile "$GGUF_OUTPUT_DIR/" --mmproj --outtype f16; then
        MMPROJ_FILE=$(find "$GGUF_OUTPUT_DIR" -name "*mmproj*.gguf" | head -n1)
        if [[ -n "$MMPROJ_FILE" ]]; then
            write_colored_output "mmproj created: $(basename "$MMPROJ_FILE")" "green"
        fi
    else
        write_colored_output "Warning: mmproj conversion failed - model may not support it" "yellow"
    fi
fi

write_colored_output "GGUF conversion complete!" "green"

# Step 6: Clean up original model files
write_step "Step 6: Cleaning up original model files"

delete_patterns=("*.safetensors" "*.bin" "*.pt" "*.pth" "*.ot" "model*.safetensors*")

deleted_size=0
deleted_count=0

for pattern in "${delete_patterns[@]}"; do
    while IFS= read -r -d '' file; do
        if [[ -f "$file" ]]; then
            file_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
            deleted_size=$((deleted_size + file_size))
            deleted_count=$((deleted_count + 1))
            write_colored_output "  Deleting: $(basename "$file")" "gray"
            rm -f "$file"
        fi
    done < <(find "$MODEL_PATH" -name "$pattern" -type f -print0 2>/dev/null)
done

deleted_size_gb=$(echo "scale=2; $deleted_size / 1073741824" | bc -l 2>/dev/null || echo "0")
write_colored_output "Deleted $deleted_count files (${deleted_size_gb} GB)" "green"

# Step 6.5: Setup HuggingFace repository
HF_REPO_READY=false
FULL_REPO_NAME=""

if [[ "$SKIP_UPLOAD" != "true" ]] && [[ -n "$HF_USERNAME" ]]; then
    write_step "Step 6.5: Setting up HuggingFace repository"

    FULL_REPO_NAME="$HF_USERNAME/$OUTPUT_REPO_NAME"
    write_colored_output "Preparing repository: $FULL_REPO_NAME" "yellow"
    write_colored_output "Checking HuggingFace authentication..." "yellow"

    IS_AUTHENTICATED=false

    if [[ -n "$HF_TOKEN" ]]; then
        export HF_TOKEN="$HF_TOKEN"
        write_colored_output "Using provided HF token for authentication" "green"
        IS_AUTHENTICATED=true
    else
        if hf auth whoami >/dev/null 2>&1; then
            hf_whoami=$(hf auth whoami 2>&1)
            write_colored_output "Authenticated as: $hf_whoami" "green"
            IS_AUTHENTICATED=true
        fi
    fi

    if [[ "$IS_AUTHENTICATED" != "true" ]]; then
        write_colored_output "Please login to HuggingFace first:" "yellow"
        write_colored_output "  hf auth login" "cyan"
        write_colored_output "Or provide a token with --hf-token parameter" "cyan"
        write_colored_output "Will continue without uploading." "yellow"
    else
        write_colored_output "Creating repository if needed..." "yellow"
        hf repo create "$OUTPUT_REPO_NAME" --repo-type model --exist-ok >/dev/null 2>&1 || true
        write_colored_output "Repository ready for incremental uploads" "green"
        HF_REPO_READY=true
    fi
else
    if [[ -z "$HF_USERNAME" ]]; then
        write_colored_output "No HuggingFace username - skipping upload setup" "gray"
    fi
    if [[ "$SKIP_UPLOAD" == "true" ]]; then
        write_colored_output "Upload disabled - skipping upload setup" "gray"
    fi
fi

# Track files for README generation
declare -a TRACKED_FILES
TRACKED_FILES+=("$(basename "$BASE_GGUF"):$(stat -c%s "$BASE_GGUF" 2>/dev/null || stat -f%z "$BASE_GGUF" 2>/dev/null):$(echo "$(basename "$BASE_GGUF")" | grep -o 'BF16\|F16' || echo 'F16')")

if [[ -n "$MMPROJ_FILE" ]]; then
    TRACKED_FILES+=("$(basename "$MMPROJ_FILE"):$(stat -c%s "$MMPROJ_FILE" 2>/dev/null || stat -f%z "$MMPROJ_FILE" 2>/dev/null):Vision Projector")

    if [[ "$HF_REPO_READY" == "true" ]]; then
        write_colored_output "Uploading mmproj file..." "yellow"
        if push_single_file "$GGUF_OUTPUT_DIR" "$MMPROJ_FILE" "$FULL_REPO_NAME" "Add vision projector (mmproj)" "true"; then
            write_colored_output "mmproj uploaded and local copy deleted" "green"
        fi
    fi
fi

# Step 7: Create all quantizations
write_step "Step 7: Creating all quantizations"

QUANTIZE_PATH="$LLAMA_CPP_DIR/$BUILD_DIR/llama-quantize"
if [[ ! -f "$QUANTIZE_PATH" ]]; then
    write_colored_output "Error: llama-quantize not found at: $QUANTIZE_PATH" "red"
    exit 1
fi

success_count=0
skipped_count=0
failed_count=0
uploaded_count=0
total_count=${#QUANT_TYPES[@]}

if [[ -n "$IMATRIX" ]] && [[ ! -f "$IMATRIX" ]]; then
    write_colored_output "Warning: Importance matrix not found at: $IMATRIX" "yellow"
    IMATRIX=""
fi

for i in "${!QUANT_TYPES[@]}"; do
    quant_type="${QUANT_TYPES[i]}"
    base_model_name=$(basename "$BASE_GGUF" .gguf)
    base_model_name=$(echo "$base_model_name" | sed 's/-BF16$\|-F16$\|-F32$//')
    output_model="$GGUF_OUTPUT_DIR/$base_model_name-$quant_type.gguf"

    write_colored_output "Processing $quant_type ($((i + 1))/$total_count)..." "yellow"

    if [[ -f "$output_model" ]]; then
        write_colored_output "  Already exists: $(basename "$output_model")" "gray"
        skipped_count=$((skipped_count + 1))
        continue
    fi

    requires_imatrix=false
    for req_type in "${REQUIRES_IMATRIX[@]}"; do
        if [[ "$quant_type" == "$req_type" ]]; then
            requires_imatrix=true
            break
        fi
    done

    if [[ "$requires_imatrix" == "true" ]] && [[ -z "$IMATRIX" ]]; then
        write_colored_output "  Skipped: $quant_type requires importance matrix" "yellow"
        skipped_count=$((skipped_count + 1))
        continue
    fi

    quant_args=()
    if [[ -n "$IMATRIX" ]]; then
        quant_args+=(--imatrix "$IMATRIX")
    fi
    quant_args+=("$BASE_GGUF" "$output_model" "$quant_type" "$THREADS")

    start_time=$(date +%s)

    if "$QUANTIZE_PATH" "${quant_args[@]}"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))

        if [[ -f "$output_model" ]]; then
            file_size=$(stat -c%s "$output_model" 2>/dev/null || stat -f%z "$output_model" 2>/dev/null || echo 0)
            size_mb=$(echo "scale=2; $file_size / 1048576" | bc -l 2>/dev/null || echo "0")
            write_colored_output "  Success: $quant_type (${size_mb}MB, ${duration}s)" "green"
            success_count=$((success_count + 1))

            TRACKED_FILES+=("$(basename "$output_model"):$file_size:$quant_type")

            if [[ "$HF_REPO_READY" == "true" ]]; then
                if push_single_file "$GGUF_OUTPUT_DIR" "$output_model" "$FULL_REPO_NAME" "Add $quant_type quantization" "true"; then
                    uploaded_count=$((uploaded_count + 1))
                else
                    write_colored_output "  Warning: Upload failed, keeping local file" "yellow"
                fi
            fi
        else
            write_colored_output "  Failed: $quant_type (output file not created)" "red"
            failed_count=$((failed_count + 1))
        fi
    else
        write_colored_output "  Failed: $quant_type (exit code: $?)" "red"
        failed_count=$((failed_count + 1))
    fi
done

echo ""
write_colored_output "Quantization Summary:" "cyan"
write_colored_output "  Successful: $success_count" "green"
write_colored_output "  Skipped: $skipped_count" "yellow"
write_colored_output "  Failed: $failed_count" "red"
if [[ "$HF_REPO_READY" == "true" ]]; then
    write_colored_output "  Uploaded: $uploaded_count" "cyan"
fi

# Step 8: Create README
write_step "Step 8: Creating README for quantized repository"

IFS=$'\n' SORTED_TRACKED_FILES=($(printf '%s\n' "${TRACKED_FILES[@]}" | sort -t: -k2 -n))

cat > "$GGUF_OUTPUT_DIR/README.md" << EOF
---
license: other
license_name: same-as-original
license_link: https://huggingface.co/$HF_REPO
base_model: $HF_REPO
tags:
- gguf
- quantized
- llama.cpp
---

# $OUTPUT_REPO_NAME

This repository contains GGUF quantizations of [$HF_REPO](https://huggingface.co/$HF_REPO).

## Quantizations

| Filename | Size | Description |
|----------|------|-------------|
EOF

for file_info in "${SORTED_TRACKED_FILES[@]}"; do
    IFS=':' read -r filename filesize quanttype <<< "$file_info"
    size_mb=$(echo "scale=2; $filesize / 1048576" | bc -l 2>/dev/null || echo "0")
    size_gb=$(echo "scale=2; $filesize / 1073741824" | bc -l 2>/dev/null || echo "0")

    if (( $(echo "$size_gb >= 1" | bc -l 2>/dev/null || echo 0) )); then
        size_str="${size_gb} GB"
    else
        size_str="${size_mb} MB"
    fi

    echo "| $filename | $size_str | $quanttype |" >> "$GGUF_OUTPUT_DIR/README.md"
done

cat >> "$GGUF_OUTPUT_DIR/README.md" << EOF

## Usage

### With llama.cpp

\`\`\`bash
hf download $HF_USERNAME/$OUTPUT_REPO_NAME <filename> --local-dir .
./llama-cli -m <filename> -p "Your prompt here"
\`\`\`

### With llama-cpp-python

\`\`\`python
from llama_cpp import Llama

llm = Llama(model_path="<filename>")
output = llm("Your prompt here", max_tokens=256)
print(output["choices"][0]["text"])
\`\`\`
EOF

if [[ -n "$MMPROJ_FILE" ]]; then
    cat >> "$GGUF_OUTPUT_DIR/README.md" << EOF

## Vision Support

\`\`\`bash
./llama-mtmd-cli -m <model-file> --mmproj $(basename "$MMPROJ_FILE") --image <image-path> -p "Describe this image"
\`\`\`
EOF
fi

cat >> "$GGUF_OUTPUT_DIR/README.md" << EOF

## Quantization Details

These quantizations were created using [llama.cpp](https://github.com/ggml-org/llama.cpp).

- **Base Model:** [$HF_REPO](https://huggingface.co/$HF_REPO)
- **Quantization Tool:** llama.cpp convert_hf_to_gguf.py + llama-quantize

## Original Model Card

---

$ORIGINAL_MODEL_CARD
EOF

write_colored_output "README.md created!" "green"

# Step 9: Finalize upload
UPLOAD_SUCCESSFUL=false

if [[ "$HF_REPO_READY" == "true" ]]; then
    write_step "Step 9: Finalizing HuggingFace upload"
    write_colored_output "Pushing README and base model..." "yellow"

    if push_single_file "$GGUF_OUTPUT_DIR" "$GGUF_OUTPUT_DIR/README.md" "$FULL_REPO_NAME" "Add README" "false"; then
        write_colored_output "README uploaded" "green"
    fi

    if [[ -f "$BASE_GGUF" ]]; then
        write_colored_output "Uploading base model ($(basename "$BASE_GGUF"))..." "yellow"
        base_type=$(echo "$(basename "$BASE_GGUF")" | grep -o 'BF16\|F16' || echo 'F16')
        if push_single_file "$GGUF_OUTPUT_DIR" "$BASE_GGUF" "$FULL_REPO_NAME" "Add $base_type base model" "true"; then
            write_colored_output "Base model uploaded and local copy deleted" "green"
            UPLOAD_SUCCESSFUL=true
        else
            write_colored_output "Base model upload failed" "red"
        fi
    else
        UPLOAD_SUCCESSFUL=true
    fi

    if [[ "$UPLOAD_SUCCESSFUL" == "true" ]]; then
        write_colored_output "All uploads complete!" "green"
        write_colored_output "Repository: https://huggingface.co/$FULL_REPO_NAME" "cyan"
    fi

elif [[ "$SKIP_UPLOAD" != "true" ]] && [[ -n "$HF_USERNAME" ]]; then
    write_step "Step 9: Uploading to HuggingFace"
    FULL_REPO_NAME="$HF_USERNAME/$OUTPUT_REPO_NAME"
    write_colored_output "Uploading to: $FULL_REPO_NAME" "yellow"

    IS_AUTHENTICATED=false
    if [[ -n "$HF_TOKEN" ]]; then
        export HF_TOKEN="$HF_TOKEN"
        IS_AUTHENTICATED=true
    else
        if hf auth whoami >/dev/null 2>&1; then
            IS_AUTHENTICATED=true
        fi
    fi

    if [[ "$IS_AUTHENTICATED" != "true" ]]; then
        write_colored_output "Not authenticated, skipping upload." "yellow"
    else
        hf repo create "$OUTPUT_REPO_NAME" --repo-type model --exist-ok >/dev/null 2>&1 || true
        all_files_uploaded=true

        while IFS= read -r -d '' file_to_upload; do
            delete_after=false
            if [[ "$file_to_upload" == *.gguf ]]; then
                delete_after=true
            fi
            if ! push_single_file "$GGUF_OUTPUT_DIR" "$file_to_upload" "$FULL_REPO_NAME" "Add $(basename "$file_to_upload")" "$delete_after"; then
                all_files_uploaded=false
            fi
        done < <(find "$GGUF_OUTPUT_DIR" \( -name "*.gguf" -o -name "README.md" \) -type f -print0)

        if [[ "$all_files_uploaded" == "true" ]]; then
            write_colored_output "Upload complete!" "green"
            write_colored_output "Repository: https://huggingface.co/$FULL_REPO_NAME" "cyan"
            UPLOAD_SUCCESSFUL=true
        else
            write_colored_output "Some uploads failed, please check manually" "yellow"
        fi
    fi
else
    write_step "Step 9: Skipping upload"
fi

# Step 10: Cleanup
if [[ "$UPLOAD_SUCCESSFUL" == "true" ]] && [[ "$KEEP_FILES" != "true" ]]; then
    write_step "Step 10: Cleaning up local files after successful upload"

    if [[ -d "$MODEL_PATH" ]]; then
        write_colored_output "Deleting original model folder: $MODEL_PATH" "yellow"
        rm -rf "$MODEL_PATH" && write_colored_output "  Deleted successfully" "green" || write_colored_output "  Warning: Could not fully delete folder" "yellow"
    fi

    if [[ -d "$GGUF_OUTPUT_DIR" ]]; then
        write_colored_output "Deleting GGUF output folder: $GGUF_OUTPUT_DIR" "yellow"
        rm -rf "$GGUF_OUTPUT_DIR" && write_colored_output "  Deleted successfully" "green" || write_colored_output "  Warning: Could not fully delete folder" "yellow"
    fi

elif [[ "$UPLOAD_SUCCESSFUL" == "true" ]] && [[ "$KEEP_FILES" == "true" ]]; then
    write_step "Step 10: Keeping local files (--keep-files specified)"
    write_colored_output "Files retained at: $GGUF_OUTPUT_DIR" "cyan"
fi

# Final Summary
echo ""
write_colored_output "============================================================" "green"
write_colored_output "  QUANTIZATION COMPLETE!" "green"
write_colored_output "============================================================" "green"
echo ""
write_colored_output "Source Model: $HF_REPO" "cyan"
write_colored_output "Multimodal: $(if [[ "$IS_MULTIMODAL" == "true" ]]; then echo "Yes"; else echo "No"; fi)" "cyan"

if [[ "$UPLOAD_SUCCESSFUL" == "true" ]]; then
    echo ""
    write_colored_output "HuggingFace Repository: https://huggingface.co/$HF_USERNAME/$OUTPUT_REPO_NAME" "cyan"
    if [[ "$KEEP_FILES" != "true" ]]; then
        write_colored_output "Local files: Cleaned up to conserve space" "green"
    else
        write_colored_output "Local files: Retained at $GGUF_OUTPUT_DIR" "cyan"
    fi
else
    write_colored_output "Output Directory: $GGUF_OUTPUT_DIR" "cyan"
    if [[ -d "$GGUF_OUTPUT_DIR" ]]; then
        echo ""
        write_colored_output "Generated Files:" "cyan"
        find "$GGUF_OUTPUT_DIR" -name "*.gguf" -type f -exec stat -c "%s %n" {} + 2>/dev/null | sort -n | while read -r size filepath; do
            size_gb=$(echo "scale=2; $size / 1073741824" | bc -l 2>/dev/null || echo "0")
            write_colored_output "  $(basename "$filepath") (${size_gb} GB)" "white"
        done
    fi
fi

echo ""
