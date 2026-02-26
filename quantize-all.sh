#!/bin/bash

# Shell script to quantize a model with llama.cpp in all available quantizations
# Automatically clones llama.cpp if not present, or pulls latest if it is.
# Usage: ./quantize-all.sh --model-folder "path/to/model/folder" [options]

set -euo pipefail

MODEL_FOLDER=""
BUILD_DIR="build/bin"
IMATRIX=""
THREADS=$(nproc)
SKIP_BUILD=false
FORCE_REBUILD=false
LLAMA_CPP_URL="https://github.com/ggml-org/llama.cpp.git"

QUANT_TYPES=(
    "Q4_0"      # 4.34G, +0.4685 ppl @ Llama-3-8B
    "Q4_1"      # 4.78G, +0.4511 ppl @ Llama-3-8B
    "Q5_0"      # 5.21G, +0.1316 ppl @ Llama-3-8B
    "Q5_1"      # 5.65G, +0.1062 ppl @ Llama-3-8B
    "IQ2_XXS"   # 2.06 bpw quantization
    "IQ2_XS"    # 2.31 bpw quantization
    "IQ2_S"     # 2.5  bpw quantization
    "IQ2_M"     # 2.7  bpw quantization
    "IQ1_S"     # 1.56 bpw quantization
    "IQ1_M"     # 1.75 bpw quantization
    "TQ1_0"     # 1.69 bpw ternarization
    "TQ2_0"     # 2.06 bpw ternarization
    "Q2_K"      # 2.96G, +3.5199 ppl @ Llama-3-8B
    "Q2_K_S"    # 2.96G, +3.1836 ppl @ Llama-3-8B
    "IQ3_XXS"   # 3.06 bpw quantization
    "IQ3_S"     # 3.44 bpw quantization
    "IQ3_M"     # 3.66 bpw quantization mix
    "Q3_K_S"    # 3.41G, +1.6321 ppl @ Llama-3-8B
    "Q3_K_M"    # 3.74G, +0.6569 ppl @ Llama-3-8B
    "Q3_K_L"    # 4.03G, +0.5562 ppl @ Llama-3-8B
    "IQ3_XS"    # 3.3 bpw quantization
    "IQ4_NL"    # 4.50 bpw non-linear quantization
    "IQ4_XS"    # 4.25 bpw non-linear quantization
    "Q4_K_S"    # 4.37G, +0.2689 ppl @ Llama-3-8B
    "Q4_K_M"    # 4.58G, +0.1754 ppl @ Llama-3-8B
    "Q5_K_S"    # 5.21G, +0.1049 ppl @ Llama-3-8B
    "Q5_K_M"    # 5.33G, +0.0569 ppl @ Llama-3-8B
    "Q6_K"      # 6.14G, +0.0217 ppl @ Llama-3-8B
    "Q8_0"      # 7.96G, +0.0026 ppl @ Llama-3-8B
    "F16"       # 14.00G, +0.0020 ppl @ Mistral-7B
    "BF16"      # 14.00G, -0.0050 ppl @ Mistral-7B
)

REQUIRES_IMATRIX=("IQ1_S" "IQ1_M" "IQ2_S" "IQ2_XXS" "IQ2_XS" "Q2_K_S")

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

test_path_exists() {
    local path="$1"
    local type="$2"
    if [[ ! -e "$path" ]]; then
        write_colored_output "Error: $type not found at: $path" "red"
        exit 1
    fi
}

find_base_model() {
    local model_folder="$1"

    local base_patterns=("*f16*.gguf" "*bf16*.gguf" "*f32*.gguf" "*unquantized*.gguf")
    for pattern in "${base_patterns[@]}"; do
        local found
        found=$(find "$model_folder" -maxdepth 1 -name "$pattern" -type f | head -n1)
        if [[ -n "$found" ]]; then
            echo "$found"
            return 0
        fi
    done

    while IFS= read -r -d '' gguf_file; do
        local name
        name=$(basename "$gguf_file" | tr '[:upper:]' '[:lower:]')
        local has_quant_suffix=false
        for quant in "${QUANT_TYPES[@]}"; do
            local quant_lower
            quant_lower=$(echo "$quant" | tr '[:upper:]' '[:lower:]')
            if [[ "$name" == *"$quant_lower"* ]]; then
                has_quant_suffix=true
                break
            fi
        done
        if [[ "$has_quant_suffix" == "false" ]]; then
            echo "$gguf_file"
            return 0
        fi
    done < <(find "$model_folder" -maxdepth 1 -name "*.gguf" -type f -print0)

    local first_gguf
    first_gguf=$(find "$model_folder" -maxdepth 1 -name "*.gguf" -type f | head -n1)
    if [[ -n "$first_gguf" ]]; then
        write_colored_output "Warning: Could not identify base model, using: $(basename "$first_gguf")" "yellow"
        echo "$first_gguf"
        return 0
    fi

    return 1
}

usage() {
    echo "Usage: $0 --model-folder <folder> [options]"
    echo ""
    echo "Options:"
    echo "  --model-folder <folder>  Path to model folder containing GGUF files (required)"
    echo "  --build-dir <dir>        Build output dir relative to llama.cpp (default: build/bin)"
    echo "  --imatrix <file>         Path to importance matrix file (optional)"
    echo "  --threads <n>            Number of threads (default: $(nproc))"
    echo "  --skip-build             Skip cloning/pulling and rebuilding llama.cpp"
    echo "  --force-rebuild          Delete build dir and force clean rebuild"
    echo "  --llama-cpp-url <url>    URL of llama.cpp repo"
    echo "  --help                   Show this help message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-folder) MODEL_FOLDER="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --imatrix) IMATRIX="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --force-rebuild) FORCE_REBUILD=true; shift ;;
        --llama-cpp-url) LLAMA_CPP_URL="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$MODEL_FOLDER" ]]; then
    echo "Error: --model-folder is required"
    usage
    exit 1
fi

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP_DIR="$SCRIPT_ROOT/llama.cpp"

# Clone or update llama.cpp
if [[ "$SKIP_BUILD" != "true" ]]; then
    if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
        write_colored_output "llama.cpp not found, cloning from $LLAMA_CPP_URL ..." "yellow"
        git clone "$LLAMA_CPP_URL" "$LLAMA_CPP_DIR"
        write_colored_output "llama.cpp cloned!" "green"
        OLD_COMMIT=""
        NEW_COMMIT=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD 2>/dev/null || echo "new")
    else
        OLD_COMMIT=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
        write_colored_output "Pulling latest llama.cpp..." "yellow"
        git -C "$LLAMA_CPP_DIR" pull 2>/dev/null || write_colored_output "Warning: git pull failed" "yellow"
        NEW_COMMIT=$(git -C "$LLAMA_CPP_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
    fi

    if [[ "$FORCE_REBUILD" == "true" ]] && [[ -d "$LLAMA_CPP_DIR/build" ]]; then
        write_colored_output "Force rebuild: deleting build directory..." "yellow"
        rm -rf "$LLAMA_CPP_DIR/build"
    fi

    QUANTIZE_BIN="$LLAMA_CPP_DIR/$BUILD_DIR/llama-quantize"
    if [[ "$FORCE_REBUILD" == "true" ]] || [[ "${OLD_COMMIT:-}" != "$NEW_COMMIT" ]] || [[ ! -f "$QUANTIZE_BIN" ]]; then
        write_colored_output "Building llama.cpp..." "yellow"
        cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build"
        cmake --build "$LLAMA_CPP_DIR/build" --config Release -j "$THREADS"
        write_colored_output "Build complete!" "green"
    else
        write_colored_output "llama.cpp up to date, skipping rebuild." "green"
    fi
fi

# Validate inputs
write_colored_output "Starting llama.cpp quantization for all formats..." "green"

test_path_exists "$MODEL_FOLDER" "Model folder"
QUANTIZE_PATH="$LLAMA_CPP_DIR/$BUILD_DIR/llama-quantize"
test_path_exists "$QUANTIZE_PATH" "llama-quantize"

if [[ -n "$IMATRIX" ]] && [[ ! -f "$IMATRIX" ]]; then
    write_colored_output "Warning: Importance matrix file not found at: $IMATRIX" "yellow"
    IMATRIX=""
fi

BASE_MODEL=$(find_base_model "$MODEL_FOLDER")
if [[ -z "$BASE_MODEL" ]]; then
    write_colored_output "Error: No GGUF files found in $MODEL_FOLDER" "red"
    exit 1
fi

write_colored_output "Base model: $(basename "$BASE_MODEL")" "cyan"
write_colored_output "Build directory: $BUILD_DIR" "cyan"
write_colored_output "Threads: $THREADS" "cyan"
if [[ -n "$IMATRIX" ]]; then
    write_colored_output "Importance matrix: $(basename "$IMATRIX")" "cyan"
else
    write_colored_output "No importance matrix provided (required for some quantizations)" "yellow"
fi

OUTPUT_FOLDER="$MODEL_FOLDER/quantized"
if [[ ! -d "$OUTPUT_FOLDER" ]]; then
    mkdir -p "$OUTPUT_FOLDER"
    write_colored_output "Created output folder: $OUTPUT_FOLDER" "green"
fi

success_count=0
skipped_count=0
failed_count=0
total_count=${#QUANT_TYPES[@]}

write_colored_output "" ""
write_colored_output "Starting quantization process..." "green"
write_colored_output "Total quantization types to process: $total_count" "green"
echo ""

for i in "${!QUANT_TYPES[@]}"; do
    quant_type="${QUANT_TYPES[i]}"
    base_model_name=$(basename "$BASE_MODEL" .gguf)
    output_model="$OUTPUT_FOLDER/$base_model_name-$quant_type.gguf"

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

    args=()
    if [[ -n "$IMATRIX" ]]; then
        args+=(--imatrix "$IMATRIX")
    fi
    args+=("$BASE_MODEL" "$output_model" "$quant_type" "$THREADS")

    start_time=$(date +%s)

    if "$QUANTIZE_PATH" "${args[@]}"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))

        if [[ -f "$output_model" ]]; then
            size_mb=$(echo "scale=2; $(stat -c%s "$output_model" 2>/dev/null || stat -f%z "$output_model" 2>/dev/null) / 1048576" | bc -l 2>/dev/null || echo "0")
            write_colored_output "  Success: $quant_type (${size_mb}MB, ${duration}s)" "green"
            success_count=$((success_count + 1))
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
write_colored_output "==================================================" "cyan"
write_colored_output "QUANTIZATION SUMMARY" "cyan"
write_colored_output "==================================================" "cyan"
write_colored_output "Total quantizations: $total_count" "white"
write_colored_output "Successful: $success_count" "green"
write_colored_output "Skipped: $skipped_count" "yellow"
write_colored_output "Failed: $failed_count" "red"
write_colored_output "Output directory: $OUTPUT_FOLDER" "cyan"

if [[ $success_count -gt 0 ]]; then
    echo ""
    write_colored_output "Quantized models:" "green"
    find "$OUTPUT_FOLDER" -name "*.gguf" -type f -exec stat -c "%s %n" {} + 2>/dev/null | sort -n | while read -r size filepath; do
        size_mb=$(echo "scale=2; $size / 1048576" | bc -l 2>/dev/null || echo "0")
        write_colored_output "  $(basename "$filepath") (${size_mb}MB)" "white"
    done
fi

echo ""
write_colored_output "Quantization complete!" "green"
