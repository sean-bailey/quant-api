from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.quant_service import (
    QUANT_TYPES,
    REQUIRES_IMATRIX,
    ConversionError,
    QuantizationError,
    clean_source_files,
    convert_to_gguf,
    create_mmproj,
    is_multimodal,
    quantize_single,
)


# ---------------------------------------------------------------------------
# QUANT_TYPES constant
# ---------------------------------------------------------------------------

def test_quant_types_has_31_entries():
    assert len(QUANT_TYPES) == 31


def test_quant_types_contains_standard_types():
    for t in ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "F16"]:
        assert t in QUANT_TYPES, f"{t} missing from QUANT_TYPES"


def test_quant_types_contains_k_quant_types():
    for t in ["Q2_K", "Q2_K_S", "Q3_K_S", "Q3_K_M", "Q3_K_L",
              "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K"]:
        assert t in QUANT_TYPES, f"{t} missing from QUANT_TYPES"


def test_quant_types_contains_iq_types():
    for t in ["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
              "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_NL", "IQ4_XS"]:
        assert t in QUANT_TYPES, f"{t} missing from QUANT_TYPES"


def test_quant_types_contains_ternary_types():
    assert "TQ1_0" in QUANT_TYPES
    assert "TQ2_0" in QUANT_TYPES


def test_quant_types_contains_bf16():
    assert "BF16" in QUANT_TYPES


def test_quant_types_has_no_duplicates():
    assert len(QUANT_TYPES) == len(set(QUANT_TYPES))


# ---------------------------------------------------------------------------
# REQUIRES_IMATRIX constant
# ---------------------------------------------------------------------------

def test_requires_imatrix_contains_all_expected_types():
    for t in ["IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "Q2_K_S"]:
        assert t in REQUIRES_IMATRIX, f"{t} missing from REQUIRES_IMATRIX"


def test_requires_imatrix_does_not_contain_common_types():
    for t in ["Q4_K_M", "Q5_K_M", "Q8_0", "F16", "BF16", "IQ2_M"]:
        assert t not in REQUIRES_IMATRIX, f"{t} should not require imatrix"


def test_requires_imatrix_is_subset_of_quant_types():
    assert REQUIRES_IMATRIX.issubset(set(QUANT_TYPES))


def test_requires_imatrix_has_no_extra_entries():
    assert len(REQUIRES_IMATRIX) == 6


# ---------------------------------------------------------------------------
# is_multimodal — config.json keys
# ---------------------------------------------------------------------------

def test_is_multimodal_true_for_vision_config_key(tmp_path):
    (tmp_path / "config.json").write_text('{"vision_config": {}}')
    assert is_multimodal(tmp_path) is True


def test_is_multimodal_true_for_visual_config_key(tmp_path):
    (tmp_path / "config.json").write_text('{"visual_config": {}}')
    assert is_multimodal(tmp_path) is True


def test_is_multimodal_true_for_image_encoder_key(tmp_path):
    (tmp_path / "config.json").write_text('{"image_encoder": {}}')
    assert is_multimodal(tmp_path) is True


def test_is_multimodal_true_for_vision_tower_key(tmp_path):
    (tmp_path / "config.json").write_text('{"vision_tower": "clip-vit-large"}')
    assert is_multimodal(tmp_path) is True


def test_is_multimodal_true_for_visual_encoder_key(tmp_path):
    (tmp_path / "config.json").write_text('{"visual_encoder": {}}')
    assert is_multimodal(tmp_path) is True


def test_is_multimodal_false_for_plain_llama_config(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "llama", "hidden_size": 4096}')
    assert is_multimodal(tmp_path) is False


# ---------------------------------------------------------------------------
# is_multimodal — preprocessor_config.json
# ---------------------------------------------------------------------------

def test_is_multimodal_true_when_preprocessor_config_exists(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "llama"}')
    (tmp_path / "preprocessor_config.json").touch()
    assert is_multimodal(tmp_path) is True


# ---------------------------------------------------------------------------
# is_multimodal — README.md keywords
# ---------------------------------------------------------------------------

def test_is_multimodal_true_for_vision_keyword_in_readme(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "llama"}')
    (tmp_path / "README.md").write_text("This is a vision language model.")
    assert is_multimodal(tmp_path) is True


def test_is_multimodal_true_for_multimodal_keyword_in_readme(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "llama"}')
    (tmp_path / "README.md").write_text("A powerful multimodal architecture.")
    assert is_multimodal(tmp_path) is True


def test_is_multimodal_true_for_vlm_keyword_in_readme(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "llama"}')
    (tmp_path / "README.md").write_text("State-of-the-art VLM for image understanding.")
    assert is_multimodal(tmp_path) is True


def test_is_multimodal_readme_keyword_check_is_case_insensitive(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "llama"}')
    (tmp_path / "README.md").write_text("VISION model for classification.")
    assert is_multimodal(tmp_path) is True


# ---------------------------------------------------------------------------
# is_multimodal — edge cases
# ---------------------------------------------------------------------------

def test_is_multimodal_false_when_no_config_json(tmp_path):
    assert is_multimodal(tmp_path) is False


def test_is_multimodal_false_when_empty_dir(tmp_path):
    assert is_multimodal(tmp_path) is False


def test_is_multimodal_handles_invalid_json_gracefully(tmp_path):
    (tmp_path / "config.json").write_text("not valid json {{{")
    # Must not raise; fall through to False
    assert is_multimodal(tmp_path) is False


def test_is_multimodal_false_when_no_indicators_but_config_exists(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "mistral"}')
    assert is_multimodal(tmp_path) is False


# ---------------------------------------------------------------------------
# convert_to_gguf
# ---------------------------------------------------------------------------

@patch("subprocess.run")
def test_convert_to_gguf_calls_convert_script(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    convert_to_gguf("/usr/bin/python3", "/tmp/llama", str(tmp_path), "/tmp/out.gguf")
    cmd = mock_run.call_args[0][0]
    assert any("convert_hf_to_gguf.py" in str(a) for a in cmd)


@patch("subprocess.run")
def test_convert_to_gguf_uses_bf16_outtype(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    convert_to_gguf("/usr/bin/python3", "/tmp/llama", str(tmp_path), "/tmp/out.gguf")
    cmd = mock_run.call_args[0][0]
    assert "--outtype" in cmd
    assert "bf16" in cmd


@patch("subprocess.run")
def test_convert_to_gguf_passes_outfile(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    convert_to_gguf("/usr/bin/python3", "/tmp/llama", str(tmp_path), "/tmp/out.gguf")
    cmd = mock_run.call_args[0][0]
    assert "--outfile" in cmd
    assert "/tmp/out.gguf" in cmd


@patch("subprocess.run")
def test_convert_to_gguf_passes_model_dir(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    convert_to_gguf("/usr/bin/python3", "/tmp/llama", str(tmp_path), "/tmp/out.gguf")
    cmd = mock_run.call_args[0][0]
    assert str(tmp_path) in cmd


@patch("subprocess.run")
def test_convert_to_gguf_uses_provided_python(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    convert_to_gguf("/custom/python3", "/tmp/llama", str(tmp_path), "/tmp/out.gguf")
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/custom/python3"


@patch("subprocess.run")
def test_convert_to_gguf_nonzero_exit_raises_conversion_error(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=1, stderr=b"conversion failed")
    with pytest.raises(ConversionError):
        convert_to_gguf("/usr/bin/python3", "/tmp/llama", str(tmp_path), "/tmp/out.gguf")


@patch("subprocess.run")
def test_convert_to_gguf_error_message_contains_exit_code(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=2, stderr=b"bad model")
    with pytest.raises(ConversionError, match="2"):
        convert_to_gguf("/usr/bin/python3", "/tmp/llama", str(tmp_path), "/tmp/out.gguf")


# ---------------------------------------------------------------------------
# create_mmproj
# ---------------------------------------------------------------------------

@patch("subprocess.run")
def test_create_mmproj_calls_convert_script(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    create_mmproj("/usr/bin/python3", "/tmp/llama", str(tmp_path), "/tmp/mmproj.gguf")
    cmd = mock_run.call_args[0][0]
    assert any("convert_hf_to_gguf.py" in str(a) for a in cmd)


@patch("subprocess.run")
def test_create_mmproj_includes_mmproj_flag(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    create_mmproj("/usr/bin/python3", "/tmp/llama", str(tmp_path), "/tmp/mmproj.gguf")
    cmd = mock_run.call_args[0][0]
    assert "--mmproj" in cmd


@patch("subprocess.run")
def test_create_mmproj_passes_outfile(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    create_mmproj("/usr/bin/python3", "/tmp/llama", str(tmp_path), "/tmp/mmproj.gguf")
    cmd = mock_run.call_args[0][0]
    assert "--outfile" in cmd
    assert "/tmp/mmproj.gguf" in cmd


@patch("subprocess.run")
def test_create_mmproj_nonzero_exit_raises_conversion_error(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=1, stderr=b"mmproj failed")
    with pytest.raises(ConversionError):
        create_mmproj("/usr/bin/python3", "/tmp/llama", str(tmp_path), "/tmp/mmproj.gguf")


# ---------------------------------------------------------------------------
# quantize_single — happy path
# ---------------------------------------------------------------------------

@patch("subprocess.run")
def test_quantize_single_returns_ok_on_success(mock_run, tmp_path):
    base = tmp_path / "model.gguf"
    base.touch()
    mock_run.return_value = MagicMock(returncode=0)
    result = quantize_single(
        binary="llama-quantize",
        base_model=str(base),
        output=str(tmp_path / "out.gguf"),
        quant_type="Q4_K_M",
        threads=4,
    )
    assert result == "ok"


@patch("subprocess.run")
def test_quantize_single_calls_binary_as_first_arg(mock_run, tmp_path):
    base = tmp_path / "model.gguf"
    base.touch()
    mock_run.return_value = MagicMock(returncode=0)
    quantize_single("/path/to/llama-quantize", str(base), str(tmp_path / "out.gguf"), "Q4_K_M", 4)
    assert mock_run.call_args[0][0][0] == "/path/to/llama-quantize"


@patch("subprocess.run")
def test_quantize_single_passes_quant_type(mock_run, tmp_path):
    base = tmp_path / "model.gguf"
    base.touch()
    mock_run.return_value = MagicMock(returncode=0)
    quantize_single("llama-quantize", str(base), str(tmp_path / "out.gguf"), "Q5_K_M", 4)
    cmd = mock_run.call_args[0][0]
    assert "Q5_K_M" in cmd


@patch("subprocess.run")
def test_quantize_single_passes_thread_count_as_string(mock_run, tmp_path):
    base = tmp_path / "model.gguf"
    base.touch()
    mock_run.return_value = MagicMock(returncode=0)
    quantize_single("llama-quantize", str(base), str(tmp_path / "out.gguf"), "Q4_K_M", 8)
    cmd = mock_run.call_args[0][0]
    assert "8" in cmd


@patch("subprocess.run")
def test_quantize_single_passes_imatrix_flag_when_provided(mock_run, tmp_path):
    base = tmp_path / "model.gguf"
    base.touch()
    imatrix = "/path/to/imatrix.bin"
    mock_run.return_value = MagicMock(returncode=0)
    quantize_single("llama-quantize", str(base), str(tmp_path / "out.gguf"), "IQ2_M", 4, imatrix=imatrix)
    cmd = mock_run.call_args[0][0]
    assert "--imatrix" in cmd
    assert imatrix in cmd


@patch("subprocess.run")
def test_quantize_single_no_imatrix_flag_when_not_provided(mock_run, tmp_path):
    base = tmp_path / "model.gguf"
    base.touch()
    mock_run.return_value = MagicMock(returncode=0)
    quantize_single("llama-quantize", str(base), str(tmp_path / "out.gguf"), "Q4_K_M", 4)
    cmd = mock_run.call_args[0][0]
    assert "--imatrix" not in cmd


@patch("subprocess.run")
def test_quantize_single_imatrix_flag_comes_before_positional_args(mock_run, tmp_path):
    base = tmp_path / "model.gguf"
    base.touch()
    imatrix = "/path/imatrix.bin"
    mock_run.return_value = MagicMock(returncode=0)
    quantize_single("llama-quantize", str(base), str(tmp_path / "out.gguf"), "IQ2_M", 4, imatrix=imatrix)
    cmd = mock_run.call_args[0][0]
    imatrix_idx = cmd.index("--imatrix")
    base_idx = cmd.index(str(base))
    assert imatrix_idx < base_idx


# ---------------------------------------------------------------------------
# quantize_single — skip conditions
# ---------------------------------------------------------------------------

def test_quantize_single_skips_if_output_already_exists(tmp_path):
    base = tmp_path / "model.gguf"
    base.touch()
    out = tmp_path / "already-exists.gguf"
    out.touch()
    result = quantize_single("llama-quantize", str(base), str(out), "Q4_K_M", 4)
    assert result == "skipped"


def test_quantize_single_skips_iq1_s_without_imatrix(tmp_path):
    base = tmp_path / "model.gguf"
    base.touch()
    result = quantize_single(
        "llama-quantize", str(base), str(tmp_path / "out.gguf"), "IQ1_S", 4, imatrix=None
    )
    assert result == "skipped"


def test_quantize_single_skips_all_imatrix_required_types_without_imatrix(tmp_path):
    base = tmp_path / "model.gguf"
    base.touch()
    for qt in REQUIRES_IMATRIX:
        result = quantize_single(
            "llama-quantize", str(base),
            str(tmp_path / "out.gguf"),
            qt, 4, imatrix=None,
        )
        assert result == "skipped", f"{qt} should be skipped without imatrix"


def test_quantize_single_does_not_skip_imatrix_type_when_imatrix_provided(tmp_path):
    base = tmp_path / "model.gguf"
    base.touch()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = quantize_single(
            "llama-quantize", str(base),
            str(tmp_path / "out.gguf"),
            "IQ1_S", 4, imatrix="/path/imatrix.bin",
        )
    assert result == "ok"


# ---------------------------------------------------------------------------
# quantize_single — failure path
# ---------------------------------------------------------------------------

@patch("subprocess.run")
def test_quantize_single_returns_failed_on_nonzero_exit(mock_run, tmp_path):
    base = tmp_path / "model.gguf"
    base.touch()
    mock_run.return_value = MagicMock(returncode=1, stderr=b"quantize error")
    result = quantize_single("llama-quantize", str(base), str(tmp_path / "out.gguf"), "Q4_K_M", 4)
    assert result == "failed"


@patch("subprocess.run")
def test_quantize_single_does_not_raise_on_binary_failure(mock_run, tmp_path):
    base = tmp_path / "model.gguf"
    base.touch()
    mock_run.return_value = MagicMock(returncode=2, stderr=b"segfault")
    # Must return "failed" and NOT raise
    result = quantize_single("llama-quantize", str(base), str(tmp_path / "out.gguf"), "Q4_K_M", 4)
    assert result == "failed"


# ---------------------------------------------------------------------------
# clean_source_files
# ---------------------------------------------------------------------------

def test_clean_removes_safetensors(tmp_path):
    (tmp_path / "model.safetensors").touch()
    clean_source_files(tmp_path)
    assert not (tmp_path / "model.safetensors").exists()


def test_clean_removes_sharded_safetensors(tmp_path):
    (tmp_path / "model-00001-of-00003.safetensors").touch()
    (tmp_path / "model-00002-of-00003.safetensors").touch()
    (tmp_path / "model-00003-of-00003.safetensors").touch()
    clean_source_files(tmp_path)
    assert not any(tmp_path.glob("*.safetensors"))


def test_clean_removes_bin_files(tmp_path):
    (tmp_path / "pytorch_model.bin").touch()
    clean_source_files(tmp_path)
    assert not (tmp_path / "pytorch_model.bin").exists()


def test_clean_removes_pt_files(tmp_path):
    (tmp_path / "model.pt").touch()
    clean_source_files(tmp_path)
    assert not (tmp_path / "model.pt").exists()


def test_clean_removes_pth_files(tmp_path):
    (tmp_path / "model.pth").touch()
    clean_source_files(tmp_path)
    assert not (tmp_path / "model.pth").exists()


def test_clean_preserves_config_json(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "config.json").touch()
    clean_source_files(tmp_path)
    assert (tmp_path / "config.json").exists()


def test_clean_preserves_readme(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "README.md").touch()
    clean_source_files(tmp_path)
    assert (tmp_path / "README.md").exists()


def test_clean_preserves_tokenizer_files(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "tokenizer.json").touch()
    (tmp_path / "tokenizer_config.json").touch()
    (tmp_path / "special_tokens_map.json").touch()
    clean_source_files(tmp_path)
    assert (tmp_path / "tokenizer.json").exists()
    assert (tmp_path / "tokenizer_config.json").exists()
    assert (tmp_path / "special_tokens_map.json").exists()


def test_clean_preserves_gguf_files(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "model-bf16.gguf").touch()
    clean_source_files(tmp_path)
    assert (tmp_path / "model-bf16.gguf").exists()


def test_clean_returns_bytes_freed(tmp_path):
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"x" * 1024)
    freed = clean_source_files(tmp_path)
    assert freed == 1024


def test_clean_returns_total_bytes_across_multiple_files(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "b.bin").write_bytes(b"y" * 200)
    (tmp_path / "c.pt").write_bytes(b"z" * 50)
    freed = clean_source_files(tmp_path)
    assert freed == 350


def test_clean_returns_zero_when_nothing_to_remove(tmp_path):
    (tmp_path / "config.json").touch()
    freed = clean_source_files(tmp_path)
    assert freed == 0


def test_clean_does_not_recurse_into_subdirs(tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    nested = subdir / "nested.safetensors"
    nested.touch()
    clean_source_files(tmp_path)
    # nested file in subdir should be untouched
    assert nested.exists()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

def test_conversion_error_is_exception():
    assert issubclass(ConversionError, Exception)


def test_quantization_error_is_exception():
    assert issubclass(QuantizationError, Exception)


def test_conversion_error_carries_message():
    err = ConversionError("bf16 step failed")
    assert "bf16 step failed" in str(err)
