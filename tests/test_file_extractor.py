"""
Tests for Nexora Multi-Format File Extractor & 10MB Limit
"""
import io
import zipfile
import pytest
from pathlib import Path
from nexora.backend.app.core.file_extractor import (
    process_file_content,
    MAX_FILE_SIZE_BYTES,
    get_file_category,
    ExtractedFileResult,
)


def test_file_category_detection():
    assert get_file_category("main.py")[0] == "code"
    assert get_file_category("app.ts")[0] == "code"
    assert get_file_category("style.css")[0] == "code"
    assert get_file_category("data.json")[0] == "data"
    assert get_file_category("config.yaml")[0] == "data"
    assert get_file_category("document.pdf")[0] == "document"
    assert get_file_category("report.docx")[0] == "document"
    assert get_file_category("archive.zip")[0] == "archive"
    assert get_file_category("backup.tar.gz")[0] == "archive"
    assert get_file_category("notes.md")[0] == "text"


def test_code_and_text_extraction():
    code = b"def add(a: int, b: int) -> int:\n    return a + b\n"
    res = process_file_content(code, "math_utils.py")
    assert res.category == "code"
    assert res.error is None
    assert "def add(a: int, b: int)" in res.text_content
    assert res.original_size == len(code)


def test_strict_10mb_limit_enforcement():
    # Exactly at or slightly below limit is accepted
    small_data = b"hello" * 100
    res_ok = process_file_content(small_data, "test.txt")
    assert res_ok.error is None

    # Over 10 MB limit must be rejected
    oversized_data = b"x" * (MAX_FILE_SIZE_BYTES + 1024)
    res_fail = process_file_content(oversized_data, "too_large.zip")
    assert res_fail.error is not None
    assert "10 MB" in res_fail.error


def test_zip_archive_safe_extraction(tmp_path: Path):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("src/calc.py", "def multiply(x, y): return x * y\n")
        zf.writestr("tests/test_calc.py", "from src.calc import multiply\nassert multiply(2, 3) == 6\n")
        zf.writestr("docs/README.md", "# Calculator Module\n")

    res = process_file_content(zip_buffer.getvalue(), "calc_project.zip", sandbox_workspace_dir=tmp_path)
    assert res.category == "archive"
    assert res.error is None
    assert len(res.extracted_files) == 3
    assert "src/calc.py" in res.extracted_files
    assert (tmp_path / "extracted_calc_project" / "src" / "calc.py").exists()


def test_zip_archive_path_traversal_protection(tmp_path: Path):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        # Malicious entry attempting to escape
        zf.writestr("../../etc/passwd", "malicious_content\n")
        zf.writestr("safe_file.txt", "safe_content\n")

    res = process_file_content(zip_buffer.getvalue(), "exploit.zip", sandbox_workspace_dir=tmp_path)
    assert res.error is None
    # Only safe file should be extracted, malicious path traversal ignored
    assert "safe_file.txt" in res.extracted_files
    assert not (tmp_path.parent / "etc" / "passwd").exists()
