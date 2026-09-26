import hashlib
from pathlib import Path
import zipfile

import pytest

from tools.build_anonymous_supplement import ARCHIVE_ROOT, build_archive, selected_files


def test_archive_is_reproducible_and_only_contains_selected_files(tmp_path):
    (tmp_path / "README.md").write_text("Anonymous example\n", encoding="utf-8")
    (tmp_path / "unlisted.log").write_text("Local notes", encoding="utf-8")
    (tmp_path / "supplement_manifest.txt").write_text("README.md\n", encoding="utf-8")
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    build_archive(tmp_path, first)
    build_archive(tmp_path, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.comment == b""
        assert set(archive.namelist()) == {
            f"{ARCHIVE_ROOT}/README.md", f"{ARCHIVE_ROOT}/CHECKSUMS.sha256"}
        for entry in archive.infolist():
            assert entry.date_time == (1980, 1, 1, 0, 0, 0)
            assert not entry.extra and not entry.comment
        expected = hashlib.sha256((tmp_path / "README.md").read_bytes()).hexdigest()
        assert archive.read(f"{ARCHIVE_ROOT}/CHECKSUMS.sha256").decode() == (
            f"{expected}  README.md\n")


@pytest.mark.parametrize("entry", ["../secret", ".git/config", "/etc/file", "C:/file",
                                       "dir/../../file", "dir\\file", "a//b"])
def test_manifest_rejects_unsafe_paths(tmp_path, entry):
    (tmp_path / "supplement_manifest.txt").write_text(entry + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe"):
        selected_files(tmp_path)


def test_missing_reviewed_file_fails_instead_of_silently_omitting_it(tmp_path):
    (tmp_path / "supplement_manifest.txt").write_text("missing.py\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing"):
        selected_files(tmp_path)
