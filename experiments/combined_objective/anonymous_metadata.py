"""Portable references for experiment metadata distributed with the supplement."""
from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any, Mapping


def artifact_reference(path: str | Path, *, root: Path) -> str:
    """Use a repository-relative path, or only the filename for external input.

    Handle saved Windows paths even when an archive is inspected on Unix.
    The enclosing benchmark/seed and content hashes identify external files;
    their original directory names are unnecessary for reproducibility.
    """
    windows_path = PureWindowsPath(str(path))
    if windows_path.is_absolute() and not Path(str(path)).is_absolute():
        return f"external/{windows_path.name}"
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return f"external/{resolved.name}"


def shareable_provenance(provenance: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Retain known content fingerprints without copying machine metadata."""
    result = {
        key: provenance[key]
        for key in ("source_result_sha256", "launcher_source_sha256", "compacted_at_utc")
        if key in provenance
    }
    if provenance.get("source_result"):
        result["source_result"] = artifact_reference(provenance["source_result"], root=root)
    return result
