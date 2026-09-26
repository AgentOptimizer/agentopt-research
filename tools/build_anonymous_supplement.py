"""Build the reviewed source selection without Git or filesystem metadata."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "supplement_manifest.txt"
ARCHIVE_ROOT = "cc-gittins-supplement"
TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def selected_files(root: Path) -> list[tuple[str, Path]]:
    root = root.resolve()
    selected = []
    seen: set[str] = set()
    for line in (root / MANIFEST).read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        relative = PurePosixPath(name)
        if ("\\" in name or ":" in name or relative.is_absolute()
                or any(part.startswith(".") for part in relative.parts)
                or name != relative.as_posix()):
            raise ValueError(f"Unsafe manifest entry: {name}")
        if name in seen or name == "CHECKSUMS.sha256":
            raise ValueError(f"Duplicate or reserved manifest entry: {name}")
        seen.add(name)
        path = root.joinpath(*relative.parts)
        if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root):
            raise ValueError(f"Symlinks are not allowed: {name}")
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or external manifest file: {name}")
        selected.append((name, path))
    if not selected:
        raise ValueError("Empty supplement manifest")
    return sorted(selected)


def archive_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{name}", date_time=TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.comment = b""
    info.extra = b""
    return info


def build_archive(root: Path, output: Path) -> dict[str, int | str]:
    files = selected_files(root)
    if any(output.resolve() == path.resolve() for _, path in files):
        raise ValueError("Output would overwrite a selected source file")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    checksums = []
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        archive.comment = b""
        for name, path in files:
            content = path.read_bytes()
            checksums.append(f"{hashlib.sha256(content).hexdigest()}  {name}\n")
            archive.writestr(archive_info(name), content, compresslevel=9)
        archive.writestr(archive_info("CHECKSUMS.sha256"),
                         "".join(checksums).encode("utf-8"), compresslevel=9)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive integrity check failed")
    temporary.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii")
    return {"files": len(files) + 1, "bytes": output.stat().st_size,
            "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "dist/anonymous-supplement.zip")
    args = parser.parse_args()
    import json
    print(json.dumps(build_archive(ROOT, args.output), indent=2))


if __name__ == "__main__":
    main()
