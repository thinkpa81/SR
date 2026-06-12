from __future__ import annotations

from pathlib import Path
from typing import Iterable


EXCLUDED_DIR_NAMES = {"__pycache__", "logs", "temp", "interim"}


def _looks_like_project_dir(candidate: Path) -> bool:
    raw_dir = candidate / "raw"
    if raw_dir.is_dir():
        has_raw_data = (
            any(raw_dir.rglob("eklips*.xls*"))
            or any(raw_dir.rglob("codebook*.xls*"))
            or any(raw_dir.rglob("*.pdf"))
        )
        has_scripts = (candidate / "python").is_dir() and any((candidate / "python").glob("*.py"))
        has_submission_dir = any(
            path.is_dir() and path.name.startswith("!SR")
            for path in candidate.iterdir()
        )
        if has_raw_data or has_scripts or has_submission_dir:
            return True
    return (candidate / "outputs_klips_sr").exists() or (candidate / "paper_final").exists()


def resolve_project_dir(start: Path | None = None) -> Path:
    anchor = (start or Path(__file__).resolve().parent).resolve()
    for candidate in [anchor, *anchor.parents]:
        if _looks_like_project_dir(candidate):
            return candidate
    if anchor.name.lower() == "raw":
        return anchor.parent
    return anchor


def resolve_scripts_dir(project_dir: Path | None = None, start: Path | None = None) -> Path:
    anchor = (start or Path(__file__).resolve().parent).resolve()
    if anchor.is_dir() and any(anchor.glob("*.py")):
        return anchor
    base_dir = (project_dir or resolve_project_dir(anchor)).resolve()
    raw_dir = base_dir / "raw"
    if raw_dir.is_dir():
        return raw_dir
    return base_dir


def discover_python_scripts(scripts_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in scripts_dir.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )


def discover_raw_excel_files(raw_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_dir.glob("*")
        if path.is_file() and path.suffix.lower() in {".xls", ".xlsx"}
    )


def iter_files(root: Path, skip_dir_names: Iterable[str] | None = None) -> list[Path]:
    skip = set(skip_dir_names or ())
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in skip for part in path.parts):
            continue
        files.append(path)
    return files
