 #!/usr/bin/env python3
"""
Load and prepare case study notebooks for documentation builds.

This script is intended to be run before Sphinx builds, both locally and in CI.

Workflow
--------
1. Clone or update each study project repository into ``../_tmp/study_root/<name>``.
2. Check out the requested ``project_ref`` in the project repository.
3. Check out the requested ``output_ref`` in the corresponding output repository at
   ``<project>/output``.
4. Read ``<project>/output/options.json`` to determine the directory containing the
   case study sources (key: ``source_directory``).
5. Copy that directory into ``docs/source/case_studies/<name>`` using an atomic replace
   to avoid stale or mixed content.
6. Set copied notebook metadata so MyST-NB does not execute them during docs builds.
7. Update the Case Studies toctree in ``docs/source/index.md`` (visible list).

Notes
-----
- Only HTTPS repository URLs are supported.
- This script only writes to ``docs/source/case_studies`` and does not touch legacy
  content in ``docs/examples``.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import stat
from glob import glob
from pathlib import Path
from typing import Iterable, Sequence

import git
import nbformat as nbf
from cadetrdm import ProjectRepo

DOCS_SOURCE = Path(__file__).resolve().parent
os.chdir(DOCS_SOURCE)


def delete_path(path: Path) -> None:
    """
    Remove a file or directory if it exists.

    Parameters
    ----------
    path
        File or directory to remove.
    """
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def checkout_ref(repo_dir: Path, ref: str) -> None:
    """
    Check out a ref in an existing git repository directory.

    This supports commit hashes, tags, local branches, and remote branches (fallback
    to ``origin/<ref>``).

    Parameters
    ----------
    repo_dir
        Path to the git repository directory.
    ref
        Ref to check out.

    Raises
    ------
    RuntimeError
        If the ref cannot be checked out.
    """
    repo = git.Repo(repo_dir)
    repo.git.fetch("--prune", "--tags")

    try:
        repo.git.checkout(ref)
        return
    except git.exc.GitCommandError:
        pass

    try:
        repo.git.checkout("-B", ref, f"origin/{ref}")
    except git.exc.GitCommandError as exc:
        raise RuntimeError(f"Ref not found: {ref} in {repo_dir}") from exc


def read_source_directory_from_options(output_dir: Path) -> str:
    """
    Read ``options.json`` in the output repository and return the source directory.

    Parameters
    ----------
    output_dir
        Path to the output repository directory (``<project>/output``).

    Returns
    -------
    str
        The value of ``source_directory``.

    Raises
    ------
    RuntimeError
        If ``options.json`` is missing or does not contain a valid ``source_directory``.
    """
    options_path = output_dir / "options.json"
    if not options_path.exists():
        raise RuntimeError(f"Missing options.json in {output_dir}")

    data = json.loads(options_path.read_text(encoding="utf-8"))
    source_dir = data.get("source_directory")
    if not isinstance(source_dir, str) or not source_dir.strip():
        raise RuntimeError(f"options.json missing valid source_directory in {output_dir}")

    return source_dir


def set_execution_mode_to_off(notebooks: Iterable[Path]) -> None:
    """
    Disable execution for notebooks by setting MyST-NB metadata.

    Parameters
    ----------
    notebooks
        Iterable of notebook paths to patch.
    """
    for nb_path in notebooks:
        try:
            mode = nb_path.stat().st_mode
            os.chmod(nb_path, mode | stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

        ntbk = nbf.read(nb_path, nbf.NO_CONVERT)
        ntbk.metadata["mystnb"] = {"execution_mode": "off"}

        with open(nb_path, "w", encoding="utf-8") as f:
            nbf.write(ntbk, f)


def update_index_with_study_tocs(docnames: Sequence[str], *, index_path: Path) -> None:
    """
    Update the Case Studies toctree in index.md by preserving existing entries and
    appending generated docnames (without duplicates).

    This keeps manually maintained entries such as `examples/...` and adds the new
    case studies afterwards.

    Parameters
    ----------
    docnames
        Docnames (relative to docs/source) to include in the Case Studies toctree.
        Docnames must not include file extensions.
    index_path
        Path to docs/source/index.md.

    Raises
    ------
    RuntimeError
        If the Case Studies toctree block cannot be found.
    """
    content = index_path.read_text(encoding="utf-8")

    # Find a fenced MyST toctree block that contains ":caption: Case Studies"
    # We match:
    #   ```{toctree}
    #   ... any lines ...
    #   ```
    block_re = re.compile(r"```{toctree}\n.*?\n```\n", flags=re.DOTALL)
    blocks = list(block_re.finditer(content))

    target_match = None
    for m in blocks:
        block_text = m.group(0)
        if ":caption: Case Studies" in block_text:
            target_match = m
            break

    if target_match is None:
        raise RuntimeError("Could not find a ```{toctree} block with ':caption: Case Studies'")

    block = target_match.group(0)
    lines = block.splitlines(keepends=False)

    # Preserve the opening and closing fences
    opening = lines[0]  # ```{toctree}
    closing = lines[-1]  # ```

    middle = lines[1:-1]

    # Separate option lines (start with ":") from entry lines
    option_lines: list[str] = []
    entry_lines: list[str] = []

    for line in middle:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        if stripped.startswith(":"):
            option_lines.append(stripped)
        else:
            entry_lines.append(stripped)

    # Merge entries: keep existing first, then add new docnames not already present
    merged_entries = list(entry_lines)
    for d in docnames:
        if d not in merged_entries:
            merged_entries.append(d)

    # Rebuild the block.
    # Keep options exactly as options, and keep the block visible (do not add :hidden:).
    # If you want to preserve an existing :hidden:, remove the filtering below.
    option_lines = [opt for opt in option_lines if opt != ":hidden:"]

    rebuilt_lines: list[str] = [opening]
    rebuilt_lines.extend(option_lines)
    rebuilt_lines.append("")  # blank line between options and entries
    rebuilt_lines.extend(merged_entries)
    rebuilt_lines.append(closing)
    rebuilt = "\n".join(rebuilt_lines) + "\n"

    updated = content[: target_match.start()] + rebuilt + content[target_match.end() :]
    index_path.write_text(updated, encoding="utf-8")


def _windows_ignore_comparison_plots():
    if platform.system() != "Windows":
        return None

    def ignore(_src: str, names: Sequence[str]) -> list[str]:
        return [n for n in names if "_comparison.png" in n]

    return ignore


def load_study_results(
    *,
    name: str,
    study_url: str,
    project_ref: str,
    output_ref: str,
    study_root: Path,
    case_studies_root: Path,
) -> list[str]:
    """
    Load a single study and return docnames for the Case Studies toctree.

    Parameters
    ----------
    name
        Local study folder name used under ``study_root`` and ``case_studies_root``.
    study_url
        HTTPS URL of the study project repository.
    project_ref
        Ref to check out in the project repository (typically ``"main"``).
    output_ref
        Ref to check out in the output repository at ``<project>/output``.
    study_root
        Root directory where project repositories are cloned.
    case_studies_root
        Target directory for generated case studies (``docs/source/case_studies``).

    Returns
    -------
    list[str]
        Docnames (relative to ``docs/source``) for notebooks to include in the toctree.
    """
    if not study_url.startswith("https://"):
        raise ValueError(f"Only HTTPS URLs are supported: {study_url}")

    project_repo = ProjectRepo(path=study_root / name, url=study_url, branch="main")
    project_repo.update()

    checkout_ref(project_repo.path, project_ref)

    output_dir = project_repo.path / "output"
    if not output_dir.exists():
        raise RuntimeError(f"Expected output repo at {output_dir}")

    checkout_ref(output_dir, output_ref)

    source_dir_name = read_source_directory_from_options(output_dir)
    src = output_dir / source_dir_name
    if not src.exists():
        raise RuntimeError(f"Source directory not found: {src}")

    target = case_studies_root / name
    tmp_target = target.with_name(f"{target.name}.__tmp__")

    delete_path(tmp_target)
    shutil.copytree(src, tmp_target, ignore=_windows_ignore_comparison_plots())

    delete_path(target)
    tmp_target.rename(target)

    notebooks = [Path(p) for p in glob((target / "**/*.ipynb").as_posix(), recursive=True)]
    set_execution_mode_to_off(notebooks)

    docnames = [
        nb.resolve().relative_to(DOCS_SOURCE).with_suffix("").as_posix()
        for nb in notebooks
    ]

    study_index = (target / "index.ipynb").resolve()
    if study_index.exists():
        return [study_index.relative_to(DOCS_SOURCE).with_suffix("").as_posix()]

    return docnames

def main() -> None:
    """
    Run the case study loader for all configured studies and update the documentation index.
    """
    index_path = DOCS_SOURCE / "index.md"
    study_root = DOCS_SOURCE / "../_tmp/study_root"
    case_studies_root = DOCS_SOURCE / "case_studies"
    case_studies_root.mkdir(parents=True, exist_ok=True)

    studies: list[tuple[str, str, str]] = [
        (
            "https://github.com/cadet/RDM-Example-Multi-State-Steric-Mass-Action",
            "main",
            "2026-02-11_14-11-23_main_406bbee",
        ),
        (
            "https://github.com/cadet/RDM-Example-Rectangular-Pulse",
            "main",
            "2026-03-06_10-01-32_main_b2115b3",
        ),
        (
            "https://github.com/cadet/RDM-Example-Simulated-Moving-Bed",
            "main",
            "2026-03-26_11-09-10_main_3a25a18_108b0d",
        ),
    ]

    toc_docnames: list[str] = []

    for url, project_ref, output_ref in studies:
        name = url.split("/")[-1].replace(".git", "").replace("RDM-Example-", "")
        toc_docnames.extend(
            load_study_results(
                name=name,
                study_url=url,
                project_ref=project_ref,
                output_ref=output_ref,
                study_root=study_root,
                case_studies_root=case_studies_root,
            )
        )

    update_index_with_study_tocs(toc_docnames, index_path=index_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"load_studies.py failed: {exc}") from exc