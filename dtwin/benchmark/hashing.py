"""Hashes e identificação do estado do código para reprodução do benchmark."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable

from dtwin.core import sha256_of


def sha256_paths(paths: Iterable[Path]) -> str:
    """Hash estável de nomes relativos/ordenados e conteúdo de vários arquivos."""
    normalized = sorted(Path(path).resolve() for path in paths)
    digest = hashlib.sha256()
    for path in normalized:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_of(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def git_state(repo: Path) -> dict[str, str | bool | None]:
    repo = Path(repo).resolve()

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )

    commit_result = run("rev-parse", "HEAD")
    if commit_result.returncode != 0:
        return {"code_commit": None, "git_dirty": True, "git_diff_sha256": None}
    status = run("status", "--porcelain", "--untracked-files=all")
    diff = run("diff", "--binary", "HEAD")
    dirty = bool(status.stdout.strip())
    untracked = run("ls-files", "--others", "--exclude-standard")
    untracked_material = bytearray()
    for relative in sorted(line for line in untracked.stdout.splitlines() if line):
        path = repo / relative
        untracked_material.extend(relative.encode("utf-8"))
        untracked_material.extend(b"\0")
        if path.is_file():
            untracked_material.extend(sha256_of(path).encode("ascii"))
        untracked_material.extend(b"\0")
    diff_material = (status.stdout + "\n" + diff.stdout).encode("utf-8") + bytes(untracked_material)
    return {
        "code_commit": commit_result.stdout.strip(),
        "git_dirty": dirty,
        "git_diff_sha256": hashlib.sha256(diff_material).hexdigest() if dirty else None,
    }


def input_hashes(volume: Path, organ_mask: Path, manifest: Path) -> dict[str, str]:
    return {
        "volume": sha256_of(Path(volume)),
        "mask_organ": sha256_of(Path(organ_mask)),
        "manifest": sha256_of(Path(manifest)),
    }
