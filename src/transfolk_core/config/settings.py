"""
Runtime settings for TransFolk projects.

This module is intentionally independent from a fixed local folder.  It is
meant to work when ``transfolk_core`` is used from different host projects,
for example:

- local development, with a parent solution folder containing several repos;
- a backend deployment on Render;
- a training/inference VM such as RunPod;
- scripts executed from nested folders inside one of the TransFolk projects.

Resolution order for ``Settings.root``:

1. Explicit ``root`` argument.
2. Environment variables, in priority order.
3. Automatic upward search from the current working directory.
4. Automatic upward search from this file location.
5. Current working directory as safe fallback.

The detected root is the active runtime project root.  In a multi-project
local checkout this may be the solution root if the process is launched from
there.  In Render or RunPod it will normally be the checked-out backend,
training, or experiment repository root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence


class Settings:
    """
    Resolve the active TransFolk runtime root in a portable way.

    Parameters
    ----------
    root:
        Optional explicit root.  Use this when a script, backend, notebook, or
        VM launcher knows the desired runtime folder.  Passing "." uses the
        current working directory.  File paths are accepted and resolved to
        their parent folder.
    env_vars:
        Optional override for the environment-variable priority list.
    require_existing:
        If True, raise FileNotFoundError when the resolved root does not exist.
    verbose:
        If True, print the detected root and the source used to obtain it.
    """

    ENV_VARS: Sequence[str] = (
        "TRANSFOLK_ROOT",          # Preferred new name.
        "TRANSFOLK_PROJECT_ROOT",  # Explicit project/runtime root.
        "TRANSFOLK_BACKEND_ROOT",  # Useful for backend deployments.
        "TRANSFOLK_CORE_ROOT",     # Useful only when core itself is the runtime project.
        "TRANSFOLK",              # Backwards compatibility with older code.
        "RENDER_PROJECT_DIR",      # Optional user-defined Render env var.
        "RUNPOD_PROJECT_DIR",      # Optional user-defined RunPod env var.
        "PROJECT_ROOT",           # Generic fallback used in some deployments.
    )

    # Markers that identify a TransFolk multi-project solution directory.
    SOLUTION_DIR_MARKERS: Sequence[str] = (
        "transfolk-core",
        "transfolk-backend",
        "transfolk-frontend",
    )

    # Markers that identify an individual Python/JS project root.
    PROJECT_FILE_MARKERS: Sequence[str] = (
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
        "setup.cfg",
        "package.json",
        "render.yaml",
        ".git",
    )

    def __init__(
        self,
        root: Optional[os.PathLike[str] | str] = None,
        *,
        env_vars: Optional[Iterable[str]] = None,
        require_existing: bool = True,
        verbose: bool = False,
    ) -> None:
        self._source = "fallback"
        self.root = self._resolve_root(
            root=root,
            env_vars=tuple(env_vars or self.ENV_VARS),
            require_existing=require_existing,
        )

        if verbose:
            print(f"TRANSFOLK SETTINGS ROOT = {self.root} ({self._source})")

    @property
    def source(self) -> str:
        """Return the source used to resolve ``root``."""
        return self._source

    def path(self, *parts: os.PathLike[str] | str) -> Path:
        """Build an absolute path under the detected project root."""
        return self.root.joinpath(*map(str, parts))

    def __repr__(self) -> str:
        return f"Settings(root={str(self.root)!r}, source={self._source!r})"

    def _resolve_root(
        self,
        *,
        root: Optional[os.PathLike[str] | str],
        env_vars: Sequence[str],
        require_existing: bool,
    ) -> Path:
        # 1. Explicit root argument.  This is the safest option for scripts,
        # Render start commands, RunPod jobs, and tests.
        if root is not None:
            resolved = self._normalise_candidate(root)
            self._source = "argument:root"
            return self._validate_root(resolved, require_existing)

        # 2. Environment variables.  This keeps deployments deterministic and
        # removes any dependency on the physical location of the repository.
        for var_name in env_vars:
            value = os.environ.get(var_name)
            if value:
                resolved = self._normalise_candidate(value)
                self._source = f"environment:{var_name}"
                return self._validate_root(resolved, require_existing)

        cwd = Path.cwd().resolve()

        # 3. Prefer automatic detection from the process working directory.
        # This is what normally matters when transfolk_core is imported by
        # transfolk-backend, a training repo, or a RunPod script.
        detected = self._find_runtime_root(cwd)
        if detected is not None:
            self._source = "auto:cwd"
            return self._validate_root(detected, require_existing)

        # 4. Secondary detection from this file.  This helps editable installs
        # and direct development inside transfolk-core.
        detected = self._find_runtime_root(Path(__file__).resolve())
        if detected is not None:
            self._source = "auto:package"
            return self._validate_root(detected, require_existing)

        # 5. Safe fallback.  Do not guess parent.parent or any fixed number of
        # levels; the caller can always pass root explicitly if needed.
        self._source = "fallback:cwd"
        return self._validate_root(cwd, require_existing)

    @classmethod
    def _normalise_candidate(cls, value: os.PathLike[str] | str) -> Path:
        text = os.fspath(value).strip()
        candidate = Path.cwd() if text == "." else Path(text).expanduser()
        candidate = candidate.resolve()

        # If the caller passes a file path, use its parent as root.
        # Existing directories are kept unchanged.
        if candidate.exists() and candidate.is_file():
            return candidate.parent.resolve()

        # For non-existing paths with a suffix, treat them as file-like paths.
        if not candidate.exists() and candidate.suffix:
            return candidate.parent.resolve()

        return candidate

    @classmethod
    def _validate_root(cls, candidate: Path, require_existing: bool) -> Path:
        if require_existing and not candidate.exists():
            raise FileNotFoundError(f"Resolved TransFolk root does not exist: {candidate}")
        if require_existing and not candidate.is_dir():
            raise NotADirectoryError(f"Resolved TransFolk root is not a directory: {candidate}")
        return candidate.resolve()

    @classmethod
    def _find_runtime_root(cls, start: Path) -> Optional[Path]:
        """
        Find the closest meaningful runtime root by walking upwards.

        Priority is intentionally conservative:
        - first, detect a multi-project solution folder when present;
        - then, detect an individual project repository;
        - otherwise, return None and let the caller fall back safely.
        """
        start = start.resolve()
        current = start if start.is_dir() else start.parent

        parents = (current, *current.parents)

        solution_root = cls._find_solution_root(parents)
        if solution_root is not None:
            return solution_root

        project_root = cls._find_project_root(parents)
        if project_root is not None:
            return project_root

        return None

    @classmethod
    def _find_solution_root(cls, candidates: Sequence[Path]) -> Optional[Path]:
        for candidate in candidates:
            existing_markers = [
                marker
                for marker in cls.SOLUTION_DIR_MARKERS
                if (candidate / marker).is_dir()
            ]

            # The solution root should contain at least two TransFolk projects.
            # This avoids incorrectly treating a project root as the solution
            # merely because it contains a folder named transfolk-core.
            if len(existing_markers) >= 2:
                return candidate.resolve()

        return None

    @classmethod
    def _find_project_root(cls, candidates: Sequence[Path]) -> Optional[Path]:
        for candidate in candidates:
            if any((candidate / marker).exists() for marker in cls.PROJECT_FILE_MARKERS):
                return candidate.resolve()

        return None
