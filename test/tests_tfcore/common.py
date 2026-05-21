from __future__ import annotations

import importlib
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable


@dataclass
class CheckResult:
    name: str
    status: str
    message: str = ""
    details: str = ""


@dataclass
class TestContext:
    project_root: Path
    src_dir: Path
    temp_dir: Path
    verbose: bool = False
    results: list[CheckResult] = field(default_factory=list)

    def record(self, status: str, name: str, message: str = "", details: str = "") -> None:
        self.results.append(CheckResult(name=name, status=status, message=message, details=details))


def add_project_to_syspath(project_root: Path) -> Path:
    src_dir = project_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return src_dir


def check(ctx: TestContext, name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        ctx.record("PASS", name)
    except SkipCheck as exc:
        ctx.record("SKIP", name, str(exc))
    except WarningCheck as exc:
        ctx.record("WARN", name, str(exc))
    except AssertionError as exc:
        ctx.record("FAIL", name, str(exc), traceback.format_exc())
    except Exception as exc:
        ctx.record("ERROR", name, f"{type(exc).__name__}: {exc}", traceback.format_exc())


class SkipCheck(Exception):
    pass


class WarningCheck(Exception):
    pass


def require_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise SkipCheck(f"Dependencia no disponible o import fallido: {module_name}. Detalle: {exc}") from exc


def assert_equal(actual, expected, message: str = "") -> None:
    assert actual == expected, message or f"Esperado {expected!r}, recibido {actual!r}"


def assert_true(condition: bool, message: str) -> None:
    assert condition, message


def assert_path_inside(path: Path, parent: Path, message: str = "") -> None:
    path = path.resolve()
    parent = parent.resolve()
    assert parent == path or parent in path.parents, message or f"{path} no está dentro de {parent}"


def make_basic_entities():
    from transfolk_core.config import (
        AllowedDurations,
        Corpus,
        Experiment,
        Model,
        MusicContext,
        RuntimeGenerate,
        RuntimeTrain,
        TokenizerAlgorithm,
        TransformerArchitecture,
    )

    corpus = Corpus(id=1, name="unified-iberian")
    tokenizer = TokenizerAlgorithm(id=1, name="momet")
    music_context = MusicContext(id=1, name="major_2_4", tonality="major", time_signature="2/4")
    allowed = AllowedDurations(id=1, name="standard", durations=[0.0, 0.25, 0.5, 1.0, 2.0])
    exp = Experiment(
        id=1,
        name="exp_test",
        corpus=corpus,
        tokenizer=tokenizer,
        music_context=music_context,
        allowed_durations=allowed,
    )
    arch = TransformerArchitecture(
        id=1,
        name="tiny_gpt",
        type="decoder_only_gpt",
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        max_seq_len=32,
    )
    rt = RuntimeTrain(id=1, name="rt_test", epochs=1, batch_size=2, learning_rate=1e-3, optimizer="adamw", loss="cross_entropy")
    rg = RuntimeGenerate(id=1, name="rg_test", temperature=1.0, max_len=8)
    model = Model(id=1, name="tiny_model", architecture=arch, experiment=exp, runtime_train=rt, vocab_file="vocab.json")
    return {
        "corpus": corpus,
        "tokenizer": tokenizer,
        "music_context": music_context,
        "allowed": allowed,
        "experiment": exp,
        "architecture": arch,
        "runtime_train": rt,
        "runtime_generate": rg,
        "model": model,
    }


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if "__pycache__" not in path.parts:
            yield path
