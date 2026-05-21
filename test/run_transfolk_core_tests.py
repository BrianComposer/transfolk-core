from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

from tests_tfcore.common import TestContext, add_project_to_syspath

TEST_FILES = [
    "test_00_imports_and_layout.py",
    "test_01_settings_paths_resolver.py",
    "test_02_entities_serialization.py",
    "test_03_config_registry_sqlite.py",
    "test_04_tokenization_core.py",
    "test_05_decoder_music21.py",
    "test_06_model_factory_forward.py",
    "test_07_training_factories.py",
    "test_08_pipeline_smoke.py",
    "test_09_static_quality_checks.py",
]

STATUS_ICONS = {
    "PASS": "OK",
    "FAIL": "FAIL",
    "ERROR": "ERROR",
    "SKIP": "SKIP",
    "WARN": "WARN",
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar el módulo de test: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def find_project_root(raw_root: str | None) -> Path:
    if raw_root:
        return Path(raw_root).resolve()

    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents]
    for c in candidates:
        if (c / "src" / "transfolk_core").exists():
            return c
    raise RuntimeError(
        "No encuentro la raíz de transfolk-core. Ejecuta este runner desde la raíz del proyecto "
        "o usa: python run_transfolk_core_tests.py --root C:/ruta/transfolk-core"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Runner de pruebas funcionales para transfolk-core.")
    parser.add_argument("--root", type=str, default=None, help="Ruta raíz de transfolk-core. Por defecto se autodetecta.")
    parser.add_argument("--verbose", action="store_true", help="Muestra detalles de errores y warnings.")
    parser.add_argument("--fail-on-warn", action="store_true", help="Devuelve código de error también si hay warnings.")
    args = parser.parse_args()

    project_root = find_project_root(args.root)
    src_dir = add_project_to_syspath(project_root)
    tests_dir = Path(__file__).resolve().parent / "tests_tfcore"

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="transfolk_core_tests_") as tmp:
        ctx = TestContext(project_root=project_root, src_dir=src_dir, temp_dir=Path(tmp), verbose=args.verbose)

        print("=" * 78)
        print("TRANSFOLK-CORE TEST REPORT")
        print("=" * 78)
        print(f"Project root: {project_root}")
        print(f"Source dir  : {src_dir}")
        print(f"Tests dir   : {tests_dir}")
        print("-" * 78)

        for file_name in TEST_FILES:
            file_path = tests_dir / file_name
            before = len(ctx.results)
            print(f"\n[{file_name}]")
            try:
                module = load_module(file_path)
                module.run_tests(ctx)
            except Exception as exc:
                import traceback
                ctx.record("ERROR", file_name, f"Error cargando/ejecutando archivo: {type(exc).__name__}: {exc}", traceback.format_exc())

            for r in ctx.results[before:]:
                print(f"  {STATUS_ICONS.get(r.status, r.status):5} {r.name}")
                if r.message:
                    print(f"        {r.message}")
                if args.verbose and r.details:
                    print(r.details.rstrip())

        elapsed = time.time() - started
        counts = {status: sum(1 for r in ctx.results if r.status == status) for status in ["PASS", "FAIL", "ERROR", "WARN", "SKIP"]}

        print("\n" + "=" * 78)
        print("SUMMARY")
        print("=" * 78)
        print(f"PASS : {counts['PASS']}")
        print(f"FAIL : {counts['FAIL']}")
        print(f"ERROR: {counts['ERROR']}")
        print(f"WARN : {counts['WARN']}")
        print(f"SKIP : {counts['SKIP']}")
        print(f"Time : {elapsed:.2f}s")

        if counts["FAIL"] or counts["ERROR"] or (args.fail_on_warn and counts["WARN"]):
            print("\nResultado: hay comprobaciones que requieren corrección.")
            return 1

        print("\nResultado: batería completada sin fallos críticos.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
