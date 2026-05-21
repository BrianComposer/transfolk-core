from __future__ import annotations

from pathlib import Path

from tests_tfcore.common import WarningCheck, assert_true, check


ESSENTIAL_IMPORTS = [
    "transfolk_core",
    "transfolk_core.config",
    "transfolk_core.config.settings",
    "transfolk_core.config.paths",
    "transfolk_core.config.resolver",
    "transfolk_core.config.entities.model",
    "transfolk_core.db.config_registry",
    "transfolk_core.model.model_factory",
    "transfolk_core.training.loss_factory",
    "transfolk_core.training.optimizer_factory",
]

OPTIONAL_DEPENDENCY_IMPORTS = [
    "transfolk_core.tokenization.tokenizer",
    "transfolk_core.tokenization.decoder",
    "transfolk_core.generation.generator",
]


def run_tests(ctx):
    def package_layout_exists():
        assert_true((ctx.src_dir / "transfolk_core").is_dir(), "No existe src/transfolk_core.")
        assert_true((ctx.project_root / "pyproject.toml").is_file(), "No existe pyproject.toml en la raíz.")

    check(ctx, "layout básico src/transfolk_core + pyproject", package_layout_exists)

    def essential_imports():
        failures = []
        for name in ESSENTIAL_IMPORTS:
            try:
                __import__(name)
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        assert_true(not failures, "Imports esenciales fallidos:\n" + "\n".join(failures))

    check(ctx, "imports esenciales del paquete", essential_imports)

    def optional_dependency_imports():
        failures = []
        for name in OPTIONAL_DEPENDENCY_IMPORTS:
            try:
                __import__(name)
            except ModuleNotFoundError as exc:
                raise WarningCheck(f"No se pudo importar {name} porque falta una dependencia opcional/declarada: {exc.name}") from exc
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        assert_true(not failures, "Imports con dependencias externas fallidos:\n" + "\n".join(failures))

    check(ctx, "imports con dependencias externas declaradas", optional_dependency_imports)

    def pyproject_uses_src_layout():
        text = (ctx.project_root / "pyproject.toml").read_text(encoding="utf-8")
        assert_true("[tool.setuptools.packages.find]" in text, "pyproject.toml no declara tool.setuptools.packages.find.")
        assert_true('where = ["src"]' in text or "where=['src']" in text.replace(" ", ""), "pyproject.toml no parece usar layout src correctamente.")

    check(ctx, "pyproject configurado para src-layout", pyproject_uses_src_layout)

    def no_pycache_or_pyc_in_repository():
        bad = []
        for p in ctx.project_root.rglob("*"):
            if "__pycache__" in p.parts or p.suffix in {".pyc", ".pyo"}:
                bad.append(str(p.relative_to(ctx.project_root)))
                if len(bad) >= 20:
                    break
        if bad:
            raise WarningCheck("Hay cachés Python dentro del repositorio. Deben eliminarse. Ejemplos:\n" + "\n".join(bad))

    check(ctx, "higiene: sin __pycache__ ni .pyc versionados", no_pycache_or_pyc_in_repository)

    def no_operational_data_inside_src():
        suspicious_suffixes = {".db", ".sqlite", ".sqlite3", ".pt", ".pth", ".joblib", ".pkl"}
        bad = [p.relative_to(ctx.project_root).as_posix() for p in (ctx.src_dir / "transfolk_core").rglob("*") if p.is_file() and p.suffix.lower() in suspicious_suffixes]
        if bad:
            raise WarningCheck("Hay datos/modelos operativos dentro de src/transfolk_core. Mover a transfolk-data, transfolk-experiments o backend/models. Archivos:\n" + "\n".join(bad[:20]))

    check(ctx, "higiene: sin datos/modelos operativos dentro de src", no_operational_data_inside_src)
