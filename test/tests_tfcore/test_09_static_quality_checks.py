from __future__ import annotations

import ast
import re

from tests_tfcore.common import WarningCheck, assert_true, check, iter_python_files


FORBIDDEN_IMPORT_SNIPPETS = [
    "from apps.",
    "import apps.",
    "from transfolk_config",
    "import transfolk_config",
    "from teimus.",
    "import teimus.",
]


def run_tests(ctx):
    def no_old_cross_project_imports():
        bad = []
        for path in iter_python_files(ctx.src_dir / "transfolk_core"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for snippet in FORBIDDEN_IMPORT_SNIPPETS:
                if snippet in text:
                    bad.append(f"{path.relative_to(ctx.project_root)} -> {snippet}")
        assert_true(not bad, "Imports antiguos o acoplamientos cruzados encontrados:\n" + "\n".join(bad[:50]))

    check(ctx, "estático: sin imports antiguos apps/transfolk_config/teimus", no_old_cross_project_imports)

    def no_sympy_false_import():
        bad = []
        for path in iter_python_files(ctx.src_dir / "transfolk_core"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "from sympy import false" in text:
                bad.append(str(path.relative_to(ctx.project_root)))
        if bad:
            raise WarningCheck("Se ha encontrado 'from sympy import false'. Sustituir por False nativo y eliminar dependencia innecesaria:\n" + "\n".join(bad))

    check(ctx, "estático: sin 'from sympy import false'", no_sympy_false_import)

    def scripts_or_tests_are_not_inside_runtime_packages():
        suspicious = []
        patterns = ["test", "architecture_test", "old"]
        for path in iter_python_files(ctx.src_dir / "transfolk_core"):
            name = path.name.lower()
            if name.startswith("test") or any(p in name for p in ["architecture_test", "old"]):
                suspicious.append(str(path.relative_to(ctx.project_root)))
        if suspicious:
            raise WarningCheck("Hay scripts de test/legacy dentro de src/transfolk_core. Mover a tests/, scripts/ o legacy/:\n" + "\n".join(suspicious[:50]))

    check(ctx, "estático: scripts test/legacy fuera del paquete runtime", scripts_or_tests_are_not_inside_runtime_packages)

    def python_files_parse_with_ast():
        bad = []
        for path in iter_python_files(ctx.src_dir / "transfolk_core"):
            try:
                ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
            except SyntaxError as exc:
                bad.append(f"{path.relative_to(ctx.project_root)}: {exc}")
        assert_true(not bad, "Archivos con SyntaxError:\n" + "\n".join(bad[:50]))

    check(ctx, "estático: todos los .py parsean con ast", python_files_parse_with_ast)

    def broad_bare_except_is_limited():
        bad = []
        for path in iter_python_files(ctx.src_dir / "transfolk_core"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(text.splitlines(), start=1):
                if re.match(r"^\s*except\s*:\s*$", line):
                    bad.append(f"{path.relative_to(ctx.project_root)}:{i}")
                    if len(bad) >= 40:
                        break
        if bad:
            raise WarningCheck("Hay bloques 'except:' sin tipo explícito. Conviene usar excepciones concretas o 'except Exception as e':\n" + "\n".join(bad[:40]))

    check(ctx, "estático: advertencia sobre except: genérico", broad_bare_except_is_limited)
