from __future__ import annotations

import os
from pathlib import Path

from tests_tfcore.common import (
    assert_equal,
    assert_path_inside,
    assert_true,
    check,
    make_basic_entities,
)


def run_tests(ctx):
    def settings_respects_explicit_root():
        from transfolk_core.config.settings import Settings

        root = ctx.temp_dir / "solution" / "transfolk-core"
        root.mkdir(parents=True)

        settings = Settings(root=str(root))

        assert_equal(
            Path(settings.root).resolve(),
            root.resolve(),
            "Settings(root=...) debe respetar exactamente la ruta pasada; no debe aplicar parent.parent.",
        )

    check(ctx, "Settings respeta root explícito", settings_respects_explicit_root)

    def settings_dot_uses_cwd():
        from transfolk_core.config.settings import Settings

        old_cwd = Path.cwd()
        work = ctx.temp_dir / "cwd_root"
        work.mkdir(parents=True, exist_ok=True)

        try:
            os.chdir(work)
            settings = Settings(root=".")

            assert_equal(
                Path(settings.root).resolve(),
                work.resolve(),
                "Settings(root='.') debe resolver al directorio actual.",
            )
        finally:
            os.chdir(old_cwd)

    check(ctx, "Settings(root='.') usa Path.cwd()", settings_dot_uses_cwd)

    def project_paths_are_predictable():
        from transfolk_core.config.paths import ProjectPaths

        root = ctx.temp_dir / "TransFolk2"
        paths = ProjectPaths(root=root)

        assert_equal(paths.data, root / "data")
        assert_equal(paths.data_tokenized, root / "data_tokenized")
        assert_equal(paths.models, root / "models")
        assert_equal(paths.outputs, root / "outputs")
        assert_equal(paths.experiments, root / "experiments")

    check(ctx, "ProjectPaths construye rutas base previsibles", project_paths_are_predictable)

    def path_resolver_core_paths():
        from transfolk_core.config.paths import ProjectPaths
        from transfolk_core.config.resolver import PathResolver

        entities = make_basic_entities()
        root = ctx.temp_dir / "TransFolk2"
        resolver = PathResolver(ProjectPaths(root=root))

        exp = entities["experiment"]
        arch = entities["architecture"]
        model = entities["model"]
        runtime = entities["runtime_generate"]

        assert_path_inside(resolver.data_raw(exp.corpus), root)
        assert_path_inside(resolver.tokenize_dir(exp), root)
        assert_path_inside(resolver.train_dir(arch, exp), root)

        assert_true(
            str(resolver.vocab_file(exp)).endswith(".json"),
            "vocab_file debe terminar en .json",
        )
        assert_true(
            str(resolver.model_file(model)).endswith(".pt"),
            "model_file debe terminar en .pt",
        )
        assert_true(
            str(resolver.generated_new_file(model, runtime)).endswith(".musicxml"),
            "generated_new_file debe terminar en .musicxml",
        )

    check(ctx, "PathResolver genera rutas principales sin salir de root", path_resolver_core_paths)

    def path_resolver_model_snapshot_signature():
        from transfolk_core.config.paths import ProjectPaths
        from transfolk_core.config.resolver import PathResolver

        entities = make_basic_entities()
        root = ctx.temp_dir / "TransFolk2"
        resolver = PathResolver(ProjectPaths(root=root))

        arch = entities["architecture"]
        exp = entities["experiment"]
        model_id = "tiny_model"

        # Firma corregida: model_snapshot_file necesita arch, exp y model_id,
        # porque internamente debe apoyarse en train_dir(arch, exp).
        path = resolver.model_snapshot_file(arch, exp, model_id)

        assert_path_inside(path, root)
        assert_true(
            path.name.endswith(".json"),
            "model_snapshot_file debe devolver un archivo JSON.",
        )
        assert_true(
            model_id in path.name,
            "model_snapshot_file debe incluir el identificador del modelo en el nombre del archivo.",
        )

    check(ctx, "PathResolver.model_snapshot_file funciona", path_resolver_model_snapshot_signature)