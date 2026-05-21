from __future__ import annotations

from tests_tfcore.common import SkipCheck, assert_equal, assert_true, check, require_module


def tiny_arch(arch_type: str):
    from transfolk_core.config.entities.transformer_architecture import TransformerArchitecture
    return TransformerArchitecture(
        id=1,
        name=f"tiny_{arch_type}",
        type=arch_type,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        max_seq_len=8,
    )


def run_tests(ctx):
    def factory_lists_expected_architectures():
        from transfolk_core.model.model_factory import ModelFactory
        names = set(ModelFactory.available_models())
        expected = {"decoder_only", "decoder_only_gpt", "decoder_only_rope", "decoder_only_relative"}
        assert_true(expected.issubset(names), f"Faltan arquitecturas en ModelFactory: {expected - names}")

    check(ctx, "ModelFactory expone arquitecturas esperadas", factory_lists_expected_architectures)

    def all_registered_models_forward_shape():
        torch = require_module("torch")
        from transfolk_core.model.model_factory import ModelFactory
        vocab_size = 20
        x = torch.tensor([[1, 3, 4, 0, 0]], dtype=torch.long)
        mask = x != 0
        for arch_type in ["decoder_only", "decoder_only_gpt", "decoder_only_rope", "decoder_only_relative"]:
            model = ModelFactory.build(tiny_arch(arch_type), vocab_size=vocab_size)
            model.eval()
            with torch.no_grad():
                y = model(x, attention_mask=mask)
            assert_equal(tuple(y.shape), (1, 5, vocab_size), f"Salida incorrecta para {arch_type}")

    check(ctx, "ModelFactory.build + forward shape para todas las arquitecturas", all_registered_models_forward_shape)

    def invalid_architecture_fails_clearly():
        from transfolk_core.model.model_factory import ModelFactory
        try:
            ModelFactory.build(tiny_arch("unknown_arch"), vocab_size=10)
        except ValueError as exc:
            assert_true("Unknown architecture type" in str(exc), "El error debe indicar arquitectura desconocida.")
            return
        raise AssertionError("ModelFactory debe lanzar ValueError con arquitectura desconocida.")

    check(ctx, "ModelFactory rechaza arquitectura desconocida", invalid_architecture_fails_clearly)

    def missing_required_parameter_fails_clearly():
        from transfolk_core.model.model_factory import ModelFactory
        arch = tiny_arch("decoder_only_gpt")
        arch.d_model = None
        try:
            ModelFactory.build(arch, vocab_size=10)
        except ValueError as exc:
            assert_true("Missing required architecture parameter" in str(exc), "El error debe indicar parámetro obligatorio ausente.")
            return
        raise AssertionError("ModelFactory debe fallar si falta d_model.")

    check(ctx, "ModelFactory rechaza parámetros incompletos", missing_required_parameter_fails_clearly)
