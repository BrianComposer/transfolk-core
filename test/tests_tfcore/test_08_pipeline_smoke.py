from __future__ import annotations

import json

from tests_tfcore.common import assert_true, check, make_basic_entities, require_module


def run_tests(ctx):
    def pipeline_loads_tiny_model_and_generates_score():
        torch = require_module("torch")
        require_module("music21")
        from music21 import stream
        from transfolk_core.model.model_factory import ModelFactory
        from transfolk_core.pipeline.pipeline import TransFolkPipeline

        entities = make_basic_entities()
        model_cfg = entities["model"]
        vocab = {
            "PAD": 0,
            "START": 1,
            "END": 2,
            "TS_2/4": 3,
            "MODE_major": 4,
            "BAR": 5,
            "NOTE_ON_60": 6,
            "DUR_1.0": 7,
            "REST": 8,
        }
        model_dir = ctx.temp_dir / "pipeline"
        model_dir.mkdir(parents=True, exist_ok=True)
        vocab_file = model_dir / "vocab.json"
        model_file = model_dir / "weights.pt"
        vocab_file.write_text(json.dumps(vocab), encoding="utf-8")

        model = ModelFactory.build(model_cfg.architecture, vocab_size=len(vocab))
        torch.save(model.state_dict(), model_file)

        pipe = TransFolkPipeline(
            model_file=str(model_file),
            vocab_file=str(vocab_file),
            model_config=model_cfg,
            device="cpu",
        )
        result = pipe.generate_from_TS_tonality(time_signature="2/4", tonality="major", max_len=2, temperature=1.0, topK=3, topP=1.0)
        assert_true(isinstance(result, stream.Score), "TransFolkPipeline.generate_from_TS_tonality debe devolver music21.stream.Score.")

    check(ctx, "Pipeline smoke: carga modelo tiny y genera Score", pipeline_loads_tiny_model_and_generates_score)
