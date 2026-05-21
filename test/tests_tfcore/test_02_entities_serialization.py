from __future__ import annotations

import json

from tests_tfcore.common import assert_equal, assert_true, check, make_basic_entities


def run_tests(ctx):
    def allowed_durations_behaves_as_collection():
        entities = make_basic_entities()
        allowed = entities["allowed"]
        assert_equal(len(allowed), 5)
        assert_equal(list(allowed), [0.0, 0.25, 0.5, 1.0, 2.0])
        assert_equal(allowed.values, allowed.durations)

    check(ctx, "AllowedDurations funciona como colección", allowed_durations_behaves_as_collection)

    def nested_model_to_dict_contains_class_markers():
        model = make_basic_entities()["model"]
        data = model.to_dict()
        assert_equal(data["__class__"], "Model")
        assert_equal(data["architecture"]["__class__"], "TransformerArchitecture")
        assert_equal(data["experiment"]["corpus"]["__class__"], "Corpus")
        assert_equal(data["experiment"]["allowed_durations"]["durations"], [0.0, 0.25, 0.5, 1.0, 2.0])

    check(ctx, "Serializable.to_dict serializa entidades anidadas", nested_model_to_dict_contains_class_markers)

    def save_and_load_json_roundtrip():
        from transfolk_core.config.entities.serializable import Serializable
        model = make_basic_entities()["model"]
        target = ctx.temp_dir / "model.json"
        model.save_json(target)
        loaded = Serializable.load_json(target)
        assert_equal(loaded.name, model.name)
        assert_equal(loaded.architecture.type, model.architecture.type)
        assert_equal(loaded.experiment.corpus.name, model.experiment.corpus.name)
        assert_equal(loaded.experiment.allowed_durations.durations, model.experiment.allowed_durations.durations)

    check(ctx, "Serializable.save_json/load_json conserva Model", save_and_load_json_roundtrip)

    def from_db_allowed_durations_parses_json():
        from transfolk_core.config.entities.allowed_durations import AllowedDurations
        row = {"id": 9, "name": "dur", "durations": json.dumps([0.5, 1.0]), "description": "x"}
        obj = AllowedDurations.from_db(row)
        assert_equal(obj.durations, [0.5, 1.0])
        assert_equal(obj.description, "x")

    check(ctx, "AllowedDurations.from_db parsea JSON", from_db_allowed_durations_parses_json)
