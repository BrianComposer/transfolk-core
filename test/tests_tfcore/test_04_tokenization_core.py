from __future__ import annotations

from tests_tfcore.common import assert_equal, assert_true, check, require_module


def run_tests(ctx):
    def vocabulary_roundtrip_is_stable():
        require_module("music21")
        from transfolk_core.tokenization.tokenizer import build_vocabulary, ids_to_tokens, tokens_to_ids
        seqs = [
            ["TS_2/4", "MODE_major", "BAR", "NOTE_ON_60", "DUR_1.0", "END"],
            ["TS_2/4", "MODE_major", "BAR", "REST", "DUR_0.5", "END"],
        ]
        vocab = build_vocabulary(seqs)
        assert_equal(vocab["PAD"], 0)
        assert_equal(vocab["START"], 1)
        assert_equal(vocab["END"], 2)
        ids = tokens_to_ids(seqs[0], vocab)
        inv = {v: k for k, v in vocab.items()}
        assert_equal(ids_to_tokens(ids, inv), seqs[0])

    check(ctx, "tokenizer: vocabulario y roundtrip tokens/ids", vocabulary_roundtrip_is_stable)

    def closest_duration_selects_nearest_allowed_value():
        require_module("music21")
        from transfolk_core.tokenization.tokenizer import closest_duration
        allowed = [0.25, 0.5, 1.0, 2.0]
        assert_equal(closest_duration(0.49, allowed), 0.5)
        assert_equal(closest_duration(1.76, allowed), 2.0)

    check(ctx, "tokenizer: closest_duration", closest_duration_selects_nearest_allowed_value)

    def pulse_beat_handles_simple_and_compound_meters():
        require_module("music21")
        from music21 import meter
        from transfolk_core.tokenization.tokenizer import compute_pulse_beat
        assert_equal(compute_pulse_beat(meter.TimeSignature("2/4")), 1.0)
        assert_equal(compute_pulse_beat(meter.TimeSignature("6/8")), 1.5)

    check(ctx, "tokenizer: compute_pulse_beat simple/compuesto", pulse_beat_handles_simple_and_compound_meters)

    def validate_measure_duration_detects_valid_and_invalid():
        require_module("music21")
        from music21 import note, stream
        from transfolk_core.tokenization.tokenizer import validate_measure_duration
        m = stream.Measure(number=1)
        for pitch in [60, 62]:
            n = note.Note(pitch)
            n.quarterLength = 1.0
            m.append(n)
        valid, beat_sum, diff = validate_measure_duration(m, expected_length=2.0, allowed_durations=[0.5, 1.0, 2.0])
        assert_true(valid, f"El compás debería ser válido: beat_sum={beat_sum}, diff={diff}")
        valid, _, _ = validate_measure_duration(m, expected_length=3.0, allowed_durations=[0.5, 1.0, 2.0])
        assert_true(not valid, "El compás debería ser inválido frente a expected_length=3.0")

    check(ctx, "tokenizer: validate_measure_duration", validate_measure_duration_detects_valid_and_invalid)
