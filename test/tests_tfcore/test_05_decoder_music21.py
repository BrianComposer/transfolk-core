from __future__ import annotations

from tests_tfcore.common import assert_equal, assert_true, check, require_module


def run_tests(ctx):
    def decoder_builds_music21_score_from_structural_tokens():
        require_module("music21")
        from music21 import stream
        from transfolk_core.tokenization.decoder import tokens_to_music21_stream
        tokens = [
            "START", "TS_2/4", "MODE_major",
            "BAR", "NOTE_ON_60", "DUR_1.0", "REST", "DUR_1.0",
            "BAR", "NOTE_ON_62", "DUR_0.5", "NOTE_ON_64", "DUR_0.5", "REST", "DUR_1.0",
            "END",
        ]
        score = tokens_to_music21_stream(tokens, allowed_durations=[0.0, 0.5, 1.0, 2.0], verbose_warnings=False)
        assert_true(isinstance(score, stream.Score), "El decoder debe devolver music21.stream.Score.")
        measures = list(score.parts[0].getElementsByClass(stream.Measure))
        assert_equal(len(measures), 2)
        assert_equal(len(measures[0].notesAndRests), 2)
        assert_equal(measures[0].timeSignature.ratioString, "2/4")

    check(ctx, "decoder: tokens estructurales -> Score con compases", decoder_builds_music21_score_from_structural_tokens)

    def decoder_ignores_or_warns_invalid_tokens_without_crashing():
        require_module("music21")
        from music21 import stream
        from transfolk_core.tokenization.decoder import tokens_to_music21_stream
        tokens = ["START", "TS_2/4", "BAR", "UNKNOWN", "NOTE_ON_60", "DUR_1.0", "DUR_1.0", "END"]
        score = tokens_to_music21_stream(tokens, allowed_durations=[1.0], verbose_warnings=False)
        assert_true(isinstance(score, stream.Score), "El decoder no debe romper con tokens desconocidos o DUR colgantes.")

    check(ctx, "decoder: robustez ante tokens inválidos", decoder_ignores_or_warns_invalid_tokens_without_crashing)

    def simple_decoder_with_ts_builds_part():
        require_module("music21")
        from music21 import stream
        from transfolk_core.tokenization.decoder import tokens_to_music21_stream_with_ts
        tokens = ["NOTE_ON_60", "DUR_1.0", "REST", "DUR_1.0"]
        part = tokens_to_music21_stream_with_ts(tokens, time_signature="2/4", allowed_durations=[1.0])
        assert_true(isinstance(part, stream.Part), "tokens_to_music21_stream_with_ts debe devolver stream.Part.")
        assert_equal(len(part.notesAndRests), 2)

    check(ctx, "decoder: tokens_to_music21_stream_with_ts", simple_decoder_with_ts_builds_part)
