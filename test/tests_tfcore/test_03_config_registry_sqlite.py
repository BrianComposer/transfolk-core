from __future__ import annotations

import sqlite3

from tests_tfcore.common import assert_equal, assert_true, check


def create_minimal_db(path):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE corpus (id INTEGER PRIMARY KEY, name TEXT NOT NULL, subcorpus TEXT, descripcion TEXT);
        CREATE TABLE tokenizer_algorithm (id INTEGER PRIMARY KEY, name TEXT NOT NULL, description TEXT);
        CREATE TABLE music_context (id INTEGER PRIMARY KEY, name TEXT NOT NULL, tonality TEXT, time_signature TEXT);
        CREATE TABLE allowed_durations (id INTEGER PRIMARY KEY, name TEXT NOT NULL, durations TEXT NOT NULL, description TEXT);
        CREATE TABLE transformer_architecture (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT, description TEXT,
            d_model INTEGER, n_heads INTEGER, n_layers INTEGER, d_ff INTEGER, dropout REAL, max_seq_len INTEGER,
            attention_type TEXT, activation TEXT, positional_encoding TEXT, layer_norm_eps REAL, bias INTEGER,
            weight_tying INTEGER, embedding_dropout REAL, residual_dropout REAL, attention_dropout REAL,
            initializer TEXT, rotary_dim INTEGER, encoder_layers INTEGER, decoder_layers INTEGER
        );
        CREATE TABLE runtime_train (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, epochs INTEGER, batch_size INTEGER, learning_rate REAL,
            weight_decay REAL, gradient_clip REAL, scheduler TEXT, warmup_steps INTEGER, accumulation_steps INTEGER,
            early_stopping INTEGER, patience INTEGER, save_every INTEGER, optimizer TEXT, loss TEXT
        );
        CREATE TABLE runtime_generate (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, temperature REAL, max_len INTEGER, num_productions INTEGER,
            top_k INTEGER, top_p REAL, repetition_penalty REAL, greedy INTEGER, seed INTEGER, device TEXT,
            mixed_precision INTEGER, num_workers INTEGER, deterministic INTEGER
        );
        CREATE TABLE experiment (
            id INTEGER PRIMARY KEY, name TEXT, id_corpus INTEGER, id_tk INTEGER, id_mc INTEGER, id_ad INTEGER, descripcion TEXT
        );
        CREATE TABLE model (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, id_ta INTEGER, id_exp INTEGER, id_rt INTEGER, description TEXT,
            train_start_time TEXT, train_end_time TEXT, train_total_time REAL, train_date TEXT, vocab_file TEXT
        );
        INSERT INTO corpus VALUES (1, 'unified-iberian', NULL, 'test corpus');
        INSERT INTO tokenizer_algorithm VALUES (1, 'momet', 'Mode-Metric tokenizer');
        INSERT INTO music_context VALUES (1, 'major_2_4', 'major', '2/4');
        INSERT INTO allowed_durations VALUES (1, 'standard', '[0.0, 0.25, 0.5, 1.0, 2.0]', 'standard durations');
        INSERT INTO transformer_architecture (id, name, type, d_model, n_heads, n_layers, d_ff, dropout, max_seq_len)
            VALUES (1, 'tiny_gpt', 'decoder_only_gpt', 16, 4, 1, 32, 0.0, 32);
        INSERT INTO runtime_train (id, name, epochs, batch_size, learning_rate, optimizer, loss)
            VALUES (1, 'rt', 1, 2, 0.001, 'adamw', 'cross_entropy');
        INSERT INTO runtime_generate (id, name, temperature, max_len) VALUES (1, 'rg', 1.0, 8);
        INSERT INTO experiment VALUES (1, 'exp_test', 1, 1, 1, 1, 'desc');
        INSERT INTO model VALUES (1, 'tiny_model', 1, 1, 1, 'desc', NULL, NULL, NULL, NULL, 'vocab.json');
        """
    )
    conn.commit()
    conn.close()


def run_tests(ctx):
    def registry_loads_minimal_database():
        from transfolk_core.db.config_registry import ConfigRegistry
        db = ctx.temp_dir / "transfolk_config_test.db"
        create_minimal_db(db)
        registry = ConfigRegistry(str(db))
        registry.load_all()
        assert_equal(len(registry.corpus), 1)
        assert_equal(len(registry.tokenizers), 1)
        assert_equal(len(registry.experiments), 1)
        assert_equal(len(registry.models), 1)
        assert_equal(registry.models[1].experiment.corpus.name, "unified-iberian")
        assert_equal(registry.models[1].architecture.type, "decoder_only_gpt")

    check(ctx, "ConfigRegistry.load_all carga DB mínima", registry_loads_minimal_database)

    def registry_find_by_name_and_update_model():
        from transfolk_core.db.config_registry import ConfigRegistry
        db = ctx.temp_dir / "transfolk_config_update_test.db"
        create_minimal_db(db)
        registry = ConfigRegistry(str(db))
        registry.load_all()
        obj = registry.find_by_name("tiny_model")
        assert_equal(obj.name, "tiny_model")
        obj.vocab_file = "new_vocab.json"
        obj.description = "updated"
        registry.update_model(obj)
        registry2 = ConfigRegistry(str(db))
        registry2.load_all()
        assert_equal(registry2.models[1].vocab_file, "new_vocab.json")
        assert_equal(registry2.models[1].description, "updated")

    check(ctx, "ConfigRegistry.find_by_name y update_model", registry_find_by_name_and_update_model)
