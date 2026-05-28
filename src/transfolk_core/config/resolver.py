from pathlib import Path
from datetime import datetime

from transfolk_core.config import RuntimeGenerate
from transfolk_core.config import Corpus
from transfolk_core.config import Experiment
from transfolk_core.config import TransformerArchitecture
from transfolk_core.config import Model
from transfolk_core.config.paths import ProjectPaths


class PathResolver:

    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    # --------------------
    # Helpers
    # --------------------

    @staticmethod
    def _safe(value: str) -> str:
        if not value:
            return "x"
        return (
            value.replace("/", "_")
                 .replace(" ", "")
        )

    @staticmethod
    def temp_slug(temperature: float) -> str:
        return f"{temperature:.1f}"

    # --------------------
    # DATA
    # --------------------
    def data_raw(self, corpus: Corpus) -> Path:
        if corpus.subcorpus is None:
            return self.paths.corpora / corpus.name / "raw"
        else:
            return self.paths.corpora / f"{corpus.name}" / f"{corpus.subcorpus}" / "raw"

    def data_clean(self, corpus: Corpus) -> Path:
        if corpus.subcorpus is None:
            return self.paths.corpora / f"{corpus.name}" / "cleaned"
        else:
            return self.paths.corpora / f"{corpus.name}" / f"{corpus.subcorpus}" / "cleaned"

    def data_report(self, corpus: Corpus) -> Path:
        if corpus.subcorpus is None:
            return self.paths.corpora / f"{corpus.name}" / "reports"
        else:
            return self.paths.corpora / f"{corpus.name}" / f"{corpus.subcorpus}" / "reports"

    def data_normalized(self, corpus: Corpus) -> Path:
        if corpus.subcorpus is None:
            return self.paths.corpora / f"{corpus.name}" / "normalized"
        else:
            return self.paths.corpora / f"{corpus.name}" / f"{corpus.subcorpus}" / "normalized"

    def data_mid(self, corpus: Corpus) -> Path:
        if corpus.subcorpus is None:
            return self.paths.corpora / f"{corpus.name}" / "mid"
        else:
            return self.paths.corpora / f"{corpus.name}" / f"{corpus.subcorpus}" / "mid"

    def data_metadata(self, corpus: Corpus) -> Path:
        if corpus.subcorpus is None:
            return self.paths.corpora / f"{corpus.name}" / "metadata"
        else:
            return self.paths.corpora / f"{corpus.name}" / f"{corpus.subcorpus}" / "metadata"



    def data_token(self, corpus: Corpus) -> Path:
        if corpus.subcorpus is None:
            return self.paths.data_tokenized / f"{corpus.name}"
        else:
            return self.paths.data_tokenized / f"{corpus.name}" / f"{corpus.subcorpus}"

    def tokenize_dir(self, exp: Experiment) -> Path:
        if exp.corpus.subcorpus is None:
            return self.paths.data_tokenized / exp.tokenizer.name / exp.corpus.name
        else:
            return (self.paths.data_tokenized
                    / exp.tokenizer.name /
                    exp.corpus.name /
                    f"{exp.corpus.subcorpus}")




    # --------------------
    # TRAINING
    # --------------------
    def train_dir(self, arch: TransformerArchitecture, exp: Experiment) -> Path:
        if exp.corpus.subcorpus is None:
            return (
                    self.paths.models_training
                    / arch.name
                    / exp.corpus.name
                    / exp.tokenizer.name
            )
        else:
            return (
                    self.paths.models_training
                    / arch.name
                    / exp.corpus.name
                    / f"{exp.corpus.subcorpus}"
                    / exp.tokenizer.name
            )





    # --------------------
    # MODELS RELEASED
    # --------------------

    def models_released_dir(self) -> Path:
        return self.paths.models / "released"

    # --------------------
    # PRODUCTIONS
    # --------------------

    def production_dir_all(self) -> Path:
        return self.paths.experiments / "generation"

    def production_dir(self, model: Model, runtime: RuntimeGenerate) -> Path:
        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)

        return (
            self.production_dir_all()
            / model.architecture.name
            / model.experiment.corpus.name
            / model.experiment.tokenizer.name
            / ton
            / ts
            / self.temp_slug(runtime.temperature)
        )

    def production_sty_dir(self, model: Model, runtime: RuntimeGenerate) -> Path:
        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)

        return (
            self.sty_curves_dir()
            / "generation"
            / model.architecture.name
            / model.experiment.corpus.name
            / model.experiment.tokenizer.name
            / ton
            / ts
            / self.temp_slug(runtime.temperature)
        )

    def sty_curves_dir(self) -> Path:
        return self.paths.experiments / "sty_curves"

    def sty_curves_classifier_dir(self, corpus: Corpus) -> Path:
        return self.sty_curves_dir() / "classifier" / corpus.name

    # --------------------
    # FILES
    # --------------------

    def sequences_file(self, exp: Experiment) -> Path:
        ts = self._safe(exp.music_context.time_signature)
        ton = self._safe(exp.music_context.tonality)
        filename = f"sequences_{exp.corpus.name}_{exp.tokenizer.name}_{ton}_{ts}.json"
        return self.tokenize_dir(exp) / filename

    def vocab_file(self, exp: Experiment) -> Path:
        ts = self._safe(exp.music_context.time_signature)
        ton = self._safe(exp.music_context.tonality)
        filename = f"vocab_{exp.corpus.name}_{exp.tokenizer.name}_{ton}_{ts}.json"
        return self.tokenize_dir(exp) / filename

    def token_errors_file(self, exp: Experiment) -> Path:
        ts = self._safe(exp.music_context.time_signature)
        ton = self._safe(exp.music_context.tonality)
        filename = f"errors_{exp.corpus.name}_{exp.tokenizer.name}_{ton}_{ts}.json"
        return self.tokenize_dir(exp) / filename


    def loss_log_file(self, model: Model) -> Path:
        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)
        filename = f"training_loss_{model.experiment.corpus.name}_{model.experiment.tokenizer.name}_{ton}_{ts}.json"
        return self.train_dir(model.architecture, model.experiment) / filename


    def model_file(self, model: Model) -> Path:
        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)
        filename = (
            f"{model.architecture.name}_"
            f"{model.experiment.corpus.name}_"
            f"{model.experiment.tokenizer.name}_"
            f"{ton}_"
            f"{ts}"
        )
        return self.train_dir(model.architecture, model.experiment) / f"{filename}.pt"

    def model_file_epoch(self, model: Model, epoch:int) -> Path:
        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)
        filename = (
            f"{model.architecture.name}_"
            f"{model.experiment.corpus.name}_"
            f"{model.experiment.tokenizer.name}_"
            f"{ton}_"
            f"{ts}_"
            f"epoch{epoch}"
        )
        return self.train_dir(model.architecture, model.experiment) / f"{filename}.pt"

    def model_cfg_file(self, model: Model) -> Path:
        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)
        filename = (
            f"{model.architecture.name}_"
            f"{model.experiment.corpus.name}_"
            f"{model.experiment.tokenizer.name}_"
            f"{ton}_"
            f"{ts}"
        )
        return self.train_dir(model.architecture, model.experiment) / f"{filename}.json"

    def model_epoch_cfg_file(self, model: Model, epoch:int) -> Path:
        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)
        filename = (
            f"{model.architecture.name}_"
            f"{model.experiment.corpus.name}_"
            f"{model.experiment.tokenizer.name}_"
            f"{ton}_"
            f"{ts}_"
            f"epoch{epoch}"
        )
        return self.train_dir(model.architecture, model.experiment) / f"{filename}.json"

    def model_snapshot_file(self, arch: TransformerArchitecture, exp: Experiment, model_id: str) -> Path:
        """
        Ruta del snapshot JSON de un modelo entrenado dentro del directorio de entrenamiento.

        El snapshot describe el modelo, su arquitectura, experimento, tokenizer,
        runtime y demás metadata necesaria para reconstruirlo o exportarlo.
        """
        filename = f"{model_id}.json"
        return self.train_dir(arch, exp) / filename

    def generated_new_file(self, model: Model, runtime: RuntimeGenerate) -> Path:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)
        filename = f"{model.architecture.name}_{model.experiment.tokenizer.name}_{ts}_{ton}_{timestamp}.musicxml"
        return self.production_dir(model, runtime) / filename

    def generated_new_file_sty(self, model: Model, runtime: RuntimeGenerate) -> Path:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        ts = self._safe(model.experiment.music_context.time_signature)
        ton = self._safe(model.experiment.music_context.tonality)
        filename = f"{model.architecture.name}_{model.experiment.tokenizer.name}_{ts}_{ton}_{timestamp}.musicxml"
        return self.production_sty_dir(model, runtime) / filename

    # --------------------
    # CLASSIFIER
    # --------------------
    def classifier_dir(self, corpus: Corpus) -> Path:
        return self.paths.models_classifier / corpus.name

    def corpus_ts_mode_report_file(self, corpus: Corpus) -> Path:
        return self.data_metadata(corpus) / "ts_mode_distribution.json"
    def corpus_ts_mode_report_file_norm(self, corpus: Corpus) -> Path:
        return self.data_metadata(corpus) / "ts_mode_distribution_normalized.json"



    # --------------------
    # CHARTS
    # --------------------

    def charts_dir(self) -> Path:
        return self.paths.experiments / "charts"

    def classifier_experiments_dir(self) -> Path:
        return self.paths.experiments / "classifier"

