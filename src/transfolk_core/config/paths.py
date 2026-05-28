from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    root: Path
    projectName: str = "transfolk"


    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def corpora(self) -> Path:
        return self.data / "corpora"

    @property
    def data_tokenized(self) -> Path:
        return self.data / "tokenization"

    @property
    def data_features(self) -> Path:
        return self.data / "features"


    @property
    def experiments(self) -> Path:
        return self.root / "experiments" / self.projectName

    @property
    def models_training(self) -> Path:
        return self.experiments / "trained"

    @property
    def db_sqlite(self) -> Path:
        return self.root / "transfolk-core"/ "src" / "transfolk_core" / "db" / "transfolk_config.db"




#ESTOS HAY QUE REVISARLOS AUN
    @property
    def models(self) -> Path:
        return self.root / "models"


    @property
    def models_classifier(self) -> Path:
        return self.root / "models_classifier/sty_curves"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"
