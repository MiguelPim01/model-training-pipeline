from src.infra.pipelines.no_test_split_pipeline import NoTestSplitPipeline
from src.infra.strategies.torch_strategy import TorchTrainingStrategy
from src.domain.templates.training_template import ITrainingTemplate
from src.infra.pipelines.csv_data_pipeline import CSVDataPipeline
from src.infra.schemas.model_config import ModelConfig


class DesinfoVacinalTemplate(ITrainingTemplate):
    """Defines the strategy and the pipeline for training Desinfo Vacinal project model.

    Args:
        ITrainingTemplate (_type_): Template base class for training models
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def run(self):
        data_pipeline = CSVDataPipeline(self.config)
        strategy = TorchTrainingStrategy(self.config)

        pipeline = NoTestSplitPipeline(strategy, data_pipeline)
        pipeline.run()
