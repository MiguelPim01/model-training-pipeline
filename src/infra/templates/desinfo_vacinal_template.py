from infra.pipelines.no_test_split_pipeline import NoTestSplitPipeline
from infra.strategies.torch_strategy import TorchTrainingStrategy
from domain.templates.training_template import ITrainingTemplate
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
        strategy = TorchTrainingStrategy(self.config)
        
        pipeline = NoTestSplitPipeline(strategy)
        pipeline.run()