from src.pipelines.desinfo_vacinal_pipeline import DesinfoVacinalPipeline
from src.templates.training_template import TrainingTemplate
from src.infra.schemas.model_config import ModelConfig
from src.strategies.desinfo_vacinal_strategy import DesinfoVacinalStrategy

class DesinfoVacinalTemplate(TrainingTemplate):
    """Defines the strategy and the pipeline for training Desinfo Vacinal project model.

    Args:
        TrainingTemplate (_type_): Template base class for training models
    """
    def __init__(self, config: ModelConfig, num_runs: int = 1):
        super().__init__()
        self.config = config
        self.num_runs = num_runs
        self.pipeline = None
    
    def run(self):
        strategy = DesinfoVacinalStrategy(self.config)
        self.pipeline = DesinfoVacinalPipeline(strategy, self.num_runs)
        
        self.pipeline.run()