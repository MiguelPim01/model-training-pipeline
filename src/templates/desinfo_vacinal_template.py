import logging
import sys

from src.pipelines.training_pipeline import TrainingPipeline
from src.templates.training_pipeline_template import TrainingPipelineTemplate
from src.infra.schemas.model_config import ModelConfig
from src.strategies.desinfo_vacinal_strategy import DesinfoVacinalStrategy

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    stream=sys.stdout)

class DesinfoVacinalTemplate(TrainingPipelineTemplate):
    def __init__(self, config: ModelConfig, num_runs: int = 1):
        super().__init__()
        self.config = config
        self.num_runs = num_runs
        self.pipeline = None
    
    def run(self):
        strategy = DesinfoVacinalStrategy(self.config)
        
        self.pipeline = TrainingPipeline(strategy, self.num_runs)
        self.pipeline.run()