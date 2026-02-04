from tqdm import tqdm
import logging
import sys

from src.domain.strategies.training_strategy import ITrainingStrategy
from src.domain.pipelines.training_pipeline import ITrainingPipeline
from src.domain.pipelines.data_pipeline import IDataPipeline

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    stream=sys.stdout)

class NoTestSplitPipeline(ITrainingPipeline):
    """Generic Pipeline for training models
    """
    
    def __init__(self, strategy: ITrainingStrategy, data_pipeline: IDataPipeline, num_runs: int = 30):
        self.strategy = strategy
        self.data_pipeline = data_pipeline
        self.num_runs = num_runs
    
    def run(self):
        logging.info(f"Starting training pipeline with {self.num_runs} runs")
        
        # Execute setup steps
        data = self.data_pipeline.load()
        
        seeds = self._get_seeds()
        
        # Train model
        for run in tqdm(range(self.num_runs), desc="Training runs", unit="run"):
            # Load dataset first (sets num_labels)
            self.strategy.load_dataset(data, seeds[run])
            self.strategy.load_datamodel()
            
            # Build model (needs num_labels from load_dataset)
            self.strategy.build_model()
            
            # Train
            self.strategy.train()
            
            # Evaluate model
            self.strategy.evaluate()
        
        self.strategy.deploy()
    
    def _get_seeds(self):
        """Fix the seeds for reproducibility

        Returns:
            seeds (list[int]): List of fixed seeds for reproducibility
        """
        seeds = [
            2911491036, 363228150, 3051923112, 1952483715, 2692766584, 2587916052, 394603965, 
            272074613, 852185489, 1999207708, 3354547254, 3532630342, 3154617754, 2369800138, 
            998672754, 452237136, 2549661715, 1206783379, 2759072755, 461076070, 1730103807, 
            3816081947, 2754467631, 1539470218, 2346164233, 2335657277, 3485442286, 474710556, 
            1257135398, 4213807470
        ]
        
        return seeds