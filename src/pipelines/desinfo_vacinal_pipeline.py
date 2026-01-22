from tqdm import tqdm
import logging
import sys

from src.strategies.training_strategy import TrainingStrategy
from src.infra.schemas.model_config import ModelConfig

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    stream=sys.stdout)

class DesinfoVacinalPipeline:
    """Generic Pipeline for training models
    """
    
    def __init__(self, strategy: TrainingStrategy, num_runs: int = 1):
        self.strategy = strategy
        self.num_runs = num_runs
    
    def run(self):
        logging.info(f"Starting training pipeline with {self.num_runs} runs")
        
        # Execute setup steps
        self.strategy.preprocess_data()
        
        seeds = self.get_seeds()
        
        # Train model
        for run in tqdm(range(self.num_runs), desc="Training runs", unit="run"):
            # Build model
            self.strategy.build_model()
            
            self.strategy.load_dataset(seeds[run])
            self.strategy.load_datamodel()
            
            # Train
            self.strategy.train()
            
            # Evaluate model
            self.strategy.evaluate()
        
        # Save metrics from the last evaluation
        self.strategy.save_metrics()
    
    def get_seeds(self):
        seeds = [
            2911491036, 363228150, 3051923112, 1952483715, 2692766584, 2587916052, 394603965, 
            272074613, 852185489, 1999207708, 3354547254, 3532630342, 3154617754, 2369800138, 
            998672754, 452237136, 2549661715, 1206783379, 2759072755, 461076070, 1730103807, 
            3816081947, 2754467631, 1539470218, 2346164233, 2335657277, 3485442286, 474710556, 
            1257135398, 4213807470
        ]
        
        return seeds