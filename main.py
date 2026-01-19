from argparse import ArgumentParser
import logging
import sys

from src.infra.schemas.model_config import ModelConfig, parse_file
from src.templates.desinfo_vacinal_template import DesinfoVacinalTemplate

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s] %(message)s',
                    stream=sys.stdout)

def parse_args():
    parser = ArgumentParser(description="Model Training Pipeline")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file for training.",
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    config_file_path = args.config
    
    try:
        model_config = parse_file(config_file_path)
    except RuntimeError as e:
        print(f"Error loading configuration: {e}")
        return
    
    logging.info(f"Loaded model configuration: {model_config.model_name} v{model_config.version}")
    
    template = DesinfoVacinalTemplate(config=model_config)
    template.run()

if __name__ == "__main__":
    main()
