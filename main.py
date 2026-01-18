from argparse import ArgumentParser

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

if __name__ == "__main__":
    main()
