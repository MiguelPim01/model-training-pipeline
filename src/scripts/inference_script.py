"""
Script that runs model inference for the database on PostgreSQL.

Basic workflow:
1. Load model configuration and initialize the inference use case (loads model from s3 bucket behind the scenes).
2. While True:
    a. Fetch posts from the database where predicted_pa is NULL within the specified time range.
    b. Run inference on the fetched posts using the loaded model.
    c. Update the database with the predicted_pa values for the processed posts.

The script also sends health status information to the inference API through a TCP socket connection, 
which can be used for monitoring and alerting purposes.
"""

import os
import sys
import json
import logging
import psycopg

from time import sleep
from socket import socket
from dotenv import load_dotenv
from argparse import ArgumentParser

from src.infra.schemas.model_config import parse_file
from src.infra.use_cases.torch_inference_use_case import TorchInferenceUseCase


# ----- Configuration -----
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s - %(message)s]',
                    stream=sys.stdout)

load_dotenv()

DB_URL = os.getenv("DB_URL")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 15))
ROWS_LIMIT = int(os.getenv("ROWS_LIMIT", 2000))
TIME_RANGE_DAYS = int(os.getenv("TIME_RANGE_DAYS", 365))
WAITING_TIME_SECONDS = int(os.getenv("WAITING_TIME_SECONDS", 60 * 30)) # 30 minutes
# -----

# ----- Basic Checks -----
if DB_URL is None: # Check if DB_URL is set
    logging.error("DB_URL environment variable is not set. Please set it in the .env file.")
    sys.exit(1)
else: # Test the connection
    try:
        with psycopg.connect(DB_URL) as remote_session:
            pass
    except Exception as e:
        logging.error(f"Failed to connect to the database: {e}")
        sys.exit(1)
# -----

# ----- Global variables -----
healthy_classification = True
healthy_update = True
# -----

def select_data():
    """
    Fetch posts from the database where predicted_pa is NULL within the specified time range.
    
    Returns:
        posts: A list of dictionaries with keys: id, message, predicted_pa or an empty list.
    """
    global healthy_classification
    
    query = f"""
        SELECT "id", "message", "predicted_pa"
        FROM "Post" p
        WHERE p."predicted_pa" IS NULL AND p."time" >= CURRENT_DATE - INTERVAL '1 day' * {TIME_RANGE_DAYS}
        LIMIT {ROWS_LIMIT};
    """
    posts = list()
    
    try:
        with psycopg.connect(conninfo=DB_URL) as remote_session:
            with remote_session.cursor() as cursor:
                try:
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    
                    posts = [{"id": row[0], "message": row[1], "predicted_pa": row[2]} for row in rows]
                    logging.info(f"Fetched {len(posts)} posts for inference.")
                    
                    healthy_classification = True
                    return posts
                except Exception as e:
                    logging.error(f"Error while executing query: {e}")
                    healthy_classification = False
    except Exception as e:
        logging.error(f"Error while fetching posts: {e}")
        healthy_classification = False

    return posts

def update_rows(values: list[tuple[int, float]]) -> bool:
    """
    Update the posts in the database with the predicted_pa values.
    
    Parameters:
        values: List of tuples containing post IDs and their corresponding predicted_pa values (id, predicted_pa).
        
    Returns:
        values_sent (bool): True if the update was successful, False otherwise.
    """
    global healthy_update
    
    update_query = """
        UPDATE "Post"
        SET "predicted_pa" = %s
        WHERE "Post".id = %s;
    """
    
    values_sent = False

    try:
        with psycopg.connect(conninfo=DB_URL) as remote_session:
            with remote_session.cursor() as cursor:
                try:
                    cursor.executemany(update_query, values) # Runs multiple update queries
                    
                    healthy_update = True
                except Exception as e:
                    logging.error(f"Error while executing update query: {e}")
                    
                    remote_session.rollback()
                    healthy_update = False
                    
                    return values_sent  
                
                remote_session.commit()
            
            values_sent = True
    except Exception as e:
        healthy_update = False
        logging.error(f"Error while sending results to remote: {e}")

    return values_sent

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
    logging.info("Starting inference script...")
    
    # Parse command-line arguments
    args = parse_args()
    config_file_path = args.config
    
    # Uses TCP socket connection and IPv4 addresses to connect with the inference API
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((("localhost", 9999)))
    
    # Creates the inference use case instance
    model_config = parse_file(config_file_path)
    inference_use_case = TorchInferenceUseCase(config=model_config)
    
    global healthy_classification
    global healthy_update
    
    curr_cycle = 1
    processed_posts = 0
    while True:
        # Sending health status to the API
        s.sendall(json.dumps({"healthy_classification": healthy_classification, "healthy_update": healthy_update}).encode('utf-8'))
        
        logging.info(f"Starting process cycle {curr_cycle}...")
        
        # Selecting data from the database
        posts = select_data()
        num_posts = len(posts)
        
        if num_posts == 0:
            logging.info(f"No posts to process, or query failed to retrieve the total number of posts. Waiting {WAITING_TIME_SECONDS / 60 : .02f} minutes before next check...")
            
            curr_cycle += 1
            sleep(WAITING_TIME_SECONDS)
            
            continue
            
        logging.info(f"Processing {num_posts} posts...")
        
        messages = [post["message"] for post in posts]
        
        # Running inference on the fetched posts
        logging.info("Running inference on the fetched posts...")
        try:
            predictions = inference_use_case.predict_proba_batch(texts=messages, batch_size=BATCH_SIZE)
            healthy_classification = True
        except Exception as e:
            logging.error(f"Error while running inference: {e}")
            
            curr_cycle += 1
            healthy_classification = False
            
            continue
        logging.info("Inference completed successfully.")
        
        # Preparing values for database update
        values = [(p, r["id"]) for r, p in zip(posts, predictions)] # the order is important for the update query, where predicted_pa comes first
        
        # Updating the database with the predicted_pa values
        logging.info("Updating the database with the predicted_pa values...")
        
        values_sent = update_rows(values)
        if not values_sent:
            logging.warning("Failed to update the database with the predicted_pa values.")
            continue
        
        logging.info(f"Database updated successfully with predicted_pa values for {num_posts} posts.")
        
        processed_posts += num_posts
        logging.info(f"Total processed posts so far: {processed_posts}")
        
        sleep(3) # Avoid overwhelming the database
        
        logging.info(f"Cycle {curr_cycle} completed.")
        curr_cycle += 1

if __name__ == "__main__":
    main()