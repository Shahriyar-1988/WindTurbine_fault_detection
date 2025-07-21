import os
import logging

# Define logging format
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Define the log directory and log file path
log_dir = "logs"
file_path = os.path.join(log_dir, "logsheet.log")

# Create the logs directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(file_path),  # Log messages to a file
        logging.StreamHandler()  # Print logs to the console
    ]
)

# Create a logger
logger = logging.getLogger("Projectlogger")

if __name__ == "__main__":
    logger.info("This is a test message!")