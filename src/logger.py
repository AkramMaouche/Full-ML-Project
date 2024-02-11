import logging
import os
from datetime import datetime

# Logger Configuration
#     filename=LOG_FILE_PATH,
#     format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
#     level=logging.INFO,


# create a logs directory if it does not exist
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

# create a log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# create a logger
logger = logging.getLogger(__name__)

# create a file handler
file_handler = logging.FileHandler(LOG_FILE_PATH)

# create a logging format
formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# add the file handler to the logger
logger.addHandler(file_handler)

# set the logging level
logger.setLevel(logging.INFO)
