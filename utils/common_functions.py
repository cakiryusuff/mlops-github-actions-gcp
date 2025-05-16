import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
import pandas as pd

logger = get_logger(__name__)

def read_yaml(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found at {file_path}")
        
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info("-"*50)
            
            logger.info("Successfully read YAML file")
            return config
    except Exception as e:
        logger.error("Error reading YAML file")
        raise CustomException("Failed to read YAML file", e)

def load_data(path):
    try:
        logger.info("Loading data")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading data from {path}")
        raise CustomException(f"Failed to load data from {path}", e)