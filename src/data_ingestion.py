import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.bucket_file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]
        
        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(f"Dataingestion started with {self.bucket_name} and file is {self.bucket_file_name}")
        
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.bucket_file_name)
            blob.download_to_filename(RAW_FILE_PATH)
            
            logger.info(f"Raw file is successfully downloaded to {RAW_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error while downloading the csv file ",  e)
            raise CustomException("Failed to download csv file", e)
            
    def run(self):
        try:
            logger.info("Starting data ingestion process")
            self.download_csv_from_gcp()
            logger.info("Data ingestion completed successfully")
        except Exception as e:
            logger.error(f"Error in data ingestion process: {str(e)}")
            raise CustomException("Data ingestion process failed", e)
        finally:
            logger.info("Data ingestion process finished")

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()