from config.paths_config import CONFIG_PATH, MODEL_OUTPUT_PATH, PROCESSED_TEST_DATA_PATH, PROCESSED_TRAIN_DATA_PATH
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.model_training import ModelTraining
from utils.common_functions import read_yaml


if __name__ == "__main__":
    # data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    # data_ingestion.run()
    
    data_processing = DataProcessing(read_yaml(CONFIG_PATH))
    data_processing.run()
    
    trainer = ModelTraining(
        read_yaml(CONFIG_PATH),
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    trainer.run()