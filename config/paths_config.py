import os
from scipy.stats import randint, uniform

######################################## DATA INGESTION

RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

CONFIG_PATH = "config/config.yaml"

########################################## DATA PROCESSING
PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_test.csv")

########################################### MODEL TRAINING
MODEL_OUTPUT_PATH = "artifacts/models/lgbm_model.pkl"

param_dist = {
    'num_leaves': randint(20, 150),
    'max_depth': randint(3, 20),
    'learning_rate': uniform(0.01, 0.3),  # 0.01 ila 0.31 arası
    'n_estimators': randint(50, 500),
    'min_child_samples': randint(5, 30),
    'subsample': uniform(0.5, 0.5),       # 0.5 ila 1.0 arası
    'colsample_bytree': uniform(0.5, 0.5),
    'reg_alpha': uniform(0, 1.0),
    'reg_lambda': uniform(0, 1.0)
}
