import os
import pandas as pd
import joblib
from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
import mlflow
import mlflow.sklearn
import time

logger = get_logger(__name__)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(f"{int(time.time() * 1000)}-my-experiment")

class ModelTraining:
    def __init__(self, config, train_path, test_path, model_output_path):
        self.config = config
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.random_search_params = self.config["model_training"]["random_search_params"]
        self.params_dist = param_dist
        
    def load_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)
            
            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)
            
            X_train = train_df.drop(columns=['labels'])
            y_train = train_df['labels']
            
            X_test = test_df.drop(columns=['labels'])
            y_test = test_df['labels']
            
            logger.info("Data loaded and split into features and target.")
            
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error loading and splitting data: {e}")
            raise CustomException("Failed to load and split data", e)
        
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Initiating LightGBM model")
            
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params["random_state"])
            
            logger.info("Starting Randomized Search for hyperparameter tuning")
            
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                cv=self.random_search_params["cv"],
                verbose=self.random_search_params["verbose"],
                n_jobs=self.random_search_params["n_jobs"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"]
            )
            
            logger.info("Starting Hyperparameter tuning")
            random_search.fit(X_train, y_train)
            logger.info("Hyperparameter tuning completed successfully!")
            
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            
            logger.info(f"Best parameters found: {best_params}")
            
            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error in training LightGBM model: {e}")
            raise CustomException("Failed to train LightGBM model", e)
        
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model performance")
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Accuracy: {accuracy}")

            return {
                "accuracy": accuracy
            }
            
        except Exception as e:
            logger.error(f"Error while evaluating model: {e}")
            raise CustomException("Error while evaluating model", e)
        
    
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info(f"Saving model to {self.model_output_path}")
            joblib.dump(model, self.model_output_path)
            logger.info("Model saved successfully!")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise CustomException("Failed to save model", e)
        

    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting model training process")
                
                logger.info("Starting our MLFLOW experiment")
                
                logger.info("Logging the training and testing dataset to MLFLOW")
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")
                
                X_train, y_train, X_test, y_test = self.load_data()
                
                model = self.train_lgbm(X_train, y_train)
                
                metrics = self.evaluate_model(model, X_test, y_test)
                
                self.save_model(model)
                
                logger.info("Logging model to MLFLOW")
                mlflow.log_artifact(self.model_output_path, artifact_path="models")
                
                logger.info("Logging model parameters and metrics to MLFLOW")
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics=metrics)
                
                logger.info("Model training process completed successfully!")
        except Exception as e:
            logger.error(f"Error in model training process: {e}")
            raise CustomException("Model training process failed", e)

if __name__ == "__main__":
    trainer = ModelTraining(
        read_yaml(CONFIG_PATH),
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    trainer.run()