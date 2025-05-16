from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from src.custom_exception import CustomException
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from utils.common_functions import read_yaml
from src.logger import get_logger
from config.paths_config import *
import pandas as pd
import numpy as np

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, config):
        self.config = config
        self.y_columns = self.config["data_processing"]["feature_names"]["y_columns"]
        
    def read_csv_file(self, path) -> pd.DataFrame:
        try:
            logger.info("Loading data")
            return pd.read_csv(path)
        except Exception as e:
            logger.error(f"Error loading data from {path}")
            raise CustomException(f"Failed to load data from {path}", e)
    
    def ohe_to_le(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Converting ohe to le")
            y = pd.DataFrame(df[self.y_columns].values.argmax(axis=1), columns=["labels"])
            # label_mapping = {i: col for i, col in enumerate(self.y_columns)}
            df = pd.concat([df.drop(self.y_columns, axis=1), y], axis=1)
            return df
        except Exception as e:
            logger.error(f"Error converting ohe to le")
            raise CustomException(f"Failed to convert ohe to le", e)
        
    def check_multicollinearity(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Checking multicollinearity")
            X = add_constant(df)
            y = df["labels"]
            vif_data = pd.DataFrame()
            vif_data["feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            
            filtered_vif_df = vif_data[
                (vif_data['VIF'].replace(np.inf, np.nan) <= 10)
            ].dropna()
            
            clean_features = filtered_vif_df['feature'].tolist()
            
            if 'const' in clean_features:
                clean_features.remove('const')
                
            df = df[clean_features]
            
            logger.info("Checked multicollinearity")
            return df
            
        except Exception as e:
            logger.error(f"Error checking multicollinearity")
            raise CustomException(f"Failed to check multicollinearity", e)
        
    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Balancing dataset")
        X = df.drop("labels", axis=1)
        y = df["labels"]
        
        smote = SMOTE(random_state=42)
        
        X_res, y_res = smote.fit_resample(X, y)
        
        balanced_df = pd.DataFrame(X_res, columns=X.columns)
        
        balanced_df["labels"] = y_res
        
        logger.info("Balanced dataset")
        return balanced_df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Selecting features")
        X = df.drop(columns=["labels"])
        y = df["labels"]
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        feature_importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({"feature": X.columns, "importance": feature_importance})
        top_features_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)
        top_features = top_features_importance_df["feature"].head(6).values
        top_df = df[top_features.tolist() + ["labels"]]
        logger.info("Selected features")
        return top_df

    def save_data(self, df: pd.DataFrame, train_path: str, test_path: str) -> None:
        try:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            logger.info("Saving data")
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data to {os.path.dirname(train_path)}")
            raise CustomException(f"Failed to save data to {os.path.dirname(train_path)}", e)
    
    def run(self):
        data = self.read_csv_file(RAW_FILE_PATH)
        
        data = self.ohe_to_le(data)
        
        data = self.check_multicollinearity(data)
        
        data = self.balance_dataset(data)
        
        data = self.select_features(data)
        
        self.save_data(data, PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH)
        
if __name__ == "__main__":
    data_processing = DataProcessing(read_yaml(CONFIG_PATH))
    data_processing.run()

    
