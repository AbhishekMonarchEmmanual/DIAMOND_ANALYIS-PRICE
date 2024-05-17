import pandas as pd 
import numpy as np 
from src.logger.logging import logging 
from src.exception.exception import customexception
from src.config.config import *
from src.config.artifacts import *
from src.utils.utils import save_data_to_csv
import os , sys 
from sklearn.model_selection import train_test_split
from pathlib import Path  
from dataclasses import dataclass
from datetime import datetime
import mlflow



class DataIngestion:
    def __init__(self, data_ingestion_config: Data_Ingestion_Config):
        logging.info("___WE ARE INITIATING DATA INGESTION CONFIGUTATION____")
        try:
            self.data_ingestion_config = data_ingestion_config
            
        except Exception as e:
            raise customexception(e,sys)
        logging.info("___END DATA INGESTION CONFIGUTATION____")   


    def initiate_data_ingestion(self):
        logging.info("___INTIATING INGESTION____")   

        try:

            train_data = pd.read_csv(self.data_ingestion_config.train_data)
            test_data = pd.read_csv(self.data_ingestion_config.test_data)
            # Log parameters and metrics
            
            train_file_path = save_data_to_csv(train_data, "train.csv", folder_path=self.data_ingestion_config.data_ingestion_dir)
            test_file_path = save_data_to_csv(test_data, "test.csv", folder_path=self.data_ingestion_config.data_ingestion_dir)
            
            # Log Artifacts 
            with mlflow.start_run(run_name='Data_Ingestion',nested=True):
                mlflow.log_param("train_data_shape", train_data.shape)
                mlflow.log_param("test_data_shape", test_data.shape)
                mlflow.log_artifact(train_file_path)
                mlflow.log_artifact(test_file_path)   
            
            logging.info("SHAPE OF TRAIN DATA : {train_data.shape}")
            logging.info("SHAPE OF TEST DATA : {test_data.shape}")
            logging.info("END OF DATA INGESTION")

            return DataIngestionArtifact(train_file_path=train_file_path, test_file_path=test_file_path)
     
        except Exception as e:
            raise customexception(e,sys) 
        





    
