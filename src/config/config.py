import pandas as pd 
import numpy as np 
from src.config.artifacts import *
from src.logger.logging import logging 
from src.exception.exception import customexception
from src.utils.utils import save_data_to_csv
import os , sys 
from sklearn.model_selection import train_test_split
from pathlib import Path  
from dataclasses import dataclass
from datetime import datetime


class Training_Pipeline_Config:

    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        
        except Exception  as e:
            raise customexception(e,sys)     


class Data_Ingestion_Config:
    try:
        def __init__(self, training_pipeline_config:Training_Pipeline_Config):

            logging.info("___DATA__INGESTION__CONFIG___INITIATED___")

            self.train_data = "https://raw.githubusercontent.com/AbhishekMonarchEmmanual/mlops_project1/main/experiments/data/train.csv"
            self.test_data  = "https://raw.githubusercontent.com/AbhishekMonarchEmmanual/mlops_project1/main/experiments/data/test.csv"
            self.train_name = "train_data.csv"
            self.test_name = "test_data.csv"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, "Data_Ingestion")

            logging.info("____DATA CONFIG ENDED____")

        def __str__(self):
            return str(self.__dict__)

    except Exception as e:
        raise customexception(e,sys)


 
class Data_Transformation_Config:
    def __init__(self, training_pipeline_config: Training_Pipeline_Config, data_ingestion_artifact: DataIngestionArtifact):
        try:

            self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, "Data_Transformation")
            self.data_preprocessor_obj = os.path.join(self.data_transformation_dir, "object")
            self.transformed_data_path = os.path.join(self.data_transformation_dir, "data")
            self.save_model_dir = os.path.join("saved_models")

        except Exception as e:
            raise customexception(e,sys)
        
        def __str__(self):
            try:
                return str(self.__dict__)
            except Exception as e:
                raise customexception(e,sys)


class Model_Trainer_Config:
    def __init__(self, training_pipeline: Training_Pipeline_Config, data_transformation_artifact:DataTransformationArtifact):
        try :
            self.data_transformation_config = data_transformation_artifact
            self.training_pipeline = training_pipeline
            self.X_train = data_transformation_artifact.X_train_path
            self.X_test = data_transformation_artifact.X_test_path
            self.Y_train = data_transformation_artifact.Y_train_path
            self.Y_test = data_transformation_artifact.Y_test_path
            self.model_save_path = os.path.join(training_pipeline.artifact_dir, "Model_Trainer")
            self.save_model_obj = os.path.join("saved_models")

        except Exception as e:
            raise customexception(e,sys)
        