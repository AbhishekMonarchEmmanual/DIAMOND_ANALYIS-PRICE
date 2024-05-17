import pandas as pd 
import numpy as np 
import os, sys, csv
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import Data_Transformation
from src.components.model_trainer import Model_Trainer
from src.config.config import Data_Transformation_Config
from src.config.config import Data_Ingestion_Config
from src.config.config import Model_Trainer_Config
from src.config.config import Training_Pipeline_Config
from src.config.artifacts import *
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor,)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from src.exception.exception import customexception
from src.logger.logging import logging
import mlflow


class Start_Model_Train:
    def __init__(self,):
        try:
            logging.info("WE ARE STARTING THE PIPELINE")
            

            training_pipeline = Training_Pipeline_Config()


            data_ingestion_config  = Data_Ingestion_Config(training_pipeline_config=training_pipeline)
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            print(data_ingestion_artifact)
            data_transformation_config = Data_Transformation_Config(training_pipeline_config=training_pipeline,
                                                                    data_ingestion_artifact=data_ingestion_artifact)


            data_transformation = Data_Transformation(data_transformation_config=data_transformation_config,
                                                    data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = data_transformation.inititate_data_transformation()
            print(data_transformation_artifact)
            model_trainer_config = Model_Trainer_Config(training_pipeline = training_pipeline, data_transformation_artifact=data_transformation_artifact)
            model_trainer = Model_Trainer(model_trainer_config = model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            print(model_trainer_artifact)

            logging.info("ENDING OF THE PIPELINE")
        except Exception as e:
            raise customexception(e,sys)

