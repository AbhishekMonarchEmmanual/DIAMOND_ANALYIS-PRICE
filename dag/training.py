import pandas as pd 
import numpy as np 
import os, sys, csv
src_path = '/mnt/e/ineuron_practice/MLOPS/model_project_mlops/src'
sys.path.insert(0, src_path)
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
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Function definitions for each task
def initiate_data_ingestion():
    training_pipeline = Training_Pipeline_Config()
    data_ingestion_config = Data_Ingestion_Config(training_pipeline_config=training_pipeline)
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    return data_ingestion_artifact

def initiate_data_transformation(ti):
    data_ingestion_artifact = ti.xcom_pull(task_ids='data_ingestion')
    training_pipeline = Training_Pipeline_Config()
    data_transformation_config = Data_Transformation_Config(
        training_pipeline_config=training_pipeline,
        data_ingestion_artifact=data_ingestion_artifact
    )
    data_transformation = Data_Transformation(
        data_transformation_config=data_transformation_config,
        data_ingestion_artifact=data_ingestion_artifact
    )
    data_transformation_artifact = data_transformation.initiate_data_transformation()
    return data_transformation_artifact

def initiate_model_trainer(ti):
    data_transformation_artifact = ti.xcom_pull(task_ids='data_transformation')
    training_pipeline = Training_Pipeline_Config()
    model_trainer_config = Model_Trainer_Config(
        training_pipeline=training_pipeline,
        data_transformation_artifact=data_transformation_artifact
    )
    model_trainer = Model_Trainer(model_trainer_config=model_trainer_config)
    model_trainer_artifact = model_trainer.initiate_model_trainer()
    return model_trainer_artifact

# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

# Define the DAG
with DAG(
    'training_pipeline_dag',
    default_args=default_args,
    description='A simple training pipeline DAG',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    # Define the tasks
    data_ingestion = PythonOperator(
        task_id='data_ingestion',
        python_callable=initiate_data_ingestion
    )

    data_transformation = PythonOperator(
        task_id='data_transformation',
        python_callable=initiate_data_transformation
    )

    model_trainer = PythonOperator(
        task_id='model_trainer',
        python_callable=initiate_model_trainer
    )

    # Set the task dependencies
    data_ingestion >> data_transformation >> model_trainer
