import pandas as pd 
import numpy as np 
from src.logger.logging import logging 
from src.exception.exception import customexception
from src.config.config import *
from src.config.artifacts import *
from src.utils.utils import *
import os , sys 
from sklearn.model_selection import train_test_split
from pathlib import Path  
from dataclasses import dataclass
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 
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
import mlflow

class Model_Trainer: 
    def __init__(self, model_trainer_config: Model_Trainer_Config):
        try:
            self.model_trainer_config = model_trainer_config

        except Exception as e:
            raise customexception(e,sys)

    def initiate_model_trainer(self,):
        try: 
            
            logging.info(f"data is loaded")
            X_train = pd.read_csv(self.model_trainer_config.X_train)
            Y_train = pd.read_csv(self.model_trainer_config.Y_train)
            X_test = pd.read_csv(self.model_trainer_config.X_test)
            Y_test = pd.read_csv(self.model_trainer_config.Y_test)
            logging.info(f"we have splitted the data in to train and test set")
            
            logging.info(f"creating the method for using different models for machine learning")
            
            models = {
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor()
            }

            # Define parameter grids for each model
            params = {
                "Linear Regression": {
                    # Parameters for Linear Regression
                },
            
                "XGBRegressor": {
                'learning_rate': [0.1, 0.01],
                'n_estimators': [32, 64, 128]
            },
                "CatBoostRegressor": {
                'learning_rate': [0.1, 0.01],
                'iterations': [100, 200]
            }
            }
            
            logging.info(f"using each model one by one")
            
            best_models : dict=evaluate_models(X_train=X_train,Y_train= Y_train,X_test= X_test,Y_test= Y_test, models= models, param=params )
            
            logging.info(f"evaluation of the model is completed here is the best model report {best_models}")
            
            os.makedirs(self.model_trainer_config.model_save_path, exist_ok = True)
            os.makedirs(self.model_trainer_config.save_model_obj, exist_ok = True)
            

            sorted_best_models = sorted(best_models.items(), key=lambda x: x[1]['test_score'], reverse=True)
            save_dict_as_json(data_dict=str(sorted_best_models),file_path = self.model_trainer_config.model_save_path, file_name='report_complete.json')
            
            best_model_name, best_model_info = sorted_best_models[0]
            best_model = best_model_info['model']
            best_hyperparameters = best_model_info['hyperparameters']
            best_model_score = best_model_info['test_score']
            print(f"+++++++++++++++++++++++Best Model is This : {best_model_name} And R2Score : {best_model_info}++++++++++++++++++++++++++++++++++" )
            if best_model_score < 0.6:
                raise customexception("No Best Method we Found")
            logging.info(f"Best Found model on both training and testing data set {best_model}____{best_model_name}")
            os.makedirs(self.model_trainer_config.model_save_path, exist_ok = True)
            os.makedirs(self.model_trainer_config.save_model_obj, exist_ok = True)
            sav_obj = save_pickle_object(obj=best_model, file_name='model.pkl',file_path=self.model_trainer_config.model_save_path)
            save_folder = save_pickle_object(obj=best_model, file_name='model.pkl', file_path=self.model_trainer_config.save_model_obj)
            logging.info(f"saving for the model is completed")
            predicted = best_model.predict(X_test)
            r2_scores = r2_score(Y_test, predicted)
            
            model_trainer_artifact = ModelTrainerArtifact(object_path=self.model_trainer_config.model_save_path, saved_model=self.model_trainer_config.save_model_obj)
            with mlflow.start_run(run_name = 'Model_Trainer',nested=True):
                mlflow.sklearn.log_model(best_model, "best_model")
                mlflow.log_param("best_model_name", best_model_name)
                mlflow.log_param("best_hyperparameters", best_hyperparameters)
                mlflow.log_metric("best_model_r2_score", r2_scores)
                mlflow.log_artifact(sav_obj)
                mlflow.log_artifact(save_folder)
                mlflow.log_param("sav_obj", sav_obj)
                mlflow.log_param("Save Model Directory", save_folder)
            logging.info(f"+++++++++++++++++++Model Training Completed our Best Model : {best_model} R2_SCORE from BEST MODEL {r2_scores}++++++++++++++++")
            logging.info(f"Model Training Artifact is created here is the save object path {model_trainer_artifact.object_path}")
            logging.info(f"MODEL TRAINING IS COMPLETED")
            
            return r2_scores, model_trainer_artifact

        except Exception as e:
            raise customexception(e,sys) 