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
import mlflow

class Data_Transformation:
    def __init__(self , data_transformation_config: Data_Transformation_Config , data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info("___WE ARE CREATING THE CONFIF OR THE DATA TRANSFORMATION")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e : 
            raise customexception(e,sys)
        
    def inititate_data_transformation(self,):
        try:
            
            logging.info("___WE ARE INITIATING THE DATA TRANSFORMATION______")
            df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            X = df.drop(columns=['id', 'price'])
            y = df['price']
            cat_cols=X.select_dtypes(include="object").columns
            num_cols=X.select_dtypes(exclude="object").columns
       

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            num_pipeline=Pipeline(
            steps=[
                ("imputer",SimpleImputer()),
                ("scaler",StandardScaler())
            ]
        )
            cat_pipeline=Pipeline(

            steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("ordinalencoder",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories]))

            ]
        )
            preprocessor=ColumnTransformer(

            [
                ("num_pipeline",num_pipeline,num_cols),
                ("cat_pipeline",cat_pipeline,cat_cols)
            ]
        )
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30, random_state=42)
            logging.info(f"X_train : \n {X_train.head()}")
            preprocessor.fit_transform(X_train)

            preprocessor.transform(X_test)

            X_train=pd.DataFrame(preprocessor.fit_transform(X_train),columns=preprocessor.get_feature_names_out())
            X_test=pd.DataFrame(preprocessor.transform(X_test),columns=preprocessor.get_feature_names_out())

            os.makedirs(self.data_transformation_config.transformed_data_path, exist_ok=True)

            X_train_path = save_data_to_csv(X_train,'X_train.csv', self.data_transformation_config.transformed_data_path)
            X_test_path = save_data_to_csv(X_test, 'X_test.csv', self.data_transformation_config.transformed_data_path)
            Y_train_path = save_data_to_csv(y_train,'Y_train.csv', self.data_transformation_config.transformed_data_path)
            Y_test_path = save_data_to_csv(y_test, 'Y_test.csv', self.data_transformation_config.transformed_data_path)

            os.makedirs(self.data_transformation_config.data_preprocessor_obj, exist_ok=True)
            os.makedirs(self.data_transformation_config.save_model_dir, exist_ok=True)

            preprocessor_obj_path = save_pickle_object(preprocessor,file_path=self.data_transformation_config.data_preprocessor_obj,file_name='preprocessor_obj.pkl' )
            save_model_path = save_pickle_object(preprocessor,file_path=self.data_transformation_config.save_model_dir,file_name='preprocessor_obj.pkl' )
            with mlflow.start_run(run_name='Data_Transformer',nested=True):

                mlflow.log_param("train_data_shape", X_train.shape)
                mlflow.log_param("test_data_shape", X_test.shape)
            
            # Log artifacts (data files)
                mlflow.log_artifact(X_train_path)
                mlflow.log_artifact(X_test_path)
                mlflow.log_artifact(Y_train_path)
                mlflow.log_artifact(Y_test_path)
                mlflow.log_artifact(preprocessor_obj_path)
                mlflow.log_artifact(save_model_path)
          
            return DataTransformationArtifact(preprocessor_obj=preprocessor_obj_path,
                                              X_train_path=X_train_path,
                                              X_test_path=X_test_path,
                                              Y_train_path=Y_train_path,
                                              Y_test_path=Y_test_path)
          
        except Exception as e : 
            raise customexception(e,sys)
    logging.info("___________DATA TRANSFORMATION IS ENDED_______________________\n {DATATRANSFORMATIONARTIFACT}")                    