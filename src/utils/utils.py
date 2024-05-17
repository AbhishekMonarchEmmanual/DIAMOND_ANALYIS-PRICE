import os , sys , csv
import pickle 
import numpy as np 
import pandas as pd 
from src.logger.logging import logging 
from src.exception.exception import customexception
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
import json
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


def save_data_to_csv(data:pd.DataFrame, file_name:str, folder_path:str)->str:
    """
    Following Function takes Input as Dataframe and 
    Save it to desired Location and return file path string 

    data : data frame 
    File_name : Name of the file you want 
    Folder_path : The path or location you want to save 
    
    """
    df = pd.DataFrame(data)
    
    # Specify the file location
    os.makedirs(folder_path, exist_ok= True)

    file_path = os.path.join(folder_path, file_name)

    # Write DataFrame to CSV file
    df.to_csv(file_path, index=False)


    print("Data has been saved to", file_path)
    return file_path


def save_pickle_object(obj, file_path, file_name):
   """IT SAVES PICKLE FILE
   obj = Give the object to save 
   File_path = Folder you want to save 
   File_name = Save the file with the name you want to
   
   """
   full_file_path = os.path.join(file_path, file_name)

   # Serialize and save the object to a file
   with open(full_file_path, 'wb') as file:
       pickle.dump(obj, file)

   print("Object has been saved as pickle file:", full_file_path)
   return full_file_path

def evaluate_models(X_train, Y_train,X_test,Y_test,models,param):
    try:
        report = {}
        best_models = {}

        for model_name, model in models.items():
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3)

            gs.fit(X_train, Y_train)

            best_model = gs.best_estimator_
            best_hyperparameters = gs.best_params_

            best_model.fit(X_train, Y_train)

            Y_train_pred = best_model.predict(X_train)
            Y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(Y_train, Y_train_pred)
            test_model_score = r2_score(Y_test, Y_test_pred)

            report[model_name] = test_model_score
            best_models[model_name] = {
                "model": best_model,
                "hyperparameters": best_hyperparameters,
                "test_score": test_model_score
            }

        return best_models


    except Exception as e:
        raise customexception(e, sys)
    
def save_dict_as_json(data_dict, file_path, file_name):
    """
    Save a dictionary as a JSON file.

    Args:
    - data_dict (dict): The dictionary to be saved as JSON.
    - file_path (str): The directory path where the file will be saved.
    - file_name (str): The name of the JSON file.

    Returns:
    - bool: True if the file was saved successfully, False otherwise.
    """
    try:
        # Concatenate file path and file name
        full_file_path = file_path + "/" + file_name

        # Writing the dictionary to a JSON file
        with open(full_file_path, "w") as json_file:
            json.dump(data_dict, json_file)
        
        print("File saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        raise customexception(e,sys) 