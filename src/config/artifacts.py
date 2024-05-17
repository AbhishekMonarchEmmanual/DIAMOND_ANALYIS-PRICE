from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str 
    test_file_path:str
  
@dataclass 
class DataTransformationArtifact: 
    preprocessor_obj:str
    X_train_path:str 
    X_test_path:str
    Y_train_path:str
    Y_test_path:str

@dataclass
class ModelTrainerArtifact:
    object_path:str
    saved_model:str