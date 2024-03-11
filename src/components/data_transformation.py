import sys, os 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,FunctionTransformer , StandardScaler
from sklearn.pipeline import Pipeline

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils import MainUtils
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    artifact_dir=os.path.join(artifact_folder)
    transformed_test_file_path=os.path.join(artifact_dir , 'test.nyp')
    transformed_train_file_path=os.path.join(artifact_dir, 'train.npy')
    transformed_object_file_path=os.path.join(artifact_dir,'preprocessor.pkl')


class DataTransformation:
    def __init__(self, feature_store_file_path):

        self.feature_store_file_path = feature_store_file_path
        self.DataTransformationConfig = DataTransformationConfig()

        self.utils=  MainUtils()


    @staticmethod
    def get_data(feature_store_file_path:str) -> pd.DataFrame:
        """
        Method Name :   get_data
        Description :   This method reads all the validated raw data from the feature_store_file_path and returns a pandas DataFrame containing the merged data. 
        
        Output      :   a pandas DataFrame containing the merged data 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.0
        """
        try:
            data = pd.read_csv(feature_store_file_path)
            data.rename(columns={'Good/Bad': TARGET_COLUMN}, inplace=True)
            return data

        except Exception as e:
            raise CustomException(e,sys) 

    def get_data_transformer_object(self):
        try:
            preprocessor = Pipeline(
                steps=[
                    ('imputer' , SimpleImputer(strategy='constant',fill_value=0)),
                    ('scaler', RobustScaler())
                    ]
                )

            return preprocessor
            

        except Exception as e:
            raise CustomException(e,sys) 

    def initiate_data_transformation(self):
        """
            Method Name :   initiate_data_transformation
            Description :   This method initiates the data transformation component for the pipeline 
            
            Output      :   data transformation artifact is created and returned 
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """        
        try:
            dataframe = self.get_data(feature_store_file_path=self.feature_store_file_path)

            X= dataframe.drop(columns=TARGET_COLUMN)
            y= np.where(dataframe[TARGET_COLUMN]== -1,0, 1)

            X_train,X_test,y_train,y_test = train_test_split(X,y , test_size=0.20 , random_state=42 )

            preprocessor = self.get_data_transformer_object()
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            preprocessor_path = self.DataTransformationConfig.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.utils.save_object(file_path=preprocessor_path , obj=preprocessor)

            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled , np.array(y_test)]

            return (train_arr,test_arr,preprocessor_path)          


        except Exception as e:
            raise CustomException(e,sys) 
