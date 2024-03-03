import sys , os
import numpy as np 
import pandas as pd 
from pymongo import MongoClient
from zipfile import Path
from src.constant import *
from src.exception import CustomException
from src.logger import logging 
from src.utils.main_utils import MainUtils 
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(artifact_folder)

class DataIngestion:
    def __init__(self):

        self.data_ingestion_config = DataIngestionConfig()
        self.MainUtils = MainUtils()

    def export_collection_as_dataframe(self , collection_name , db_name):
        try:
            mongo_client = MongoClient(MONGO_DB_URL)

            collection = mongo_client[db_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if '_id' in df.columns.to_list():
                df = df.drop(columns=['_id'] , axis=1)

            df.replace({'na': np.nan} , inplace = True)
            return df 
            
        except Exception as e:
            raise CustomException(e,sys)