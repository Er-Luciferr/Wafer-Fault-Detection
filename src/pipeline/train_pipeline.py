import sys , os

from src.components.data_ingestion import DataIngestion
from src.components.data_transfomation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainingPipeline:

    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            feature_store_file_path = data_ingestion.initiate_data_ingetion()
            return feature_store_file_path

        except Exception as e:
            raise CustomException(e,sys)

    def start_data_transformation(self):
        try:
            data_transfomation = DataTransformation(feature_store_file_path=feature_store_file_path)
            train_arr , test_arr, preprocessor_path = data_transfomation.initiate_data_transformation()
            return train_arr,test_arr,preprocessor_path

        except Exception as e:
            raise CustomException(e,sys)

    def start_model_training(self):
        try:
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(train_array , test_array)
            return model_score

        except Exception as e:
            raise CustomException(e,sys)

    def run_pipeline(self):
        try:
            feature_store_file_path = self.start_data_ingestion()
            train_arr,test_arr,preprocessor_path=self.start_data_transformation(feature_store_file_path)
            r2_square = self.start_model_training(train_arr,test_arr)
            print('Training completed. Trained model score:', r2_square)
        except Exception as e:
            raise CustomException(e,sys)