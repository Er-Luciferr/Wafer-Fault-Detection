import sys, os
from typing import List,Generator,Tuple
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,train_test_split

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils import MainUtils

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder= os.path.join(artifact_folder)
    trained_model_path= os.path.join(artifact_folder,'model.pkl')
    expected_accuracy = 0.45
    model_config_file_path = os.path.join('config' , 'model.yaml')

class ModelTrainer:
    def __inti__(self):

        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()

        self.models = {
            'XGBClassifier':XGBClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC':SVC(),
            'RandomForestClassifier':RandomForestClassifier()
        }


    def evaluate_models(self, X , y , models):
        try:
            X_train, X_test , y_train , y_test = train_test_split(
                X,y , test_size=.20 , random_state=42
            )
            report = {}
            for i in range(len(list(models))):
                model = list(models.values())[i]
                model.fit(X_train,y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                train_model_score = accuracy_score(y_train , y_train_pred)
                test_model_score = accuracy_score(y_test , y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            return report
        
        except Exception as e:
            raise CustomException(e,sys) from e 

    def get_best_model(self,  
                     X_train:np.array,
                     y_train:np.array,
                     X_test:np.array,
                     y_test:np.array):
        try:
            model_report :dict = self.evaluate_models(
                X_train = X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=self.models
            )

            print(model_report)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model_object = self.models[best_model_name]

            return best_model_name,best_model_object,best_model_score

        except Exception as e:
            raise CustomException(e,sys) from e

        
    def finetune_best_model(self,
                            best_model_object:object,
                            best_model_name,
                            X_train,
                            y_train,
                            ) -> object:

        try:
            model_param_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]


            grid_search = GridSearchCV(
                best_model_object, param_grid=model_param_grid, cv=5, n_jobs=-1, verbose=1 )
            
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_

            print("best params are:", best_params)

            finetuned_model = best_model_object.set_params(**best_params)
            

            return finetuned_model

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f"Splitting training and testing input and target feature")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            

            logging.info(f"Extracting model config file path")


            



            logging.info(f"Extracting model config file path")

            model_report: dict = self.evaluate_models(X=X_train, y=y_train, models=self.models)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]


            best_model = self.models[best_model_name]


            best_model = self.finetune_best_model(
                best_model_name= best_model_name,
                best_model_object= best_model,
                X_train= X_train,
                y_train= y_train
            )

            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            best_model_score = accuracy_score(y_test, y_pred)
            
            print(f"best model name {best_model_name} and score: {best_model_score}")


            if best_model_score < 0.5:
                raise Exception("No best model found with an accuracy greater than the threshold 0.6")
            
            logging.info(f"Best found model on both training and testing dataset")

 
        

            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_path}"
            )

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            
            return self.model_trainer_config.trained_model_path

            

            

        except Exception as e:
            raise CustomException(e, sys)

