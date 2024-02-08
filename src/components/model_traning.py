from logger import logging 
from exception import CustomExeption 

from sklearn.metrics import r2_score 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor 
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from dataclasses import dataclass 
import os 
import sys

@dataclass 
class ModeltraningConfig : 
    trained_model_file_path = os.path.join("artifact","model.pkl")

class ModelTaining(): 
    def __init__(self) -> None:
        self.model_trainer_config = ModeltraningConfig()
    

    def initiateModelTrainer(self,train_array,test_array):

        try: 
            
            logging.info("split train and test input")
            X_train,y_train,X_test,y_test = (

                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            model ={

                "Random Forset": RandomForestRegressor(), 
                "Decision Tree": DecisionTreeRegressor(),
                "GradianBoost" : GradientBoostingRegressor(), 
                "Linear Rigression":LinearRegression(),
                "XGB Regressor": XGBRegressor(), 
                "Catboosting Regressor": CatBoostRegressor(), 
                "Adaboost Regressor" : AdaBoostRegressor(),
                "KNeighbors Regressor": KNeighborsRegressor()

            }

            model_report:dict = evaluate_models()



        except Exception as e:
            raise CustomExeption(e,sys)

