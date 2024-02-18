from src.logger import logging 
from src.exception import CustomException
from src.utils import evaluate_models
from src.utils import saved_object

from sklearn.metrics import r2_score 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
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

class ModelTrainer : 
    def __init__(self):
        self.model_trainer = ModeltraningConfig()

    def initiate_model_tariner(self,train_array,test_array): 
        try: 
            logging.info("Split train and test data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models ={
                'Random Forest': RandomForestRegressor(), 
                'Ridge':Ridge(),
                'Desicion Tree': DecisionTreeRegressor(),
                'Linear Rigrission': LinearRegression(), 
                'AdBoost Regressor': AdaBoostRegressor(),
                'GradianBoost Regressor': GradientBoostingRegressor(),
                'XGboost Regressor' : XGBRegressor(),
                'Catboost Regressor' : CatBoostRegressor(verbose=False), 
                'KNeighbors Regressor' : KNeighborsRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,
                                                X_test=X_test,y_test=y_test,
                                                  models=models,param=params ) 

            ## to get best model 
            best_model_score = max(sorted(model_report.values()))
            print('modelScore :',best_model_score)

            ## get best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            print('modelName:',best_model_name)

            best_model = models[best_model_name]
             

            if best_model_score<0.6: 
                raise CustomException("No best model found")
            
            logging.info('best found model on both training and testing dataset')

            saved_object(
                file_path= self.model_trainer.trained_model_file_path,obj=best_model)
            
            predicted = best_model.predict(X_test)

            result_r2score = r2_score(y_test,predicted)

            return result_r2score
        
        except Exception as e:
            raise CustomException(e,sys)
        