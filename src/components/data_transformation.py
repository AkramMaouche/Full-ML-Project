from src.logger import logging 
from src.exception import CustomExeption 

from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder ,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import os 
import sys 

import pandas as pd 
import numpy as np 
from dataclasses import dataclass


@dataclass
class DatatransforamtinConfig:
    preprossor_obj_file_path  = os.path.join('artifact','preprocessor.pkl')

class DataTransformation(): 

    def __init__(self):
        self.Data_transormation_config = DatatransforamtinConfig()

    def get_datatranformer_object(self):

        try :
            cat_columns = ['race_ethnicity',
                           'parental_level_of_education',
                           'lunch',
                           'test_preparation_course',
                           'gender']
            
            num_columns = ['reading_score','writing_score']

            num_pipline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encodeing',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            

        
        except Exception as e : 
            raise CustomExeption(sys,e)
        
