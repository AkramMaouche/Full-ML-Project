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

from utils import saved_object


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

            logging.info(f'Categorical Features :{cat_columns}')
            logging.info(f'numerical features : {num_columns}')

            preprocessor = ColumnTransformer(

                [
                    ('numerical pipline',num_pipline,num_columns),
                    ('categorical pipline',cat_pipline,cat_columns)

                ]
            )
            return preprocessor
        
        except Exception as e : 
            raise CustomExeption(e,sys)

    def initiate_data_transformation(self): 
        try:
            train_df = pd.read_csv('train.csv')
            test_df = pd.read_csv('test.csv')

            logging.info('Read train and test data')
            logging.info('obtaining preprocessor object')

            preprocessor_obj = self.get_datatranformer_object()

            target_column = 'reading_score' 
            numerical_columns = ['math_score','writing_score']
            
            input_feature_train_df = train_df.drop(columns =[target_column],axis = 1)
            target_feature_train_df = train_df[target_column] 

            input_feature_test_df = test_df.drop(columns = [target_column],axis = 1)
            target_feature_test_df = test_df[target_column] 

            logging.info(
                f'Applying preprocessing object on training dataframe and testing dataframe'
                )
            input_feature_train_df_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_df_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_(input_feature_train_df_arr,np.array(target_feature_train_df))
            test_arr = np.c_[input_feature_test_df_arr,np.array(target_feature_test_df)]

            logging.info(f'saved preprocessing object')


            saved_object(self.Data_transormation_config.preprossor_obj_file_path,obj=preprocessor_obj)


            return(
                train_arr,
                test_arr,
                self.Data_transormation_config.preprossor_obj_file_path
            )



            



        except Exception as e :
            raise CustomExeption(e,sys)
        
