from src.logger import logging 
from src.exception import CustomException

from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder ,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import os 
import sys 

import pandas as pd 
import numpy as np 
from dataclasses import dataclass
from src.utils import saved_object

@dataclass
class DataTransformationconfig: 

    preproccesor_obj_file = os.path.join('artifact','preprocessor.pkl')

class DataTransformation: 

    def __init__(self): 
        self.data_transformation_config = DataTransformationconfig()
    
    def get_data_tranformer_object(self):
        '''
        This function is responsible for data transformation 

        '''

        try: 
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = [
                'gender',
                'parental_level_of_education',
                'race_ethnicity',
                'lunch',
                'test_preparation_course'
            ]

            num_pipline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipline = Pipeline(
                steps=[

                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean= False))
                ]  
            )   

            logging.info(f"Categorical coulmns are:{categorical_columns}")
            logging.info(f'Numerical Columns:{numerical_columns}')

            processor = ColumnTransformer(

                [
                   ('num_pipline',num_pipline,numerical_columns),
                   ('cat_pipline',cat_pipline,categorical_columns) 
                ]
            )

            return processor

        except Exception as e : 
            raise CustomException(e, sys)

    def initiat_data_transformation(self,train_path,test_path):
        try : 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read Train and test data')
            logging.info('Obtraining preprocessing object')
            
            preprocessing_obj = self.get_data_tranformer_object()

            target_column_name ="math_score"
            numerical_columns = ["reading_score","writing_score"]

            input_features_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f'applying preprocessing object on training and testing dataframe')

            input_features_train_processing = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_processing = preprocessing_obj.transform(input_features_test_df)


            train_arr = np.c_[input_features_train_processing,np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_processing,np.array(target_feature_test_df)]

            logging.info("saved preprocessing data")

            saved_object(
                file_path=self.data_transformation_config.preproccesor_obj_file,
                obj= preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preproccesor_obj_file,
            )
        
        except Exception as e:
            raise CustomExeption(e,sys) 
            

