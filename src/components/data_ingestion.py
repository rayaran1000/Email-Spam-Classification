#Importing System Libraries
import os
import sys
from dataclasses import dataclass

#Importing Dataframe handling libraries
import numpy as np
import pandas as pd

#Importing logger and exception handlers
from src.logger import logging
from src.exception import CustomException

#Scikit Learn libraries and functions
from sklearn.model_selection import train_test_split

#Testing
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig: #Defining data paths for raw , train and test data files

    raw_data_path : str = os.path.join('artifacts','raw.csv')

    train_data_path : str = os.path.join('artifacts','train.csv')

    test_data_path : str = os.path.join('artifacts','test.csv')

class DataIngestion:

    def __init__(self):

        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiated data ingestion")

        try:
            
            raw_df = pd.read_csv('notebooks\data\Email Spam.csv')

            X = raw_df['Message']
            y = raw_df['Category']

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True) # Creating the artifacts directory

            raw_df.to_csv(self.data_ingestion_config.raw_data_path,index=False) # Raw data saved in csv

            logging.info("Initiated train test split")
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y) # Data Split based on target column for equal distribution among train and test data

            #Creating the Train and Test dataframes for saving in csv format
            train_df = pd.concat([X_train,y_train],axis=1)
            test_df = pd.concat([X_test,y_test],axis=1)

            train_df.to_csv(self.data_ingestion_config.train_data_path)
            test_df.to_csv(self.data_ingestion_config.test_data_path)

            logging.info("Finished the Data Ingestion Process")

            return (

                train_df,
                test_df

            )
       
        except Exception as e:
            raise CustomException(e,sys)
        


if __name__ == '__main__':

    data_ingestion = DataIngestion()

    train_df,test_df = data_ingestion.initiate_data_ingestion()

    data_transfomation = DataTransformation()

    cleaned_lemmatized_train_df , cleaned_lemmatized_test_df = data_transfomation.initiate_data_transformation(train_df,test_df)

    model_trainer = ModelTrainer()

    best_model_name , best_model_score = model_trainer.initiate_model_training(cleaned_lemmatized_train_df,cleaned_lemmatized_test_df)

    print('Best Model Name: ',best_model_name)
    print('Best Model Score: ',best_model_score)
    

       






