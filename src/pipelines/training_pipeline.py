import os
import sys

#Libraries for logging and Custom Exception
from src.exception import CustomException 
from src.logger import logging 

#Libraries for data ingestion
from src.components.data_ingestion import DataIngestion

#Libraries for data transformation
from src.components.data_transformation import DataTransformation

#Libraries for model trainer
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:

    def __init__(self):

        self.data_ingestion = DataIngestion()
        self.data_transfomation = DataTransformation()
        self.model_trainer = ModelTrainer()

    
    def initiate_training_pipeline(self):

        try:

            logging.info("Training pipeline initiated")

            train_df,test_df = self.data_ingestion.initiate_data_ingestion()
            logging.info("Data Ingestion completed successfully")

            cleaned_lemmatized_train_df , cleaned_lemmatized_test_df = self.data_transfomation.initiate_data_transformation(train_df,test_df)
            logging.info("Data Transformation completed successfully")

            best_model_name , best_model_score = self.model_trainer.initiate_model_training(cleaned_lemmatized_train_df,cleaned_lemmatized_test_df)
            logging.info("Model Training completed successfully")

            return (
                
                best_model_name,
                best_model_score

            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":

    training_pipeline = TrainingPipeline()

    training_pipeline.initiate_training_pipeline()