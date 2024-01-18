#Importing System Libraries
import os
import sys
from dataclasses import dataclass

#Importing logger and exception handlers
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

#Scikit Learn libraries and functions
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

#Importing other modular codes
from src.components.data_cleaner import DataCleaningTransformer

#Importing Spacy 
import spacy

@dataclass
class DataTransformationConfig:
    processor_file_path: str = os.path.join('artifacts', 'processor.pkl')

class DataTransformationFunctions:
    def lemmatizer(self, text_string, nlp):
        doc = nlp(text_string)
        return " ".join([word.lemma_ for word in doc])

class DataTransformationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        return X['Message'].apply(lambda x: " ".join([word.lemma_ for word in self.nlp(x)]))

class DataTransformationObject:
    def __init__(self):
        self.data_cleaning_transformer = DataCleaningTransformer()
        self.nlp = spacy.load("en_core_web_sm")
        self.data_transformation_transformer = DataTransformationTransformer(self.nlp)

    def data_transformation_pipeline(self):
        transform_pipeline = Pipeline(
            
            steps=[

                ('cleaning', self.data_cleaning_transformer),
                ('lemmatizer', self.data_transformation_transformer)

            ]
        )

        logging.info("Data transformation pipeline ran successfully")

        return transform_pipeline

class DataTransformation:
    def __init__(self):
        self.data_transformer_config = DataTransformationConfig()
        self.data_transformer_object = DataTransformationObject()

    def initiate_data_transformation(self, train_df , test_df):

        try:
            
            pipeline = self.data_transformer_object.data_transformation_pipeline()
            
            text_column = ['Message']
            
            train_df['Message_transformed'] = pipeline.fit_transform(train_df[text_column])

            test_df['Message_transformed'] = pipeline.fit_transform(test_df[text_column])

            #Target column encoding ( Spam == 1 , Ham == 0)

            train_df['Spam'] = train_df['Category'].apply(lambda x:1 if x=='spam' else 0)

            test_df['Spam'] = test_df['Category'].apply(lambda x:1 if x=='spam' else 0)

            logging.info("Data Target column encoding done successfully")

            save_object(

                file_path = self.data_transformer_config.processor_file_path,
                obj=self.data_transformer_object.data_transformation_pipeline()

            )

            logging.info("Transformation Pipeline saved successfully")
 
            return (

                train_df,
                test_df

            )
        
        except Exception as e:
            raise CustomException(e,sys)
            




