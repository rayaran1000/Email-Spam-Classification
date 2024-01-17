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
from src.utils import save_object

#Scikit Learn libraries and functions
from sklearn.pipeline import Pipeline,FunctionTransformer
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
        logging.info("Data transformation pipeline ran successfully")
        return X['Message'].apply(lambda x: " ".join([word.lemma_ for word in self.nlp(x)]))

class DataTransformationObject:
    def __init__(self):
        self.data_cleaning_transformer = DataCleaningTransformer()
        self.nlp = spacy.load("en_core_web_sm")
        self.data_transformation_transformer = DataTransformationTransformer(self.nlp)

    def data_transformation_pipeline(self):
        transform_pipeline = Pipeline([
                ('cleaning', self.data_cleaning_transformer),
                ('lemmatizer', self.data_transformation_transformer)
            ]
        )
        return transform_pipeline

class DataTransformation:
    def __init__(self):
        self.data_transformer_config = DataTransformationConfig()
        self.data_transformer_object = DataTransformationObject()

    def initiate_data_transformation(self, train_df):
        pipeline = self.data_transformer_object.data_transformation_pipeline()
        text_column = ['Message']
        train_df['Message_transformed'] = pipeline.fit_transform(train_df[text_column])
        return train_df
     




