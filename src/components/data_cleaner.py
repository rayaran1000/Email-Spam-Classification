#Importing System Libraries
import os
import sys
import string
from dataclasses import dataclass

#Importing Dataframe handling libraries
import numpy as np
import pandas as pd

#Importing logger and exception handlers
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

#NLTK libraries and functions
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords

#Importing sklearn library and modules for building Custom Pipeline for Data Cleaning
from sklearn.pipeline import Pipeline,FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

@dataclass
class DataCleanerConfig:

    cleaner_file_path : str=os.path.join('artifacts','cleaner.pkl')

class DataCleaner:

    def __init__(self):

        self.data_cleaner_config = DataCleanerConfig()

        self.data_cleaner_transformer = DataCleaningTransformer()


    def initiate_data_cleaning(self,train_df,test_df):

        try:

            text_column = ['Message']

            train_df['Message_cleaned'] = self.data_cleaner_transformer.fit_transform(train_df[text_column])

            test_df['Message_cleaned'] = self.data_cleaner_transformer.fit_transform(test_df[text_column])

            return (
                
                train_df,
                test_df

            )
       
        except Exception as e:
            raise CustomException(e,sys)

class DataCleanerFunctions:

    def lower_case_converter(self,text_string): # Lowercase convertor

        return text_string.apply(lambda x: " ".join([word.lower() for word in x.split()]))
    
    def word_tokenizer(self,text_string): # Tokenizor

        return text_string.apply(lambda x: word_tokenize(x))
    
    def punctuation_remover(self,text_string): # Punctuations remover

        text_string = text_string.apply(lambda x : [word for word in x if word not in string.punctuation]) # Removing punctuations using string function

        dot_punctuations = ['..','...'] 

        text_string = text_string.apply(lambda x : " ".join([word for word in x if word not in dot_punctuations])) # Removing dots other than full-stop

        return text_string.str.replace('[^\w\s]', '') # Removing white spaces using regex
    
    def stopword_remover(self,text_string): # Stopwords Remover

        stop_words = stopwords.words('english')

        return text_string.apply(lambda x : " ".join([word for word in x.split() if word not in stop_words]))
    
    def abbreviation_mapper(self,text_string):

        abbreviation_mapping = { 'u': 'you','2':'to','ur': 'your','n': 'and','gt': 'great','lt': 'little','nt':'not','4':'for','ü':'you',
                                'txt':'text','r':'are','da':'the','pls':'please'} # Based on top 50 recurring words 

        #Converting Abbreviations to actual phrases using the above dictionary(only applied to top recurring abreviations)
        return text_string.apply(lambda x : " ".join([abbreviation_mapping.get(word,word) for word in x.split()]))
    
class DataCleaningTransformer(BaseEstimator, TransformerMixin): # Custom transformer to clean data in a pipeline

    def __init__(self):
        
        self.data_cleaner_function = DataCleanerFunctions()
        
    def fit(self, X, y=None):
        return self

    def transform(self, X): # Cleaning pipeline

        logging.info("Data cleaning pipeline ran successfully")

        return X.apply(lambda col: self.data_cleaner_function.lower_case_converter(col)) \
            .apply(lambda col: self.data_cleaner_function.word_tokenizer(col)) \
            .apply(lambda col: self.data_cleaner_function.punctuation_remover(col)) \
            .apply(lambda col: self.data_cleaner_function.stopword_remover(col)) \
            .apply(lambda col: self.data_cleaner_function.abbreviation_mapper(col))      



    

        

