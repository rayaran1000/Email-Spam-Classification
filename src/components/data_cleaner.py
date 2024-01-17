#Importing System Libraries
import os
import sys
import string

#Importing logger and exception handlers
from src.logger import logging

#NLTK libraries and functions
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords

#Importing sklearn library and modules for building Custom Transformer for Data Cleaning
from sklearn.base import BaseEstimator, TransformerMixin

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



    

        

