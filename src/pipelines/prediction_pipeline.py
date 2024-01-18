import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,message):
        try:

            model_path=os.path.join("artifacts","model.pkl") # Model path
            vectorizer_path=os.path.join("artifacts","vectorizer.pkl") # Vectorizer path
            preprocessor_path=os.path.join('artifacts','processor.pkl') # Transformer path
            print("Before Loading")
            model=load_object(file_path=model_path) #Loads the Model from the Pickle file
            vectorizer=load_object(file_path=vectorizer_path) #Loads the Vectorizer from the Pickle file
            preprocessor=load_object(file_path=preprocessor_path) #Loads the Preprocessor from Pickle file
            print("After Loading")
            data_scaled=preprocessor.transform(message)
            data_vectorized=vectorizer.transform(data_scaled)
            preds=model.predict(data_vectorized)

            #Converting the prediction back to the original category format
            if preds == 0:
                preds_final = 'Ham'
            elif preds == 1:
                preds_final = 'Spam'

            return preds_final
                         
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData: # Class responsible for mapping all the inputs that we are getting in the HTML webpage with the backend
    def __init__(self,        
        Message: str,
        ):

#Assigning these values(coming from web application)
        self.Message = Message

    def get_data_as_data_frame(self): #Returns all our input data as dataframe, because we train our models using dataframes
        try:
            custom_data_input_dict = {
                "Message": [self.Message]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)