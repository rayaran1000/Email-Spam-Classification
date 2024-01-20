#Importing System Libraries
import os
import sys
from dataclasses import dataclass

#Importing logger and exception handlers
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

#Importing CountVectorizor
from sklearn.feature_extraction.text import CountVectorizer

#Hyperparameterization
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

#Importing the Models and Evaluation metrices
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import roc_auc_score

@dataclass
class DataVectorizerConfig:

    vectorizer_file_path: str = os.path.join('artifacts', 'vectorizer.pkl') # File will contain the vectorizer configuration

class ModelTrainerConfig:

    model_file_path: str = os.path.join('artifacts', 'model.pkl') # File will contain the model 

class DataVectorizer:

    def __init__(self):

        self.data_vectorizer_config = DataVectorizerConfig()

    def initiate_data_vectorization(self,train_df, test_df):

        try:

            #Vectorizing the text column for model training
            cv = CountVectorizer()

            X_train = cv.fit_transform(train_df['Message_transformed']).toarray()

            cv.fit(train_df['Message_transformed'])

            X_test = cv.transform(test_df['Message_transformed']).toarray()

            y_train = train_df['Spam']
            y_test = test_df['Spam']

            save_object(

                file_path=self.data_vectorizer_config.vectorizer_file_path,
                obj = cv
            )

            return (

                X_train,
                X_test,
                y_train,
                y_test

            )

        except Exception as e:
            raise CustomException(e,sys)
        
class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()
        self.data_vectorizer = DataVectorizer()

    def initiate_model_training(self,train_df,test_df):

        try:

            X_train,X_test,y_train,y_test = self.data_vectorizer.initiate_data_vectorization(train_df,test_df)

            models = { # Dictionary of Models we will be trying
                    "Random Forest" : RandomForestClassifier(),
                    "Decision Tree" : DecisionTreeClassifier(),
                    "Logistic Regression" : LogisticRegression(),
                    "K Neighbours Classifier" : KNeighborsClassifier(),
                    "Adaboost Classifier" : AdaBoostClassifier(),
                    "Support Vector Classifier" : SVC(),
                    "Multinomial Bayes" : MultinomialNB(),
                }
        
            params={ # Creating a dictionary with the parameters for each Model(Hyperparameter values)
                    "Decision Tree": {
                        'criterion':['gini', 'log_loss', 'entropy'],
                    },
                    "Random Forest":{
                        'criterion':['gini', 'log_loss', 'entropy'],
                    },
                    "Logistic Regression":{},
                    "K Neighbours Classifier" :{
                            'n_neighbors': [3, 5, 7, 9],
                    },
                    "Adaboost Classifier":{
                        'learning_rate':[.1,.01,0.5,.001],
                    },
                    "Support Vector Classifier":{
                        'kernel':['linear', 'poly'],
                        'C':[0.001,0.01,0.1]
                    },
                    "Multinomial Bayes":{
                        'alpha':[0.001,0.01,0.1,1,10]
                    },
                    
                }
            
            model_report:dict=self.evaluate_model(X_train, y_train, X_test , y_test, models, params) # This function created inside the utils.py

            #Sorting and extracting the best model using the model score
            best_model_score = max(sorted(model_report.values()))

            #To get the best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] # Nested lists concept used here

            best_model = models[best_model_name]

            best_model.fit(X_train, y_train) # Helps in saving a fitted model as object

            logging.info("Best found model on both training and testing datasets")

            save_object(# Creating the Model.pkl file corresponding to the best model that we will get
                
                    file_path=self.model_trainer_config.model_file_path,
                    obj=best_model

            )

            return (

                best_model_name,
                best_model_score,

            )
          
        except Exception as e:
            raise CustomException(e,sys)
          
          
    def evaluate_model(self,X_train, y_train, X_test , y_test, models, params):

        try:

            report = {}

            stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            for model_name, model in models.items():
                param = params[model_name]

                gs = GridSearchCV(model, param, cv=stratified_kfold)
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_

                # Perform Stratified K-Fold cross-validation
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')
                mean_cv_score = cv_scores.mean()
                

                # Train the best model on the training dataset
                best_model.fit(X_train, y_train)

                # Make predictions on the entire dataset
                y_pred = best_model.predict(X_test)

                # Calculate the ROC AUC score on the entire dataset
                roc_score = roc_auc_score(y_test, y_pred)

                #Average of the ROC AUC score and cross validation score for selecting the best model
                model_score = (mean_cv_score + roc_score) / 2

                report[model_name] = {model_score}

            return report
   
        except Exception as e:
            raise CustomException(e,sys)
        
        
        









    


