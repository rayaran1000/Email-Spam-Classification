# Email Spam Classification Project Overview

## Primary Objective
1. **Data Collection:**
   Efficiently gather diverse email samples, including both spam and non-spam (ham) messages, to create a comprehensive dataset.

2. **Data Preprocessing:**
   Clean and preprocess the collected data to enhance the quality of features used in classification algorithms.

3. **Feature Extraction:**
   Identify relevant features from the email content, such as text, sender information, and header details.

4. **Model Development:**
   Implement machine learning or deep learning models for email spam classification, ensuring high accuracy and generalization.

## Secondary Objectives
1. **Feature Importance Analysis:**
   Conduct analysis to determine the most significant features influencing the classification model.

2. **Explainability:**
   Ensure interpretability of the model's decisions, providing insights into why certain emails are classified as spam.

3. **Evaluation Metrics:**
   Utilize appropriate metrics (e.g., precision, recall, F1 score) to assess the performance of the classification model.

4. **Real-time Classification:** 
   Explore methods for real-time spam classification to enhance email security.

5. **Adaptability:**
   Design the classification system to adapt to evolving spam patterns and techniques.

6. **User Interface:**
   Develop a user-friendly interface for stakeholders to interact with the classification system and manage settings.

## Directory Structure 

```plaintext
/project
│   README.md
│   requirements.txt
|   application.py
|   setup.py
|   Webpage
└───artifacts
|   └───model.pkl
|   └───processor.pkl
|   └───vectorizer.pkl
|   └───raw.csv
|   └───train.csv
|   └───test.csv
└───logs
└───notebooks
|   └───data
|        └───Email Spam.csv
|       Data Analysis , Data Processing , Model Training and Evaluation.ipynb   
└───src
|   exception.py
|   logger.py
|   utils.py
|   └───components
|       └───data_ingestion.py
|       └───data_cleaner.py
|       └───data_transformation.py
|       └───model_trainer.py
|   └───pipelines
|       └───training_pipeline.py
|       └───prediction_pipeline.py
└───templates
|   └───home.html
|   └───index.html

```
## Installation

For Installing the necessery libraries required 

```bash
  pip install -r requirements.txt
```
    
## Deployment

To deploy this project run

1. To start the training pipeline 

```bash
  python src/pipelines/training_pipeline.py
```

2. Once the model is trained, to run the Flask application

```bash
  python application.py
```

3. Go to 127.0.0.1/predictdata to get the webpage

4. Type the Email you want to predict

## Exploratory Data Analysis Path followed:

> 1. Get Word count of each message

> 2. Average Word Length of each message

> 3. Stopwords count and rate of each message


## Review Data Cleaning

> 1. Removing null records

> 2. Removing Punctuations

> 3. Removing stopwords

> 4. Converting abbreviations to words for top 50 recurring words

> 5. Target column encoding ( 1 for Spam and 0 for Ham)

## Model Prediction

> 1. Performed Lemmatization using Spacy

> 2. Converted the Message column into sparse matrix using Count Vectorizer ( Bag of Words approach)

> 3. Train and Test Split using Stratified fold

> 4. Models used for Training : Random Forest , Decision Tree, Logistic Regression , K neighbours classifier , Adaboost Classifier , Support Vector Classifier , Multinomial Naive Bayes

> 5. Cross Validation used : Stratified K fold

> 6. Evaluation metrices used : ROC_AUC_Score and Cross validation score from Stratified K fold


## Acknowledgements

I would like to express my gratitude to the following individuals and resources that contributed to the successful completion of this Salees Forecasting project:

- **[Kaggle]**: Special thanks to Kaggle for providing access to the message spam/ham dataset to gain valuable insights into the industry challenges.

- **Open Source Libraries**: The project heavily relied on the contributions of the open-source community. A special mention to libraries such as nltk, pandas, and TextBlob, which facilitated data analysis, model development, and visualization.

- **Online Communities**: I am grateful for the support and knowledge shared by the data science and machine learning communities on platforms like Stack Overflow, GitHub, and Reddit.

This project was a collaborative effort, and I am grateful for the support received from all these sources.


