import os,sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import MainUtils
from sklearn.metrics import log_loss

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.utils  =  MainUtils()
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Gaussian Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(),
        }
        
    def train_and_evaluate_models(self,X_train,y_train,X_test,y_test,models):
        report = {}
        logging.info('Training Models')
        for i in models:
            logging.info(f'Training {i}')
            models[i].fit(X_train,y_train)
            logging.info(f'Evaluating {i}')
            preds = models[i].predict(X_test)
            loss = log_loss(y_test,preds)
            report[i] = loss
        return report
            
    
    def initiate_model_trainer(self,train_arr,test_arr):
        
        X_train = train_arr[:,:-1]
        y_train = train_arr[:,-1]
        X_test  = test_arr[:,:-1]
        y_test  = test_arr[:,-1]
        
        # Train and Evaluate Models 
        report = self.train_and_evaluate_models(X_train,y_train,X_test,y_test,models=self.models)
        # Return the Model with best Log Loss Score
        best_model_score = max(sorted(report.values())) 
        best_model_name = list(report.keys())[
            list(report.values()).index(best_model_score)
        ]
        best_model = self.models[best_model_name]
        return (best_model_name,best_model_score,best_model)
    