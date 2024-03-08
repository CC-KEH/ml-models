import os,sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    
    def run_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            dataset_path = data_ingestion.initiate_data_ingestion()
            return dataset_path
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def run_data_transformation(self,dataset_path):
        try:
            data_tranformation = DataTransformation()
            train_arr,test_arr,preprocessor_path = data_tranformation.initiate_data_transformation()
            return train_arr,test_arr,preprocessor_path
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def run_model_trainer_and_evaluation(self,train_arr,test_arr):
        try:
            model_trainer = ModelTrainer()
            evaluation_report = model_trainer.initiate_model_trainer()
            return evaluation_report
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def run_pipeline(self):
        try:
            # Data Ingestion
            dataset_path = self.run_data_ingestion()
            
            # Data Transformation
            train_arr,test_arr,preprocessor_path = self.run_data_transformation(dataset_path)
            
            # Model Training and Evaluation
            evaluation_report = self.run_model_trainer_and_evaluation(test_arr,test_arr)
            
        except Exception as e:
            raise CustomException(e,sys)    