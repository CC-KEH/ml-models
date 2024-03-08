# https://raw.githubusercontent.com/sunnysavita10/credit_card_pw_hindi/main/creditCardFraud_28011964_120214.csv
import pandas as pd
import sys,os,json
from dataclasses import dataclass
from src.utils import MainUtils
from src.constant import MONGO_COLLECTION_NAME,MONGO_DATABASE_NAME,MONGO_URI
from src.exception import CustomException
from pymongo import MongoClient
import numpy as np
from src.logger import logging
@dataclass
class DataIngestionConfig:
    artifacts_folder = os.path.join('artifacts')
    
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()

    def import_collection_as_dataframe(self):
        try:
            mgdb = MongoClient(MONGO_URI)
            collection = mgdb[MONGO_DATABASE_NAME][MONGO_COLLECTION_NAME]
            df = pd.DataFrame(list(collection.find()))
            
            logging.info('Reading Data')
            
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"],axis=1)
            
            df.replace("nan",np.nan)
            
            logging.info('Converting Data to CSV File')
            
            raw_file_path = self.data_ingestion_config.artifacts_folder
            os.makedirs(raw_file_path,exist_ok=True)
            dataset_path = os.path.join(raw_file_path,'ccf.csv')
            df.to_csv(dataset_path,index=False)
            return dataset_path
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            logging.info('Importing Data from MongoDB')
            dataset_path = self.import_collection_as_dataframe()
            logging.info('Successfully Imported Data from MongoDB')
            return dataset_path
        
        except Exception as e:
            raise CustomException(e,sys)