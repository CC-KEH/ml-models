import sys, os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import MainUtils
from src.constant import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()
        self.utils = MainUtils()
        
    def get_preprocessor(self):
        try:
            logging.info('Creating Preprocessor Object')
            preprocessor = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='mean')),
                ('scaler',StandardScaler()),
            ])
            
            logging.info('Saving Preprocessor in artifacts/preprocessor.pkl')
            preprocessor_path = self.data_tranformation_config.preprocessor_path
            os.makedirs(os.path.dirname(preprocessor_path),exist_ok=True)
            self.utils.save_object(path=preprocessor_path,obj=preprocessor)
            
            return (preprocessor,preprocessor_path)
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,dataset_path):
        
        preprocessor,preprocessor_path = self.get_preprocessor()
        
        #  Split Dataset into Train and Testing Set
        logging.info('Reading Dataset')
        df = pd.read_csv(dataset_path)
        
        logging.info('Segregating Dataset into Features and Label.')
        
        X = df[:,:-1]
        Y = df[:,-1]
        
        logging.info('Preprocessing Data')
        
        X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True)
        logging.info('Applying Preprocessor on Training and Testing Set.')
        
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)
        
        train_arr = np.c_[X_train_scaled,np.array(y_train)]
        test_arr = np.c_[X_test_scaled,np.array(y_test)]
        
        logging.info('Finished Data Transformation Successfully.')
        
        # Return Training and Testing Path
        return(train_arr,test_arr,preprocessor_path)
        