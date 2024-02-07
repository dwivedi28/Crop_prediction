import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
class CustomData:
    def __init__(self,
                 N: float,
                 P: float,
                 K: float,
                 temperature: float,
                 humidity: float,
                 ph: float,
                 rainfall: float,
                 ):
        
        self.N = N
        self.P = P
        self.K = K
        self.temperature = temperature
        self.humidity = humidity
        self.ph = ph
        self.rainfall = rainfall

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'N': [self.N],
                'P': [self.P],
                'K': [self.K],
                'temperature': [self.temperature],
                'humidity': [self.humidity],
                'ph': [self.ph],
                'rainfall': [self.rainfall]  # Corrected column name
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise CustomException(e, sys)
