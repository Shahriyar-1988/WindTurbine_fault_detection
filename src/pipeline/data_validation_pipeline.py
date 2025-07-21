from src.config.configuration import ConfigurationManager
from src.components.data_validation import DataValidation
from pathlib import Path
import pandas as pd
from src import logger

class DataValidationPipeline:
    def __init__(self):
        pass
    def initiate_data_validation(self,data_path:Path):
        df=pd.read_csv(data_path)
        config=ConfigurationManager()
        data_validation_config=config.get_data_validation_config()
        data_validation=DataValidation(data_validation_config)
        data_validation.validate_data(df)



