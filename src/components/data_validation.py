import pandas as pd
from src import logger
from src.entity.config_entity import DataValidationConfig
from src.utils.common import read_yaml
from src.constants import SCHEMA_FILE_PATH
from pathlib import Path

class DataValidation:
    def __init__(self,config: DataValidationConfig):
        self.config=config
    def load_schema(self):
        schema = read_yaml(SCHEMA_FILE_PATH)
        return schema["COLUMNS"]
    def validate_data(self, df:pd.DataFrame)->bool:
        expected_schema=self.load_schema()
        validated=True
        # 1st: Check for missing columns
        missing_cols=[col for col in expected_schema.keys() if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in data: {missing_cols}")
            validated=False
        else: 
            logger.info("All expected columns are available!")
        #2nd: Check for datatype match
        for col, expected_dtype in expected_schema.items():
            actual_dtype=str(df[col].dtype)
            if actual_dtype!=expected_dtype:
                logger.warning(f"Column {col} datatype doesn't match with expected {expected_dtype}")
                validated=False
        #3rd: Check for missing values
        null_counts=df.isnull().sum()
        if null_counts.any():
            logger.warning(f"Dataset contains {null_counts[null_counts>0]} null values")


        # 4th: Check for duplications
        duplicated_rows=df.duplicated().sum()
        if duplicated_rows>0:
            logger.warning(f"Dataset contains {duplicated_rows} duplicated rows!")
        with open(self.config.status_file, 'w') as f:
                        f.write(f"Validation status: {validated}")
        if validated:
             validated_data_path= Path(self.config.data_dir)/"Validated_data.csv"
             df.to_csv(validated_data_path,index=False)
             
        return validated
    
            
