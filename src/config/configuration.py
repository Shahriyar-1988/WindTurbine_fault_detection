from src.constants import *
from src.entity.config_entity import (DataIngestionConfig,
                                      DataValidationConfig,
                                      DataTransformationConfig)
from src.utils.common import read_yaml,create_directory

class ConfigurationManager:
    def __init__(self,config_file_path=CONFIG_FILE_PATH,
                 params_file_path=PARAMS_FILE_PATH,
                 schema_file_path=SCHEMA_FILE_PATH):
        self.config=read_yaml(config_file_path)
        self.params=read_yaml(params_file_path)
        self.schema=read_yaml(schema_file_path)
    def get_data_ingestion_config(self)->DataIngestionConfig:
            config=self.config.data_ingestion
            create_directory([config.root_dir,config.data_dir])
            data_ingestion_config=DataIngestionConfig(
                root_dir=config.root_dir,
                data_url=config.data_url,
                data_dir=config.data_dir
            )

            return data_ingestion_config  
    
    def get_data_validation_config(self)->DataValidationConfig:
         config=self.config.data_validation
         create_directory([config.data_dir])
         data_validation_config=DataValidationConfig(
              data_dir=config.data_dir,
              status_file=config.STATUS_FILE
         )
         return data_validation_config
    def get_data_transformation_config(self)->DataTransformationConfig:
         config=self.config.data_transformation
         create_directory([config.root_dir])
         data_transformation_config=DataTransformationConfig(
              data_dir=config.data_dir, root_dir=config.root_dir
         )
         return data_transformation_config
    