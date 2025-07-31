from src.constants import *
from src.entity.config_entity import (DataIngestionConfig,
                                      DataValidationConfig,
                                      DataTransformationConfig,
                                      ModelTrainingConfig,
                                      ClassifierTrainingConfig,
                                      ClassifierEvaluationConfig)
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
    def get_model_training_config(self)->ModelTrainingConfig:
         config=self.config.model_training
         create_directory([config.root_dir,
                           config.metrics_dir])
         model_training_config=ModelTrainingConfig(
              root_dir=config.root_dir,
              train_data_path=config.train_data_path,
              metrics_dir=config.metrics_dir,
              model_name=config.model_name
         )
         return model_training_config
    def get_classifier_training_config(self)->ClassifierTrainingConfig:
         config=self.config.classifier_training
         create_directory([config.root_dir,
                           config.metrics_dir])
         classifier_training_config=ClassifierTrainingConfig(
              root_dir=config.root_dir,
              data_path=config.data_path,
              metrics_dir=config.metrics_dir,
              encoder_path=config.encoder_path,
              model_name=config.model_name

         )
         return classifier_training_config
    def get_classifier_evaluation_config(self)->ClassifierEvaluationConfig:
          config=self.config.classifier_evaluation
          create_directory([config.metrics_file_path])
          classifier_evaluation_config=ClassifierEvaluationConfig(
               test_data_paths=config.test_data_paths,
               cls_model_path=config.cls_model_path,
               metrics_file_path=config.metrics_file_path
          )
          return classifier_evaluation_config
    
    