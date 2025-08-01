from src import logger
from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_validation_pipeline import DataValidationPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.pipeline.model_training_pipeline import TrainingPipeline
from src.pipeline.classifier_training_pipeline import ClassifierTrainingPipeline
from src.pipeline.classifier_evaluation_pipeline import ClassifierEvaluationPipeline
from pathlib import Path

logger.info("Welcome to WindTurbine defect finding project")
# STAGE_NAME = "Data Ingestion Stage"
# try:
#     logger.info(f">>>>{STAGE_NAME} started <<<<")
#     obj=DataIngestionPipeline()
#     raw_data_path=obj.initiate_data_ingestion()
#     logger.info(f">>>> {STAGE_NAME} completed <<<<\n\n")
# except Exception as e:
#     raise e

# STAGE_NAME="Data Validation Stage"  


# try:
#         logger.info(f">>>> {STAGE_NAME} started <<<<")
#         obj=DataValidationPipeline()
#         raw_data_path='artifacts\data_ingestion\data\combined_raw_data.csv'
#         obj.initiate_data_validation(raw_data_path)
#         logger.info(f">>>> {STAGE_NAME} completed <<<<\n\n")
# except Exception as e:
#         logger.exception(e)
#         raise e
# STAGE_NAME="Data Transformation Stage" 
# try:
#     logger.info(f">>>>{STAGE_NAME} started <<<<")
#     obj=DataTransformationPipeline()
#     obj.initiate_data_transformation()
#     logger.info(f">>>> {STAGE_NAME} completed <<<<\n\n")
# except Exception as e:
#     raise e

# STAGE_NAME="Model Training Stage" 
# try:
#     logger.info(f">>>>{STAGE_NAME} started <<<<")
#     obj=TrainingPipeline()
#     obj.initiate_model_training()
#     logger.info(f">>>> {STAGE_NAME} completed <<<<\n\n")
# except Exception as e:
#     raise e
STAGE_NAME="Classifier Training Stage" 
try:
    logger.info(f">>>>{STAGE_NAME} started <<<<")
    obj=ClassifierTrainingPipeline()
    obj.initiate_classifier_training()
    logger.info(f">>>> {STAGE_NAME} completed <<<<\n\n")
except Exception as e:
    raise e
STAGE_NAME = "Classifier evaluation Stage"
try:
    logger.info(f">>>>{STAGE_NAME} started <<<<")
    obj=ClassifierEvaluationPipeline()
    obj.initiate_classifier_evaluation()
    logger.info(f">>>> {STAGE_NAME} completed <<<<\n\n")
except Exception as e:
    raise e