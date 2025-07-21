from src.config.configuration import ConfigurationManager
from src.components.data_transformation import DataTransformation
from src import logger

class DataTransformationPipeline:
    def __init__(self):
        pass
    def initiate_data_transformation(self):
        config=ConfigurationManager()
        data_transformation_config=config.get_data_transformation_config()
        data_transformation=DataTransformation(data_transformation_config)
        data_transformation.data_transformer()

