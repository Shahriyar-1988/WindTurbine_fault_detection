from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src import logger

# Data ingestion stage
class DataIngestionPipeline:
    def __init__(self):
        pass
    def initiate_data_ingestion(self):
        config=ConfigurationManager()
        data_ingestion_config=config.get_data_ingestion_config()
        data_ingestion=DataIngestion(data_ingestion_config)
        data_ingestion.download_data()
        final_data_path=data_ingestion.combined_data()
        return final_data_path
    