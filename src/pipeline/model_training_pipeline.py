from src.config.configuration import ConfigurationManager
from src.components.model_training import ModelTraining

class TrainingPipeline:
    def __init__(self):
        pass
    def initiate_model_training(self):
        config=ConfigurationManager()
        training_config=config.get_model_training_config()
        model_training=ModelTraining(training_config)
        model_training.trainer()

