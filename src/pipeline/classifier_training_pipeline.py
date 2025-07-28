from src.config.configuration import ConfigurationManager
from src.components.classifier_training import ClassifierTraining

class ClassifierTrainingPipeline:
    def __init__(self):
        pass
    def initiate_classifier_training(self):
        config=ConfigurationManager()
        cls_config=config.get_classifier_training_config()
        cls_training=ClassifierTraining(cls_config)
        cls_training.trainer()
        

