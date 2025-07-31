from src.config.configuration import ConfigurationManager
from src.components.classifier_evaluation import ClassifierEvaluation

class ClassifierEvaluationPipeline:
    def __init__(self):
        pass
    def initiate_classifier_evaluation(self):
        config=ConfigurationManager()
        evaluation_config=config.get_classifier_evaluation_config()
        model_evaluation=ClassifierEvaluation(config=evaluation_config)
        model_evaluation.model_evaluator()