import os
from pathlib import Path
from src.entity.config_entity import ClassifierEvaluationConfig
import mlflow
from src import logger
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix, f1_score, precision_score,accuracy_score

class ClassifierEvaluation:
    def __init__(self,config:ClassifierEvaluationConfig):
        self.config=config
    @staticmethod
    def evaluation_metrics(actual,predict):
        cls_rep = classification_report(actual,predict, output_dict=True,digits=3)
        f1=f1_score(actual,predict,average='macro')
        pr=precision_score(actual,predict,average='macro')
        acc=accuracy_score(actual,predict)
    
        return cls_rep, f1,pr,acc
    def model_logger(self,cls_model,X_test,y_test):

        y_pred=(cls_model.predict(X_test)>0.5).astype(int).flatten()
        y_test=y_test.flatten()
        cls_rep,f1,prec,acc=self.evaluation_metrics(y_test,y_pred)
        load_dotenv()
        MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"Turbine fault detection experiment")
        with mlflow.start_run():

            mlflow.log_dict(cls_rep,"classification_report.json")
            mlflow.log_metrics({"f1":round(f1,3),"precision":round(prec,3), "accuracy":round(prec,3)})
            cm=confusion_matrix(y_test,y_pred)
            disp=ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="viridis",values_format='d')
            plt.title(f"Confusion Matrix for fault detection model")
            plt.tight_layout()
            cm_path = os.path.join(self.config.metrics_file_path,"confusion_matrix.jpg")
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)

    def model_evaluator(self):
        X_test=np.load(self.config.test_data_paths[0])
        y_test=np.load(self.config.test_data_paths[1])
        cls_model= load_model(self.config.cls_model_path)
        self.model_logger(cls_model,X_test,y_test)




