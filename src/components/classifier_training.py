import os
from src.entity.config_entity import ClassifierTrainingConfig
from ClassifierHead.classifier_model import classifier_head
from src import logger
from src.constants import *
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.callbacks import History
from keras.utils import to_categorical
from src.utils.common import read_yaml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from beartype import beartype


class ClassifierTraining:
    def __init__(self,config:ClassifierTrainingConfig):
        self.config=config
        self.params=read_yaml(PARAMS_FILE_PATH).classifier_training
        self.schema=read_yaml(SCHEMA_FILE_PATH)
    @beartype
    def model_fitter(self,X:np.ndarray, y:np.ndarray)->History:
        cls=classifier_head(self.config.encoder_path,
                                   hidden_dims=self.params["hidden_dims"],
                                   drop_out=self.params["drop_out"],
                                   num_classes=self.params["num_classes"]
        )


        # Save the architecture of each model
        keras.utils.plot_model(cls, to_file=os.path.join(self.config.root_dir, "classifier_architecture.png"), show_shapes=True)
       
        # Save the summary of the autoencoder model
        cls_save_path=os.path.join(self.config.root_dir,"classifier_summary.txt")
        with open(cls_save_path,"w", encoding='utf-8') as f:
            cls.summary(print_fn=lambda x:f.write(x + "\n"))
        f1_score=keras.metrics.F1Score(average="macro")
        
        cls.compile(optimizer=keras.optimizers.Adam(learning_rate=self.params["learning_rate"]),
                        loss="binary_crossentropy",
                        metrics=[f1_score,"accuracy"]
                    )

        logger.info("Classifier model was compiled successfully!")
        model_path=os.path.join(self.config.root_dir,self.config.model_name)
        callbacks_=[
            keras.callbacks.EarlyStopping(monitor="val_loss",
                                                patience=10,
                                                restore_best_weights=True),
           keras.callbacks.ModelCheckpoint(filepath=model_path,save_best_only=True)
        ]
        batch_size = self.params.get("batch_size")
        epochs = self.params.get("epochs")
        validation_split = self.params.get("validation_split")

        if None in [batch_size, epochs, validation_split]:
            raise ValueError("One or more training parameters are None. Check params.yaml.")
        y_reshaped=y.reshape(-1,1) # to make y shape suitable for f1-score
    

        history = cls.fit(
            X,y_reshaped,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks_,
            verbose="auto"
        )
        logger.info("The model was fitted on the data and saved successfully!")
        metrics_df = pd.DataFrame(history.history)
        metrics_df.to_csv(os.path.join(self.config.metrics_dir, "classifier_training_metrics.csv"), index=False)
        logger.info("Metrics saved successfully!")
        
        cls_path=os.path.join(self.config.root_dir,"classifier.keras")
        cls.save(cls_path)
        logger.info("Classifier saved successfully!")


        return history
    
    def trainer(self):

        data=pd.read_csv(self.config.data_path)
        target_col=self.schema["TARGET_COLUMN"]
        normal_label=self.schema["NORMAL_LABEL"]
        X=data.drop(target_col,axis=1).to_numpy()
        X_data=X.reshape((-1,X.shape[1]//self.params["feature_per_sensor"],self.params["feature_per_sensor"]))
        logger.info(f"Classifier feature data was reshaped from {X.shape} into {X_data.shape} successfully!")
        y_data=data[target_col].apply(lambda x:1 if x==normal_label else 0).to_numpy()
        X_train,X_val,y_train,y_val = train_test_split(X_data,y_data,test_size=0.2, stratify=y_data)
        np.save(os.path.join(self.config.root_dir, "X_val_cls.npy"), X_val)
        np.save(os.path.join(self.config.root_dir, "y_val_cls.npy"), y_val)
        logger.info("Validation data was saved successfuly for evaluation!")
        history=self.model_fitter(X_train,y_train)
        self.history_plot(history,self.config.root_dir)
       

    @staticmethod
    def history_plot(history,save_dir):
         metrics=history.history.keys()
         for metric in metrics:
            if not metric.startswith('val_'):
                plt.figure(figsize=(8, 4))
                plt.plot(history.history[metric], label=f'Train {metric}')
                val_metric = f'val_{metric}'
                if val_metric in history.history:
                    plt.plot(history.history[val_metric], label=f'Val {metric}')
                plt.xlabel('Epoch')
                plt.ylabel(metric)
                plt.title(f'Classifier Training and Validation {metric}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                # Save the figure
                plot_path = os.path.join(save_dir, f"{metric}_classifier_plot.png")
                plt.savefig(plot_path)
                plt.close()