import os
from src.entity.config_entity import ClassifierTrainingConfig
from ClassifierHead.classifier_model import classifier_head
from src import logger
from src.constants import *
import numpy as np
import pandas as pd
import keras
from keras.callbacks import History
from src.utils.common import read_yaml
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from keras.metrics import Precision,AUC
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
     
        cls.compile(optimizer=keras.optimizers.Adam(learning_rate=self.params["learning_rate"]),
                        loss="binary_crossentropy",
                        metrics=["accuracy"]
                    )

        logger.info("Classifier model was compiled successfully!")
        model_path=os.path.join(self.config.root_dir,self.config.model_name)
        callbacks_=[
            keras.callbacks.EarlyStopping(monitor="val_loss",
                                                patience=5,
                                                restore_best_weights=True),
           keras.callbacks.ModelCheckpoint(filepath=model_path,save_best_only=True)
        ]
        batch_size = self.params.get("batch_size")
        epochs = self.params.get("epochs")
        val_split = self.params.get("validation_split")

        if None in [batch_size, epochs, val_split]:
            raise ValueError("One or more training parameters are None. Check params.yaml.")
    
        history = cls.fit(
            X,y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=val_split,
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
    def data_transformer(self,data:pd.DataFrame)->tuple:
        target_col=self.schema["TARGET_COLUMN"]
        normal_label=self.schema["NORMAL_LABEL"]
        X_data=data.drop(target_col,axis=1).to_numpy()
        y_data=data[target_col].apply(lambda x:1 if x==normal_label else 0).to_numpy()
        X_train,X_val,y_train,y_val = train_test_split(X_data,y_data,test_size=0.2, stratify=y_data)
        # Imputation and scaling
        cls_imputer=KNNImputer(n_neighbors=3)
        cls_scaler=StandardScaler()
        data_transformer=Pipeline(
            [("imputer",cls_imputer),
             ("scaler",cls_scaler),
             ]
        )
       
        X_transformed_train=data_transformer.fit_transform(X_train)
        X_transformed_val=data_transformer.transform(X_val)
         # imbalanced data management for training
        adasyn_=ADASYN(random_state=32)
        X_train_balanced,y_train_balanced=adasyn_.fit_resample(X_transformed_train,y_train)
        print(y_train_balanced.shape)
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        print(dict(zip(unique, counts)))

        
        X_train_balanced=X_train_balanced.reshape((-1,X_train_balanced.shape[1]//self.params["feature_per_sensor"],self.params["feature_per_sensor"]))
        logger.info(f"Training feature data was reshaped from {X_train.shape} into {X_train_balanced.shape} successfully!")
        X_transformed_val=X_transformed_val.reshape((-1,X_transformed_val.shape[1]//self.params["feature_per_sensor"],self.params["feature_per_sensor"]))
        logger.info(f"Validation feature data was reshaped from {X_val.shape} into {X_transformed_val.shape} successfully!")
    
        np.save(os.path.join(self.config.root_dir, "X_val_cls.npy"), X_transformed_val)
        np.save(os.path.join(self.config.root_dir, "y_val_cls.npy"), y_val)
        logger.info("Validation data was saved successfuly for evaluation!")
        return X_train_balanced,y_train_balanced
    
    def trainer(self):

        data=pd.read_csv(self.config.data_path)
        X_train,y_train=self.data_transformer(data)
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