import os
from src.config.configuration import ModelTrainingConfig
from AutoEncoder.AE_model import auto_encoder_model
from src import logger
from src.constants import PARAMS_FILE_PATH
import numpy as np
import pandas as pd
import keras
from src.utils.common import read_yaml
import matplotlib.pyplot as plt

class ModelTraining:
    def __init__(self,config:ModelTrainingConfig):
        self.config=config
        self.params=read_yaml(PARAMS_FILE_PATH)
    def model_fitter(self,input_dim:int,data:pd.DataFrame):

        encoder,decoder,model=auto_encoder_model(input_dim,
                                 latent_dim=self.params["latent_dim"],
                                 hidden_dims=self.params["hidden_dims"],drop_out=self.params["drop_out"])
        
        assert model is not None, "Model returned as None" # This will raise an assertion error if the model is not defined properly
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.params["learning_rate"]),
                        loss="mse",
                        metrics=['mse','mae', 'cosine_similarity']
                    )

        logger.info("AutoEncoder model was compiled successfully!")
        model_path=os.path.join(self.config.root_dir,self.config.model_name)
        callbacks_=[
            keras.callbacks.EarlyStopping(monitor="val_loss",
                                                patience=5,
                                                restore_best_weights=True),
           keras.callbacks.ModelCheckpoint(filepath=model_path,save_best_only=True)
        ]
        batch_size = self.params.get("batch_size")
        epochs = self.params.get("epochs")
        validation_split = self.params.get("validation_split")

        if None in [batch_size, epochs, validation_split]:
            raise ValueError("One or more training parameters are None. Check params.yaml.")
    

        history = model.fit(
            x=data,y=data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks_,
            verbose="auto"
        )
        logger.info("The model was fitted on the data and saved successfully!")
        metrics_df = pd.DataFrame(history.history)
        metrics_df.to_csv(os.path.join(self.config.metrics_dir, "training_metrics.csv"), index=False)
        logger.info("Metrics saved successfully!")
        
        encoder_path=os.path.join(self.config.root_dir,"encoder.keras")
        decoder_path=os.path.join(self.config.root_dir,"decoder.keras")
        encoder.save(encoder_path)
        decoder.save(decoder_path)
        logger.info("Encoder/Decoder saved successfully!")


        return history
    
    def trainer(self):
        train_data=pd.read_csv(self.config.train_data_path)
        history=self.model_fitter(input_dim=train_data.shape[1],data=train_data)
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
                plt.title(f'Training and Validation {metric}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                # Save the figure
                plot_path = os.path.join(save_dir, f"{metric}_training_plot.png")
                plt.savefig(plot_path)
                plt.close()




    



