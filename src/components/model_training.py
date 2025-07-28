import os
from src.entity.config_entity import ModelTrainingConfig
from AutoEncoder.AE_model import auto_encoder_model
from src import logger
from src.constants import PARAMS_FILE_PATH
import numpy as np
import pandas as pd
import keras
from src.utils.common import read_yaml
from keras.callbacks import History
from beartype import beartype
import matplotlib.pyplot as plt

class ModelTraining:
    def __init__(self,config:ModelTrainingConfig):
        self.config=config
        self.params=read_yaml(PARAMS_FILE_PATH).model_training
    @beartype
    def model_fitter(self,input_dim:int,data:np.ndarray)->History:

        encoder,decoder,model=auto_encoder_model(input_dim,
                                 latent_dim=self.params["latent_dim"],
                                 hidden_dims=self.params["hidden_dims"],
                                 feature_per_sensor=self.params["feature_per_sensor"],
                                 drop_out=self.params["drop_out"])
        # Save the architecture of each model
        keras.utils.plot_model(encoder, to_file=os.path.join(self.config.root_dir, "encoder_architecture.png"), show_shapes=True)
        keras.utils.plot_model(decoder, to_file=os.path.join(self.config.root_dir, "decoder_architecture.png"), show_shapes=True)
        keras.utils.plot_model(model, to_file=os.path.join(self.config.root_dir, "auto_encoder_architecture.png"), show_shapes=True)
        # Save the summary of the autoencoder model
        ae_save_path=os.path.join(self.config.root_dir,"auto_encoder_summary.txt")
        with open(ae_save_path,"w", encoding='utf-8') as f:
            model.summary(print_fn=lambda x:f.write(x + "\n"))
        
        assert model is not None, "Model returned as None" # This will raise an assertion error if the model is not defined properly
        huber_loss = keras.losses.Huber(delta=0.5)
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.params["learning_rate"]),
                        loss=huber_loss,
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
        latent_vector_path=os.path.join(self.config.root_dir,"latent_vector.np")
        latent_vectors = encoder.predict(data)
        np.save(latent_vector_path,latent_vectors)
        encoder.save(encoder_path)
        decoder.save(decoder_path)
        logger.info("Encoder/Decoder saved successfully!")


        return history
    
    def trainer(self):
        train_data=pd.read_csv(self.config.train_data_path)
        X_train=train_data.to_numpy().reshape((-1,train_data.shape[1]//self.params["feature_per_sensor"],self.params["feature_per_sensor"]))
        logger.info(f"Input data was reshaped from {train_data.shape} into {X_train.shape}")
        history=self.model_fitter(input_dim=X_train.shape[1]*X_train.shape[2],data=X_train)
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




    



