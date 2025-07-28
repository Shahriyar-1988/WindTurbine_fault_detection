from keras.models import Model,load_model
from keras.layers import Dense,Dropout

def classifier_head(encoder_path,hidden_dims,drop_out, num_classes=1):
    encoder=load_model(encoder_path)
    encoder.trainable=False

    # Build classifier head
    x=encoder.output
    for h_dim in hidden_dims:
        x=Dense(h_dim,activation='relu')(x)
    x=Dropout(drop_out,name="Classifier_dropout")(x)
    output_=Dense(num_classes,activation='sigmoid')(x)
    classifier=Model(inputs=encoder.input,outputs=output_, name="Classification_model")
    return classifier



