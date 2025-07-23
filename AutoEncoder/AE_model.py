
import keras
from keras import layers

def auto_encoder_model(input_dim,latent_dim,hidden_dims,drop_out=0.2):
    # Encoder block
    inputs_=keras.Input(shape=(input_dim,))
    x=inputs_
    for h_dim in hidden_dims:
        x = layers.Dense(h_dim)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    x = layers.Dropout(drop_out)(x)
    latent_layer = layers.Dense(latent_dim, activation='relu')(x)
    encoder = keras.Model(inputs_, latent_layer, name="Encoder")

    # Decoder block
    decoder_input = keras.Input(shape=(latent_dim,))
    x = decoder_input
    for h_dim in hidden_dims[::-1]:
        x = layers.Dense(h_dim)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    x = layers.Dropout(drop_out)(x)
    outputs_ = layers.Dense(input_dim, activation='linear')(x)
    decoder = keras.Model(decoder_input, outputs_, name="Decoder")

    # AutoEncoder
    ae_input = keras.Input(shape=(input_dim,))
    encoded_output = encoder(ae_input)
    decoded_output = decoder(encoded_output)
    auto_encoder = keras.Model(ae_input, decoded_output, name="AutoEncoder")

    return encoder, decoder, auto_encoder
    # for h_dim in hidden_dims:
    #     x=layers.Dense(h_dim,activation='relu')(x)
    # x=layers.Dropout(drop_out)(x)
    # latent_layer=layers.Dense(latent_dim,activation='relu')(x)
    # encoder=keras.Model(inputs_,latent_layer,name="Encoder")

    # # Decoder block
    # decoder_input=keras.Input(shape=(latent_dim,))
    # x=decoder_input
    # for h_dim in hidden_dims[::-1]:
    #     x=layers.Dense(h_dim,activation='relu')(x)
    # x=layers.Dropout(drop_out)(x)
    # outputs_=layers.Dense(input_dim,activation='linear')(x)
    # decoder=keras.Model(decoder_input,outputs_, name="decoder")
    # # AutoEncoder
    # ae_input=keras.Input(shape=(input_dim,))
    # encoded_output=encoder(ae_input)
    # decoded_output=decoder(encoded_output)
    # auto_encoder=keras.Model(ae_input,decoded_output,name="AutoEncoder")
    # return encoder,decoder, auto_encoder



