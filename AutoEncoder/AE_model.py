
import keras
from keras import layers

def auto_encoder_model(input_dim,latent_dim,hidden_dims,feature_per_sensor,drop_out=0.2):
# Encoder block
# The input dimension is the number of features only
    num_features=input_dim
    num_sensors=num_features//feature_per_sensor
    encoder_input=layers.Input(shape=(num_sensors,feature_per_sensor))
    x=encoder_input
    for h_dim in hidden_dims:
        x=layers.Conv1D(filters=h_dim,kernel_size=feature_per_sensor,padding="valid",activation="relu")(x)
        x=layers.BatchNormalization()(x)
    x=layers.Dropout(drop_out)(x)
    x=layers.Flatten()(x)
    latent_layer=layers.Dense(latent_dim,activation='relu',name="latent_vector")(x)
    encoder =keras.Model(encoder_input,latent_layer,name="encoder")
# Decoder block
    decoder_input = layers.Input(shape=(latent_dim,))
    # Match dimensions for reshaping
    x = layers.Dense(num_sensors * hidden_dims[-1], activation='relu')(decoder_input)
    x = layers.Reshape((num_sensors, hidden_dims[-1]))(x)

    for h_dim in hidden_dims[::-1][1:]:
        x = layers.Conv1DTranspose(
            filters=h_dim,
            kernel_size=feature_per_sensor,
            strides=2,
            padding="valid",
            activation="relu"
        )(x)
        x = layers.BatchNormalization()(x)

    x = layers.Dropout(drop_out)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(num_sensors * feature_per_sensor, activation='linear')(x)
    x = layers.Reshape((num_sensors, feature_per_sensor))(x)

    decoder_output = x  # shape should match (num_sensors, feature_per_sensor) or close

    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # ----- Autoencoder -----
    ae_input = encoder_input
    encoded_out = encoder(ae_input)
    decoded_out = decoder(encoded_out)
    autoencoder = keras.Model(ae_input, decoded_out, name="autoencoder")

    return encoder, decoder, autoencoder









    # # Encoder block
    # inputs_=keras.Input(shape=(input_dim,))
    # x=inputs_
    # for h_dim in hidden_dims:
    #     x = layers.Dense(h_dim)(x)
    #     x = layers.BatchNormalization()(x)
    #     x = layers.Activation('relu')(x)
    # x = layers.Dropout(drop_out)(x)
    # latent_layer = layers.Dense(latent_dim, activation='relu')(x)
    # encoder = keras.Model(inputs_, latent_layer, name="Encoder")

    # # Decoder block
    # decoder_input = keras.Input(shape=(latent_dim,))
    # x = decoder_input
    # for h_dim in hidden_dims[::-1]:
    #     x = layers.Dense(h_dim)(x)
    #     x = layers.BatchNormalization()(x)
    #     x = layers.Activation('relu')(x)
    # x = layers.Dropout(drop_out)(x)
    # outputs_ = layers.Dense(input_dim, activation='linear')(x)
    # decoder = keras.Model(decoder_input, outputs_, name="Decoder")

    # # AutoEncoder
    # ae_input = keras.Input(shape=(input_dim,))
    # encoded_output = encoder(ae_input)
    # decoded_output = decoder(encoded_output)
    # auto_encoder = keras.Model(ae_input, decoded_output, name="AutoEncoder")

    # return encoder, decoder, auto_encoder




