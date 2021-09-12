from keras.layers import Input, Conv2D, MaxPooling2D, Dropout,\
    BatchNormalization, concatenate, Conv2DTranspose, Activation
from keras.models import Model
# from keras.utils.vis_utils import plot_model


# two joined conv2d in every layer
def two_conv2d_layers(data_in, filters, kernel_size=3):
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   kernel_initializer='he_normal',
                   padding='same')(data_in)
    layer = Activation('relu')(BatchNormalization()(layer))
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   kernel_initializer='he_normal',
                   padding='same')(layer)
    layer = Activation('relu')(BatchNormalization()(layer))
    return layer


# descending layer
def down_layer(data_in, filters, kernel_size=3, dropout_rate=0.0):
    layer = two_conv2d_layers(data_in, filters, kernel_size)

    # for layer in the middle
    if dropout_rate != 0.0:
        drop = Dropout(dropout_rate)(MaxPooling2D((2, 2))(layer))
        return layer, drop

    return layer


# ascending layer
def up_layer(data_in, data_join, filters, kernel_size, dropout_rate):
    layer = Conv2DTranspose(filters=filters,
                            kernel_size=(kernel_size, kernel_size),
                            strides=(2, 2),
                            padding='same')(data_in)
    layer = Dropout(dropout_rate)(concatenate([layer, data_join]))
    layer = two_conv2d_layers(layer, filters, kernel_size)

    return layer


def unet():
    mix_in = Input(shape=(None, None, 1), name='input')
    loss_alg = "mean_squared_error"
    last_activation = "sigmoid"
    dropout_rate = 0.1
    filters = 8
    kernel_size = 3

    # down layers
    down_1, drop_1 = down_layer(mix_in, filters, kernel_size, dropout_rate)
    down_2, drop_2 = down_layer(drop_1, filters * 2, kernel_size, dropout_rate)
    down_3, drop_3 = down_layer(drop_2, filters * 4, kernel_size, dropout_rate)
    down_4, drop_4 = down_layer(drop_3, filters * 8, kernel_size, dropout_rate)
    down_5, drop_5 = down_layer(drop_4, filters * 16, kernel_size, dropout_rate)
    down_6, drop_6 = down_layer(drop_5, filters * 32, kernel_size, dropout_rate)

    # middle layer
    middle = down_layer(drop_6, filters * 64, kernel_size)

    # up layers
    up_6 = up_layer(middle, down_6, filters * 32, kernel_size, dropout_rate)
    up_5 = up_layer(up_6, down_5, filters * 16, kernel_size, dropout_rate)
    up_4 = up_layer(up_5, down_4, filters * 8, kernel_size, dropout_rate)
    up_3 = up_layer(up_4, down_3, filters * 4, kernel_size, dropout_rate)
    up_2 = up_layer(up_3, down_2, filters * 2, kernel_size, dropout_rate)
    up_1 = up_layer(up_2, down_1, filters, kernel_size, dropout_rate)

    vocal_out = Conv2D(1, (1, 1), activation=last_activation)(up_1)
    model = Model(inputs=mix_in, outputs=vocal_out)

    model.compile(loss=loss_alg, optimizer="adam", metrics=['accuracy'])
    print("Model has", model.count_params(), "params")

    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    return model


# shorter version of unet architecture, less parameters, quicker training
def unet_short():
    mix_in = Input(shape=(None, None, 1), name='input')
    loss_alg = "mean_squared_error"
    last_activation = "sigmoid"
    dropout_rate = 0.1
    filters = 8
    kernel_size = 3

    # down layers
    down_1, drop_1 = down_layer(mix_in, filters, kernel_size, dropout_rate)
    down_2, drop_2 = down_layer(drop_1, filters * 2, kernel_size, dropout_rate)
    down_3, drop_3 = down_layer(drop_2, filters * 4, kernel_size, dropout_rate)
    down_4, drop_4 = down_layer(drop_3, filters * 8, kernel_size, dropout_rate)

    # middle layer
    middle = down_layer(drop_4, filters * 16, kernel_size)

    # up layers
    up_4 = up_layer(middle, down_4, filters * 8, kernel_size, dropout_rate)
    up_3 = up_layer(up_4, down_3, filters * 4, kernel_size, dropout_rate)
    up_2 = up_layer(up_3, down_2, filters * 2, kernel_size, dropout_rate)
    up_1 = up_layer(up_2, down_1, filters, kernel_size, dropout_rate)

    vocal_out = Conv2D(1, (1, 1), activation=last_activation)(up_1)
    model = Model(inputs=mix_in, outputs=vocal_out)

    model.compile(loss=loss_alg, optimizer="adam", metrics=['accuracy'])
    print("Model has", model.count_params(), "params")

    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    return model
