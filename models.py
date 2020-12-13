from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose, Dense, Flatten, Reshape, InputLayer,Layer,MaxPool2D,Dropout
from tensorflow.keras.layers import InputSpec, Input, Dense,BatchNormalization,Activation,GlobalAveragePooling2D,Concatenate,Add
def vgg_block(input_layer,no_of_filters,no_of_conv,if_batch_normalization=False,if_dropout_dense=False,if_dropout_conv=False,
             dropout_value=0.5):
    for _ in range(no_of_conv):
        input_layer = Conv2D(no_of_filters,(3,3), padding='same')(input_layer)
        if if_batch_normalization:
            input_layer = BatchNormalization()(input_layer)
        input_layer = Activation('relu')(input_layer)
        if if_dropout_conv:
            input_layer = Dropout(dropout_value)(input_layer)
    input_layer = MaxPool2D((2,2), strides=(2,2))(input_layer)
    return input_layer
def Classifier_VGG_MNIST(input_shape=(28,28,1),if_batch_normalization=False,if_dropout_dense=False,
                  if_dropout_conv=False,dropout_value=0.5):
    input_ = Input(shape=input_shape)
    vgg_layer1 = vgg_block(input_,32,2,if_batch_normalization=if_batch_normalization,if_dropout_dense=if_dropout_dense,
                           if_dropout_conv=if_dropout_conv,dropout_value=dropout_value)
    vgg_layer2 = vgg_block(vgg_layer1,64,2,if_batch_normalization=if_batch_normalization,if_dropout_dense=if_dropout_dense,
                           if_dropout_conv=if_dropout_conv,dropout_value=dropout_value)
    vgg_layer3 = vgg_block(vgg_layer2,128,2,if_batch_normalization=if_batch_normalization,if_dropout_dense=if_dropout_dense,
                           if_dropout_conv=if_dropout_conv,dropout_value=dropout_value)
    x = Flatten()(vgg_layer3)
    if if_dropout_dense:
        x = Dropout(dropout_value)(x)
    x = Dense(256,activation="relu")(x)
    if if_dropout_dense:
        x = Dropout(dropout_value)(x)
    x = Dense(64,activation="relu")(x)
    output = Dense(10)(x)
    model = Model(inputs=input_,outputs=output)
    return model
def Classifier_VGG_CIFAR10(input_shape=(32,32,3),if_batch_normalization=False,if_dropout_dense=False,
                    if_dropout_conv=False,dropout_value=0.5):
    input_ = Input(shape=input_shape)
    vgg_layer1 = vgg_block(input_,64,2,if_batch_normalization=if_batch_normalization,if_dropout_dense=if_dropout_dense,
                           if_dropout_conv=if_dropout_conv,dropout_value=dropout_value)
    vgg_layer2 = vgg_block(vgg_layer1,128,2,if_batch_normalization=if_batch_normalization,if_dropout_dense=if_dropout_dense,
                           if_dropout_conv=if_dropout_conv,dropout_value=dropout_value)
    vgg_layer3 = vgg_block(vgg_layer2,256,2,if_batch_normalization=if_batch_normalization,if_dropout_dense=if_dropout_dense,
                           if_dropout_conv=if_dropout_conv,dropout_value=dropout_value)
    vgg_layer4 = vgg_block(vgg_layer3,512,2,if_batch_normalization=if_batch_normalization,if_dropout_dense=if_dropout_dense,
                           if_dropout_conv=if_dropout_conv,dropout_value=dropout_value)
    x = Flatten()(vgg_layer4)
    if if_dropout_dense:
        x = Dropout(dropout_value)(x)
    x = Dense(256,activation="relu")(x)
    if if_dropout_dense:
        x = Dropout(dropout_value)(x)
    x = Dense(64,activation="relu")(x)
    output = Dense(10)(x)
    model = Model(inputs=input_,outputs=output)
    return model
    