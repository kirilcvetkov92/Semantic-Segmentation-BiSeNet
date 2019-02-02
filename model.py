import tensorflow as tf
from keras.layers import *
from keras.applications import *
from keras.applications.xception import preprocess_input

def ConvBnAct(x, n_filters=64, kernel=(2,2), strides=(1,1), padding='valid', activation=tf.nn.relu):
        filters = n_filters

        conv_ =  Conv2D(filters=filters,
                      kernel_size = kernel,
                      strides = strides, 
                      padding=padding)

        batch_norm = BatchNormalization()

        activation = Activation(activation)

        x = conv_(x) 
        x = batch_norm(x)
        x = activation(x)

        return x


def ConvAct( x, n_filters, kernel=(1,1), activation = tf.nn.relu, pooling=False):

      poolingLayer = AveragePooling2D(pool_size=(1,1), padding='same')
      convLayer = Conv2D(filters = n_filters,
                         kernel_size = kernel,
                         strides=1)

      activation = Activation(activation)

      if pooling:
        x = poolingLayer(x)

      x = convLayer(x)
      x = activation(x)

      return x


def AttentionRefinmentModule(inputs, n_filters):
  
    filters = n_filters
    
    poolingLayer = AveragePooling2D(pool_size = (1,1), padding='same')
    
    x = poolingLayer(inputs)
    x = ConvBnAct(x, kernel = (1,1), n_filters = filters, activation = tf.nn.sigmoid)

    return multiply([inputs,x])



def FeatureFusionModule(input_f, input_s,n_filters):

    concate = Concatenate(axis=-1)([input_f, input_s])
    
    branch0 =  ConvBnAct(concate, n_filters=n_filters, kernel=(3, 3), padding='same') 
    branch_1  = ConvAct(branch0, n_filters=n_filters, pooling=True, activation = tf.nn.relu)
    branch_1 = ConvAct(branch_1, n_filters=n_filters, pooling=False, activation = tf.nn.sigmoid)
    
    x = multiply([branch0, branch_1])
    return  Add()([branch0, x])


def ContextPath(layer_13, layer_14):
    
    globalmax = GlobalAveragePooling2D()
    
    net_4 = AttentionRefinmentModule(layer_13, n_filters=1024)
    net_5 = AttentionRefinmentModule(layer_14, n_filters=2048)
    
    global_channels = globalmax(net_5)
    net_5_scaled = multiply([global_channels, net_5])
    
    net_4 = UpSampling2D(size=(4,4),  interpolation='bilinear')(net_4)
    net_5_scaled =  UpSampling2D(size=(4,4),  interpolation='bilinear')(net_5_scaled)

    
    cnc = Concatenate(axis=-1)([net_4, net_5_scaled])
 
    return cnc


def FinalModel(x, layer_13, layer_14):

    x = ConvBnAct(x, 32, strides=2)
    x = ConvBnAct(x, 64, strides=2)
    x = ConvBnAct(x, 156, strides=2)
    
    #context path
    cp = ContextPath(layer_13, layer_14)
    fusion = FeatureFusionModule(cp, x, 32)
    ans = UpSampling2D(size=(8,8),  interpolation='bilinear')(fusion)

    return ans


def get_model():
    inputs = Input(shape=(608,608,3))
    x = Lambda(lambda image: preprocess_input(image))(inputs)

    xception = Xception(weights='imagenet', input_shape=(608, 608, 3), include_top=False)

    tail_prev = xception.get_layer('block13_pool').output
    tail = xception.output

    output = FinalModel(x, tail_prev, tail)

    return inputs, xception.input, output

