import tensorflow as tf
import numpy as np

def start_block(input):
    X = tf.keras.layers.Conv2D(32,3,strides = 1, padding= 'same',name = 'conv_0',use_bias=False)(input)
    X = tf.keras.layers.BatchNormalization(name='bnorm_0')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1,name='leaky_0')(X)

    X = tf.keras.layers.ZeroPadding2D(1)(X) #size adjustments for downsampling
    X = tf.keras.layers.Conv2D(64,3,strides = 2,padding = 'valid',name = 'conv_1',use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_1')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_1')(X)
    return X

def block1(input):
    X = tf.keras.layers.Conv2D(32,1,strides=1,padding = 'same',name='conv_2',use_bias=False)(input)
    X = tf.keras.layers.BatchNormalization(name='bnorm_2')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1,name='leaky_2')(X)

    X = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', name='conv_3', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_3')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_3')(X)

    #residual layer
    X = X + input

    return X

def block2(input,block_index):
    i = block_index
    X = tf.keras.layers.Conv2D(64,1,strides=1,padding='same',name='conv_'+str(i),use_bias=False)(input)
    X = tf.keras.layers.BatchNormalization(name='bnorm_'+str(i))(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1,name='leaky_'+str(i))(X)

    X = tf.keras.layers.Conv2D(128,3,strides=1,padding='same',name='conv_'+str(i+1),use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_'+str(i+1))(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1,name='leaky_'+str(i+1))(X)

    #residual
    X = X + input

    return X

def block3(input,block_index):
    i = block_index
    X = tf.keras.layers.Conv2D(128,1,strides=1,padding = 'same',name='conv_'+str(i),use_bias=False)(input)
    X = tf.keras.layers.BatchNormalization(name='bnorm_'+str(i))(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1,name='leaky_'+str(i))(X)

    X = tf.keras.layers.Conv2D(256,3,strides=1,padding='same',name='conv_'+str(i+1),use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_' + str(i+1))(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_' + str(i+1))(X)

    #residual
    X = X + input

    return X

def block4(input,block_index):
    i = block_index
    X = tf.keras.layers.Conv2D(256, 1, strides=1, padding='same', name='conv_' + str(i), use_bias=False)(input)
    X = tf.keras.layers.BatchNormalization(name='bnorm_' + str(i))(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_' + str(i))(X)

    X = tf.keras.layers.Conv2D(512, 3, strides=1, padding='same', name='conv_' + str(i + 1), use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_' + str(i + 1))(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_' + str(i + 1))(X)

    # residual
    X = X + input

    return X

def block5(input,block_index):
    i = block_index
    X = tf.keras.layers.Conv2D(512, 1, strides=1, padding='same', name='conv_' + str(i), use_bias=False)(input)
    X = tf.keras.layers.BatchNormalization(name='bnorm_' + str(i))(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_' + str(i))(X)

    X = tf.keras.layers.Conv2D(1024, 3, strides=1, padding='same', name='conv_' + str(i + 1), use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_' + str(i + 1))(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_' + str(i + 1))(X)

    # residual
    X = X + input

    return X

def process_box(input,input_image,anchors):
    out_shape = input.get_shape().as_list()
    n_anchors = 3
    num_classes = 80

    input = tf.reshape(input, [-1, n_anchors * out_shape[1] * out_shape[2], 5 + num_classes])

    box_centers = input[:, :, 0:2]
    box_shapes = input[:, :, 2:4]
    confidence = input[:, :, 4:5]
    classes = input[:, :, 5:num_classes + 5]

    box_centers = tf.sigmoid(box_centers)
    confidence = tf.sigmoid(confidence)
    classes = tf.sigmoid(classes)

    anchors = tf.tile(anchors, [out_shape[1] * out_shape[2], 1])
    box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)

    x = tf.range(out_shape[1], dtype=tf.float32)
    y = tf.range(out_shape[2], dtype=tf.float32)

    cx, cy = tf.meshgrid(x, y)
    cx = tf.reshape(cx, (-1, 1))
    cy = tf.reshape(cy, (-1, 1))
    cxy = tf.concat([cx, cy], axis=-1)
    cxy = tf.tile(cxy, [1, n_anchors])
    cxy = tf.reshape(cxy, [1, -1, 2])

    strides = (input_image.shape[1] // out_shape[1], input_image.shape[2] // out_shape[2])
    box_centers = (box_centers + cxy) * strides

    prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

    return prediction

def YOLOv3Net(model_size):
    inputs = input_image = tf.keras.Input(shape=model_size)
    inputs = inputs/255.0 #normalization

    X = start_block(inputs)
    X = block1(X)

    #downsample
    X = tf.keras.layers.ZeroPadding2D(1)(X)
    X = tf.keras.layers.Conv2D(128,3,strides=2,padding='valid',name='conv_5',use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_5')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_5')(X)

    X = block2(X,6)
    X = block2(X,6+3)

    #downsample
    X = tf.keras.layers.ZeroPadding2D(1)(X)
    X = tf.keras.layers.Conv2D(256, 3, strides=2, padding='valid', name='conv_12', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_12')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_12')(X)

    X = block3(X,13)
    X = block3(X, 13 + 1*3)
    X = block3(X, 13 + 2*3)
    X = block3(X, 13 + 3*3)
    X = block3(X, 13 + 4*3)
    X = block3(X, 13 + 5*3)
    X = block3(X, 13 + 6*3)
    X = block3(X, 13 + 7*3)
    layer_36 = X

    # downsample
    X = tf.keras.layers.ZeroPadding2D(1)(X)
    X = tf.keras.layers.Conv2D(512, 3, strides=2, padding='valid', name='conv_37', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_37')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_37')(X)

    X = block4(X,38)
    X = block4(X, 38 + 1*3)
    X = block4(X, 38 + 2*3)
    X = block4(X, 38 + 3*3)
    X = block4(X, 38 + 4*3)
    X = block4(X, 38 + 5*3)
    X = block4(X, 38 + 6*3)
    X = block4(X, 38 + 7*3)
    layer_61 = X  # required later

    # downsample
    X = tf.keras.layers.ZeroPadding2D(1)(X)
    X = tf.keras.layers.Conv2D(1024, 3, strides=2, padding='valid', name='conv_62', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_62')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_62')(X)

    X = block5(X,63)
    X = block5(X, 63 + 1*3)
    X = block5(X, 63 + 2*3)
    X = block5(X, 63 + 3*3)

    ##############################################################################################

    X = tf.keras.layers.Conv2D(512, 1, strides=1, padding='same', name='conv_75', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_75')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_75')(X)

    X = tf.keras.layers.Conv2D(1024, 3, strides=1, padding='same', name='conv_76', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_76')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_76')(X)

    X = tf.keras.layers.Conv2D(512, 1, strides=1, padding='same', name='conv_77', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_77')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_77')(X)

    X = tf.keras.layers.Conv2D(1024, 3, strides=1, padding='same', name='conv_78', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_78')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_78')(X)

    X = tf.keras.layers.Conv2D(512, 1, strides=1, padding='same', name='conv_79', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_79')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_79')(X)
    X_route = X

    X = tf.keras.layers.Conv2D(1024, 3, strides=1, padding='same', name='conv_80', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_80')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_80')(X)

    X = tf.keras.layers.Conv2D(255,1,strides=1,padding='same',name='conv_81',use_bias=True)(X)

    anchors = [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]

    # output number 1
    out_pred = process_box(X,input_image,anchors[6:]) #layer 82

    #routing to layer -4
    X = X_route#layer 83

    X = tf.keras.layers.Conv2D(256, 1, strides=1, padding='same', name='conv_84', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_84')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_84')(X)

    X = tf.keras.layers.UpSampling2D(2)(X)#layer 85

    #routing again layers -1,61
    X = tf.concat([X,layer_61],axis=-1)#layer 86


    X = tf.keras.layers.Conv2D(256, 1, strides=1, padding='same', name='conv_87', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_87')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_87')(X)

    X = tf.keras.layers.Conv2D(512, 3, strides=1, padding='same', name='conv_88', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_88')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_88')(X)

    X = tf.keras.layers.Conv2D(256, 1, strides=1, padding='same', name='conv_89', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_89')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_89')(X)

    X = tf.keras.layers.Conv2D(512, 3, strides=1, padding='same', name='conv_90', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_90')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_90')(X)

    X = tf.keras.layers.Conv2D(256, 1, strides=1, padding='same', name='conv_91', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_91')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_91')(X)
    X_route = X

    X = tf.keras.layers.Conv2D(512, 3, strides=1, padding='same', name='conv_92', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_92')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_92')(X)

    X = tf.keras.layers.Conv2D(255, 1, strides=1, padding='same', name='conv_93', use_bias=True)(X)

    #ouput number 2
    prediction = process_box(X, input_image, anchors[3:6]) #layer 94
    out_pred = tf.concat([out_pred,prediction],axis=1)

    #routing back to layer -4
    X = X_route#layer 95

    X = tf.keras.layers.Conv2D(128,1,strides=1,padding='same',name='conv_96',use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_96')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_96')(X)
    X = tf.keras.layers.UpSampling2D(2)(X) #layer 97

    #routing (concatenating) layers -1 and 36
    X = tf.concat([X, layer_36], axis=-1) #layer 98

    X = tf.keras.layers.Conv2D(128, 1, strides=1, padding='same', name='conv_99', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_99')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_99')(X)

    X = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', name='conv_100', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_100')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_100')(X)

    X = tf.keras.layers.Conv2D(128, 1, strides=1, padding='same', name='conv_101', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_101')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_101')(X)

    X = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', name='conv_102', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_102')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_102')(X)

    X = tf.keras.layers.Conv2D(128, 1, strides=1, padding='same', name='conv_103', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_103')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_103')(X)

    X = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', name='conv_104', use_bias=False)(X)
    X = tf.keras.layers.BatchNormalization(name='bnorm_104')(X)
    X = tf.keras.layers.LeakyReLU(alpha=0.1, name='leaky_104')(X)

    X = tf.keras.layers.Conv2D(255, 1, strides=1, padding='same', name='conv_105', use_bias=True)(X)

    prediction = process_box(X, input_image, anchors[0:3])
    out_pred = tf.concat([out_pred, prediction], axis=1)


    model = tf.keras.Model(inputs = input_image,outputs = out_pred)
    #model.summary()
    return model

#model = YOLOv3Net((256,256,3))