import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Input, Lambda
import tensorflow.keras.backend as K

class ConvModel(Model):

    def __init__(self, input_shape=None):
        super(ConvModel, self).__init__()
        self.resnet = ResNet50(include_top=False, input_shape=input_shape)
        self.resnet.trainable = False
        self.flatten = Flatten()
        self.dense = Dense(512)
        self.bn = BatchNormalization()

    def call(self, input):

        x = self.resnet(input)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.bn(x)

        return x
    
def calculate_distance(x):
    return K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True))


def distance_shape(shapes):
    shape1, _ = shapes
    return shape1[0], 1

def siamese(input_shape=None):

    left_image = Input(shape=input_shape)
    right_image = Input(shape=input_shape)

    conv_model = ConvModel(input_shape=input_shape)

    left_embedding = conv_model(left_image)
    right_embedding = conv_model(right_image)

    distance = Lambda(calculate_distance, output_shape=distance_shape)([left_embedding, right_embedding])
    pred = Dense(1, activation="sigmoid")(distance)

    model = Model(inputs=[left_image, right_image], outputs=pred)

    return model

def encode_image(model, img):
    img = img.resize((50, 50))
    img = np.array(img) / 255
    img = img[np.newaxis, :]
    return model.predict(img)