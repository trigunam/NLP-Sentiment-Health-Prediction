# Load libraries
import flask
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model

from embedding import PositionalEmbedding
from transform import TransformerEncoder

from keras.layers import TextVectorization

# instantiate flask
app = flask.Flask(__name__)

# load the model, and pass in the custom metric function
model = load_model('fmc.h5', custom_objects={
                   'PositionalEmbedding': PositionalEmbedding,
                   'TransformerEncoder': TransformerEncoder})

with open('vocab.txt') as f:
    vocab = f.read()
    f.close()

max_tokens = 20000
text_vectorization = TextVectorization(vocabulary=eval(vocab))

print("Predicting accuracy in % {}".format(
    model.predict(text_vectorization(["Could not detect session type", "The request has succeeded", "The server has fulfilled the request but does not need to return an entity-body, and might want to return updated meta-information"]))))
