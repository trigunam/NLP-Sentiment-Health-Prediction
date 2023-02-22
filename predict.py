# Load libraries
import flask
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model

from embedding import PositionalEmbedding
from transform import TransformerEncoder

max_length = 600
max_tokens = 20000
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
)

# instantiate flask
app = flask.Flask(__name__)

# load the model, and pass in the custom metric function
model = load_model('senti.h5', custom_objects={
                   'PositionalEmbedding': PositionalEmbedding,
                   'TransformerEncoder': TransformerEncoder})

i1 = 19999
i2 = 18999
i3 = 12999
i4 = text_vectorization("some text")

x = pd.DataFrame.from_dict(
    {'error': [i1, i2, i3, i4]}, orient='index').transpose()
print("Predicting accuracy for {} in % {}".format(
    [i1, i2, i3, i4], model.predict(x)))
