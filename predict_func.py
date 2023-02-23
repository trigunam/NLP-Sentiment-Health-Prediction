# Load libraries
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model

from embedding import PositionalEmbedding
from transform import TransformerEncoder

from keras.layers import TextVectorization

# load the model, and pass in the custom metric function
model = load_model('fmc.h5', custom_objects={
                   'PositionalEmbedding': PositionalEmbedding,
                   'TransformerEncoder': TransformerEncoder})

with open('vocab.txt') as f:
    vocab = f.read()
    f.close()

max_tokens = 20000
text_vectorization = TextVectorization(vocabulary=eval(vocab))

input_values_to_predict = []

critical_values = []
noise_values = []

# Max 11 KB can be provided to predict
max_bytes_to_read = 11 * 1024

min_bytes_to_read = 1024

# Sentiment score: 0.99986863
# with open('dataset/train/critical/00383702.txt') as f:
#     critical_values.append(f.read(min_bytes_to_read))

# Sentiment score: 0.9999938
with open('dataset/train/critical/00383702.txt') as f:
    critical_values.append(f.read(max_bytes_to_read))

# Sentiment score: 0.9998738
# with open('dataset/train/critical/01428066.txt') as f:
#     critical_values.append(f.read(min_bytes_to_read))

# Sentiment score: 0.99999756
with open('dataset/train/critical/01428066.txt') as f:
    critical_values.append(f.read(max_bytes_to_read))

# Sentiment score: 0.99999756
with open('dataset/train/critical/03948840.txt') as f:
    critical_values.append(f.read(max_bytes_to_read))

# Sentiment score: 0.9999968
with open('dataset/train/critical/54415570.txt') as f:
    critical_values.append(f.read(max_bytes_to_read))

# Sentiment score: 0.99997026
# with open('dataset/train/noise/0FGq1NsY1.txt') as f:
#     noise_values.append(f.read(min_bytes_to_read))

# Sentiment score: 0.99999964
with open('dataset/train/noise/0FGq1NsY1.txt') as f:
    noise_values.append(f.read(max_bytes_to_read))

# Sentiment score: 0.999977
# with open('dataset/train/noise/0XDNcwdMZ.txt') as f:
#     noise_values.append(f.read(min_bytes_to_read))

# Sentiment score: 0.9999996
with open('dataset/train/noise/0XDNcwdMZ.txt') as f:
    noise_values.append(f.read(max_bytes_to_read))

# Sentiment score: 0.99999976
with open('dataset/train/noise/1nnidsGZN.txt') as f:
    noise_values.append(f.read(max_bytes_to_read))

# Sentiment score: 0.99999946
with open('dataset/train/noise/mJ2mtD3wI.txt') as f:
    noise_values.append(f.read(max_bytes_to_read))

# Sentiment score for:
# Critical   | Noise
# 0.9999938  | 0.99999964
# 0.99999756 | 0.9999996
# 0.99999756 | 0.99999976
# 0.9999968  | 0.99999946
# Use threshold 0.999998
# sc > 0.999998 => noise
# sc <= 0.999998 => critical

input_values_to_predict += critical_values
input_values_to_predict += noise_values

print("Sentiment score % \n{}".format(
    model.predict(text_vectorization(input_values_to_predict))))
