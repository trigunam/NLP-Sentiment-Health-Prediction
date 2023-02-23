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

input_values_to_predict.append(
    "16-Feb-2023 03: 14: 05.484, [ERROR], (PerlResource.java: 1391), NEW_DEVICE_LISTING_API_FLOW: Total time taken by new device list(GETALL) API is 0.108 seconds, com.cisco.api.external.rest.resource.PerlResource,  ajp-nio-127.0.0.1-9009-exec-5")

input_values_to_predict.append(str({"version": "7.3.0", "requestId": "78d23952-4b24-4761-90ac-8cf3487e1ec7", "data": {"userName": "admin", "subsystem": "API",
                                                                                                                      "message": "POST https://10.10.47.7/api/fmc_platform/v1/auth/revokeaccess No Content (204) - The server has fulfilled the request but does not need to return an entity-body, and might want to return updated meta-information", "sourceIP": "10.10.47.16", "domainUuid": "e276abec-e0f2-11e3-8169-6d9ed49b625f", "time": "1676819035756"}, "deleteList": []}))

input_values_to_predict.append(
    "11-Feb-2023 04: 44: 12.331, [INFO], (JsonRESTServerResource.java: 107)com.cisco.nm.vms.api.rest.SaveDomainChangesServerResource, ajp-nio-127.0.0.1-9009-exec-2 Data not a valid JSON.")

print("Predicting accuracy in % {}".format(
    model.predict(text_vectorization(input_values_to_predict))))
