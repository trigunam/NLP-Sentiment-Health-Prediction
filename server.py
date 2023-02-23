# load Flask
from flask_cors import CORS, cross_origin
import flask
import pickle
import tensorflow as tf
from keras.layers import TextVectorization
from keras.models import load_model

from datetime import datetime

from embedding import PositionalEmbedding
from transform import TransformerEncoder

app = flask.Flask(__name__)
# define a predict function as an endpoint
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/predict", methods=["GET"])
@cross_origin()
def predict():
    data = {"success": True, 'prediction': predict_log_data()}
    return flask.jsonify(data)

with open('tv_layer.pkl', 'rb') as tv_file:
    from_disk = pickle.load(tv_file)
load_vectorizer = TextVectorization.from_config(from_disk['config'])
load_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
load_vectorizer.set_weights(from_disk['weights'])

def get_gpt_response(logs):
    # https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior
    return datetime.now().strftime('%d-%b-%Y %I:%M:%S') + ' => 🚨 FMC-1 might be unstable!. Based on the recent periodic health check performed on your device, our systems predict that FMC-1 might fail, unless a corrective action is taken. For more details contact TAC support.'

def predict_log_data():
    threshold = 0.999998
    logs = get_device_logs()
    # load the model, and pass in the custom metric function
    model = load_model('fmc.h5', custom_objects={
        'PositionalEmbedding': PositionalEmbedding,
        'TransformerEncoder': TransformerEncoder})
    sentiment_score = model.predict(load_vectorizer([logs]))
    if sentiment_score > threshold:
        return '✅Cd-FMC health check-up completed. No issues found'
    return get_gpt_response(logs)


# Max 11 KB can be provided to predict
max_bytes_to_read = 11 * 1024


def get_device_logs():
    # Sentiment score: 0.9999938
    with open('dataset/train/critical/00383702.txt') as f:
        return f.read(max_bytes_to_read)

# start the flask app, allow remote connections
app.run(host='0.0.0.0', port=5500)
