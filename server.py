# load Flask
from flask_cors import CORS, cross_origin
import flask
import pickle
import tensorflow as tf
from keras.layers import TextVectorization
from keras.models import load_model

from embedding import PositionalEmbedding
from transform import TransformerEncoder

app = flask.Flask(__name__)
# define a predict function as an endpoint
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/predict", methods=["GET"])
@cross_origin()
def predict():
    data = {"success": True, 'prediction': executePrediction()}
    return flask.jsonify(data)

with open('tv_layer.pkl', 'rb') as tv_file:
    from_disk = pickle.load(tv_file)
load_vectorizer = TextVectorization.from_config(from_disk['config'])
load_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
load_vectorizer.set_weights(from_disk['weights'])

def getGPTResponseData(logs):
    return '🚨Alert!! Based on the recent health checkup, the Cd-FMC is having trouble connecting to other devices, '

def executePrediction():
    threshold = 0.7
    logs = getDeviceLogs()
    # load the model, and pass in the custom metric function
    model = load_model('senti.h5', custom_objects={
        'PositionalEmbedding': PositionalEmbedding,
        'TransformerEncoder': TransformerEncoder})
    sentiment_score = model.predict(load_vectorizer([logs]))
    if sentiment_score < threshold:
        return '✅Cd-FMC health check-up completed. No issues found'
    return getGPTResponseData(logs)

def getDeviceLogs():
    return "I first saw this back in the early 90s on UK TV, i did like it then but i missed the chance to tape it, many years passed but the film always stuck with me and i lost hope of seeing it TV again, the main thing that stuck with me was the end, the hole castle part really touched me, its easy to watch, has a great story, great music, the list goes on and on, its OK me saying how good it is but everyone will take there own best bits away with them once they have seen it, yes the animation is top notch and beautiful to watch, it does show its age in a very few parts but that has now become part of it beauty, i am so glad it has came out on DVD as it is one of my top 10 films of all time. Buy it or rent it just see it, best viewing is at night alone with drink and food in reach so you don't have to stop the film.<br /><br />Enjoy"
# start the flask app, allow remote connections
app.run(host='0.0.0.0', port=5500)
