##########################################################################
## Imports
##########################################################################
import os
from flask import Flask
from flask import request
from flask import render_template
from flask import url_for
from flask.json import jsonify
import numpy as np
import joblib
import pandas as pd
from keras.models import load_model

# Initialize the Flask application
app = Flask(__name__)

##########################################################################
## Routes
##########################################################################

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/api/hello")
def hello():
    """
    Return a hello message
    """
    return jsonify({"hello": "world"})

@app.route("/api/hello/<name>")
def hello_name(name):
    """
    Return a hello message with name
    """
    return jsonify({"hello": name})

@app.route("/api/whoami")
def whoami():
    """
    Return a JSON object with the name, ip, and user agent
    """
    return jsonify(
        name=request.remote_addr,
        ip=request.remote_addr,
        useragent=request.user_agent.string
    )

@app.route("/api/whoami/<name>")
def whoami_name(name):
    """
    Return a JSON object with the name, ip, and user agent
    """
    return jsonify(
        name=name,
        ip=request.remote_addr,
        useragent=request.user_agent.string
    )

model = load_model('model.h5')

# Define the classify endpoint
@app.route('/classify', methods=['POST'])
def classify():
    # Get the data from the request
    data = request.get_json()
    
    # Extract the row of pixels
    pixels = pd.DataFrame(data['pixels']).T
    
    # Use the model to make a prediction
    prediction = model.predict(pixels)
    print(prediction[0])
    
    prediction_list = prediction.tolist()

    return jsonify({'predict': prediction_list[0]})

# Start the Flask application
if __name__ == '__main__':
    app.run()




