from flask import Flask, request, render_template, jsonify
import json
import os
import time
import random
import pandas as pd
import pickle
import numpy as np
# import webapi

# create app
app = Flask(__name__, static_url_path="")
    
@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')


@app.route('/get_token', methods=['GET','POST'])
def token():
    data = json.loads(request.data)
    with open('/Users/Sehokim/capstone/data/access_token.pkl', 'wb') as s:
        pickle.dump(data['access_token'], s)
    with open('/Users/Sehokim/capstone/data/refresh_token.pkl', 'wb') as e:
        pickle.dump(data['refresh_token'], e)
    return jsonify({'result':'authorized'})


@app.route('/input_data', methods=['GET', 'POST'])
def input_data():
    data = json.loads(request.data)
    with open('/Users/Sehokim/capstone/data/start.pkl', 'wb') as s:
        pickle.dump(data['start'], s)
    with open('/Users/Sehokim/capstone/data/end.pkl', 'wb') as e:
        pickle.dump(data['end'], e)
    return jsonify({'start': data['start'], 'end': data['end']})


@app.route('/get_prediction', methods=['GET'])
def build_model():
    import connect_api
    with open('/Users/Sehokim/capstone/data/prediction.pkl', 'rb') as pred:
        prediction = pickle.load(pred)
    return jsonify({'prediction' : prediction})


# if __name__ == '__main__':
#     # unpickle 
#     # connect to the database
#     mc = pymongo.MongoClient(host="localhost", port=27017)
#     db = mc['fraud']
#     transactions_coll = db['transactions']
    
#     client = EventAPIClient()
#     client.collect()
#     app.run(host='0.0.0.0', port=8080, debug=True)