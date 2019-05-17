from flask import Flask, request, render_template, jsonify
import json
import os
import time
import random
import pandas as pd
import pickle
import numpy as np
import requests
# import webapi

# create app
app = Flask(__name__, static_url_path="")
bucket = 'https://s3.us-west-2.amazonaws.com/stonechild88'

@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')


@app.route('/put_token', methods=['GET','POST'])
def put_token():
    res = requests.put(bucket + '/token_data.json', data=request.data, headers={'Content-Type':'Application/json'})

    if res.status_code == 200:
        return jsonify({'result':'Authorized'})
    else:
        return jsonify({'result':'Authorization Failed'})
        


@app.route('/input_data', methods=['GET', 'POST'])
def input_data():
    input_data = json.loads(request.data)
    res = requests.put(bucket + '/input_data.json', data=request.data, headers={'Content-Type':'Application/json'})
    if res.status_code == 200:
        return jsonify({'start':input_data['start'], 'end': input_data['end']})
    else:
        return jsonify({'result':'Error in registering start and end time'})


@app.route('/get_prediction', methods=['GET'])
def build_model():
    try:
        import compute
        res = requests.get(bucket + '/pred.json')
        if res.status_code == 200:
            return jsonify(json.loads(res.text))
        else:
            return jsonify({'line1' : 'Error receiving the data', 'line2': ' '})
    except:
        return jsonify({'line1' : 'Error computing the data', 'line2': ' '})

# if __name__ == '__main__':
#     # unpickle 
#     # connect to the database
#     mc = pymongo.MongoClient(host="localhost", port=27017)
#     db = mc['fraud']
#     transactions_coll = db['transactions']
    
#     client = EventAPIClient()
#     client.collect()
#     app.run(host='0.0.0.0', port=8080, debug=True)



