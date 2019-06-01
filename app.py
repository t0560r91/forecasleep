from flask import Flask, request, render_template, jsonify
import json
import os
import requests


# create app
app = Flask(__name__, static_url_path="")
bucket = 'https://s3.us-west-2.amazonaws.com/stonechild88'


# renders 'index.html' upon pageload
@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')


# put token to cloud storage
# send status_code to client
@app.route('/put_token', methods=['GET','POST'])
def put_token():
    res = requests.put(bucket + '/token_data.json', 
                        data=request.data, 
                        headers={'Content-Type':'Application/json'})
    if res.status_code == 200:
        return jsonify({'result':'Authorized'})
    else:
        return jsonify({'result':'Authorization Failed'})
        

# put input_data to cloud storage
# send status_code to client
@app.route('/input_data', methods=['GET', 'POST'])
def input_data():
    input_data = json.loads(request.data)
    res = requests.put(bucket + '/input_data.json', 
                        data=request.data, 
                        headers={'Content-Type':'Application/json'})
    if res.status_code == 200:
        return jsonify({'start':input_data['start'], 'end': input_data['end']})
    else:
        return jsonify({'result':'Error in registering start and end time'})


# run compute.py which will put a prediction data to cloud storage
# request prediction data from the cloud storage
# send to client
@app.route('/get_prediction', methods=['GET'])
def build_model():
    os.system("python compute.py")
    
    res = requests.get(bucket + '/pred.json')
    if res.status_code == 200:
        return jsonify(json.loads(res.text))
    else:
        return jsonify({'line1' : 'Error receiving the data', 'line2': ' '})

    # return jsonify({'line1' : 'Error computing the data', 'line2': ' '})


# prop the server listening to port 443 using ssl certs saved in the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', 
            port=443, 
            debug=True, 
            ssl_context=('/etc/letsencrypt/live/forecasleep.com/fullchain.pem',
                            '/etc/letsencrypt/live/forecasleep.com/privkey.pem'))



