from flask import Flask, redirect

app = Flask(__name__)

@app.route('/')
def home():
	return redirect('https://forecasleep.com')

if __name__ == '__main__':
	app.run(host='0.0.0.0', port='80')



