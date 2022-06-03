import joblib
from flask import Flask
from flask_restful import Api
from flask import request, jsonify

app = Flask(__name__)
api = Api(app)
@app.route('/')
@app.route('/index')
def home():
    return "aplikacja ze srodowiskiem produkcyjnym API"

@app.route('/api/predict_perceptron', methods=['GET'])
def predykcja():
    sepal_length = float(request.args.get('sl'))
    petal_length = float(request.args.get('pl'))
    dane = [sepal_length, petal_length]
    model = joblib.load('model.sav')
    pred = int(model.predict([dane]))
    return jsonify(features = dane, predicted_class = pred)



if __name__ == '__main__':
    app.run()