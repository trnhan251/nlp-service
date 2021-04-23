from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
from model import Model

app = Flask(__name__)
model = Model('./nlp-models/fine-tuned-model.h5')
model.create_model()


@app.route("/")
def index():
    return 'NLP Service is running'


@app.route("/api/predict", methods=['POST'])
def prediction():
    content = request.json
    sentence1 = content['sentence1']
    sentence2 = content['sentence2']
    prediction_result = model.check_similarity(sentence1, sentence2)
    result = str(prediction_result['True'])
    return jsonify(result)


if __name__ == '__main__':
    app.run()
