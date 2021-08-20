import os
import yaml
from azure.storage.blob import ContainerClient
from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
from model import Model

app = Flask(__name__)
model = Model('nlp-models')
model.create_model()


def load_config():
    dir_root = os.path.dirname(os.path.abspath(__file__))
    with open(dir_root + "/config.yaml", "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.FullLoader)


def get_files(dir):
    with os.scandir(dir) as entries:
        for entry in entries:
            if entry.is_file() and not entry.name.startswith('.'):
                yield entry


def upload(files, connection_string, container_name):
    container_client = ContainerClient.from_connection_string(connection_string, container_name)
    print("Uploading models to blob storage...")

    for file in files:
        blob_client = container_client.get_blob_client(file.name)
        with open(file.path, "rb") as data:
            blob_client.upload_blob(data)
            print(f'{file.name} uploaded to blob storage...')
            os.remove(file)
            print(f'{file.name} removed from local')


config = load_config()
print(*config)


@app.route("/")
def index():
    return 'NLP Service is running'


@app.route("/api/predict", methods=['POST'])
def prediction():
    content = request.json
    sentence1 = content['sentence1']
    sentence2 = content['sentence2']
    prediction_result = model.check_similarity(sentence1, sentence2)
    result = str(prediction_result)
    return jsonify(result)


if __name__ == '__main__':
    app.run(5002)
