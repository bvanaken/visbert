from flask import Flask, request, jsonify, render_template
import waitress

import model
import visualize
import logging
import os
from utils import decode_text
from utils_squad import get_question_indices, get_answer_indices
import argparse

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

base_route = "/visbert"

app = Flask(__name__, static_url_path='/static')


@app.route(base_route + "/")
def index():
    return render_template("demo.html")


@app.route(base_route + "/predict", methods=['POST'])
def get_output():
    data = request.get_json()

    logger.info(data)

    input_sample = data["sample"]
    model_name = data["model"]

    logger.info(input_sample)

    answer_text = decode_text(input_sample["answer"])
    context = decode_text(input_sample["context"])

    answer_dict = {
        "text": answer_text,
        "answer_start": context.find(answer_text)
    }

    sample = {"id": decode_text(input_sample["id"]),
              "question": decode_text(input_sample["question"]),
              "context": context,
              "answer": answer_dict}

    prediction, layers, token_indices = generate_model_output(sample, model_name)

    output = {
        'hidden_states': layers,
        'prediction': prediction,
        'token_indices': token_indices
    }

    return jsonify(output)


def generate_model_output(sample, model_name):

    prediction, hidden_states, features = model.tokenize_and_predict(sample, model_name)

    # build pca-layer list from hidden states
    tokens = features.tokens
    layers = []
    for layer in hidden_states:

        # cut off padding
        token_vectors = layer[0][:len(tokens)]

        # dimensionality reduction
        layer_reduced = visualize.reduce(token_vectors, "pca", 2)

        # build json with point information
        points = []
        for i, val in enumerate(layer_reduced[0]):
            point = {
                'x': val,
                'y': layer_reduced[1][i],
                'label': tokens[i]
            }

            points.append(point)

        layers.append(points)

    # build indices object
    question_indices = get_question_indices(tokens)
    # answer_indices = get_answer_indices(features) ## Decide whether to highlight ground truth or predicted answer
    answer_indices = prediction["start_index"], prediction["end_index"] + 1

    token_indices = {
        'question': {
            'start': question_indices[0],
            'end': question_indices[1],
        },
        'answer': {
            'start': answer_indices[0],
            'end': answer_indices[1],
        }
    }

    return prediction, layers, token_indices


def run():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("model_dir", help="directory where model files are stored")
    # args = parser.parse_args()

    logger.debug("Init BERT models")
    model.init()

    logger.debug("Run app")
    app.run(host='localhost', port=1337)
    # waitress.serve(app.run("0.0.0.0", port=1337))


if __name__ == '__main__':
    run()
