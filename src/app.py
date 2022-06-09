import torch
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
import waitress

import model
import visualize
import logging
import os
from utils import decode_text, current_milli_time
from data_utils import get_question_indices, get_answer_indices, find_sup_char_ids
import argparse

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

base_route = ""
layer_wise = False

app = Flask(__name__, static_url_path='/static')


@app.route(base_route + "/")
def index():
    return render_template("demo.html")


@app.route(base_route + "/static/<path:filename>")
def image(filename):
    return send_file("./static/" + filename)


@app.route(base_route + "/predict", methods=['POST'])
def get_output():
    data = request.get_json()

    input_sample = data["sample"]
    model_name = data["model"]
    sup_ids = input_sample["sup_ids"] if "sup_ids" in input_sample else None

    answer_text = decode_text(input_sample["answer"])
    context = decode_text(input_sample["context"])

    answer_start = context.lower().find(answer_text.lower())

    answer_dict = {
        "text": answer_text,
        "answer_start": answer_start
    }

    print(sup_ids)

    if not sup_ids:
        if answer_start != -1 and answer_text != "":
            sup_ids = [find_sup_char_ids(context, answer_start)]

    sample = {"id": decode_text(input_sample["id"]),
              "question": decode_text(input_sample["question"]),
              "context": context,
              "answer": answer_dict,
              "sup_ids": sup_ids}

    prediction, layers, token_indices = generate_model_output(sample, model_name, layer_wise_reduction=layer_wise)

    output = {
        'hidden_states': layers,
        'prediction': prediction,
        'token_indices': token_indices
    }

    return jsonify(output)


def generate_model_output(sample, model_name, layer_wise_reduction=False):
    prediction, hidden_states, features = model.tokenize_and_predict(sample, model_name)

    start_time = current_milli_time()

    # build pca-layer list from hidden states
    tokens = features.tokens
    layers = []

    if layer_wise_reduction:

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

    else:

        token_vectors = [layer[0][:len(tokens)] for layer in hidden_states]

        flat_list = torch.cat(token_vectors)
        layer_reduced = visualize.reduce(flat_list, "pca", 2)

        for i, layer in enumerate(hidden_states):

            pca_result_x = layer_reduced[0][len(tokens) * i:len(tokens) * i + len(tokens)]
            pca_result_y = layer_reduced[1][len(tokens) * i:len(tokens) * i + len(tokens)]

            # build json with point information
            points = []
            for i, val in enumerate(pca_result_x):
                point = {
                    'x': val,
                    'y': pca_result_y[i],
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
        },
        'sups': [
        ]
    }

    for sup in features.sup_ids:
        print(tokens[sup[0]:sup[1] + 1])

        token_indices['sups'].append({'start': sup[0], 'end': sup[1]})

    end_time = current_milli_time()
    logger.info("Postprocessing Time: {} ms".format(end_time - start_time))

    return prediction, layers, token_indices


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="directory where model files are stored")
    parser.add_argument("layer_wise", help="whether to apply PCA reduction layer wise", action='store_false')
    args = parser.parse_args()
    global layer_wise
    layer_wise = args.layer_wise

    logger.debug("Init BERT models")
    model.init(args.model_dir)

    logger.debug("Run app")
    waitress.serve(app.run("0.0.0.0", port=1337))


if __name__ == '__main__':
    run()
