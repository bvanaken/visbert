
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
import waitress
from scipy.spatial.distance import cosine
import model
import visualize
import logging
import numpy as np
import random
import json
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import os
from utils import decode_text, current_milli_time
from data_utils import  NERSample
import argparse
import scipy
from scipy.special import softmax, log_softmax
# TODO you can create class here its job is to save the different computations each time and pass it to the next fucntion
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

base_route = ""
focus = ""
prediction = None
app = Flask(__name__, static_url_path='/static')


@app.route(base_route + "/")
def index():
    return render_template("demo.html")


@app.route(base_route + "/static/<path:filename>")
def image(filename):
    return send_file("./static/" + filename)

@app.route(base_route + "/data_type", methods=['POST'])
def change_data():
    data = request.get_json()
    model_name = data['model_name']

    data_name = model.get_data_type(model_name)

    output = {
        'model_name': model_name,
        'data_name': data_name,
    }

    return jsonify(output)

@app.route(base_route + "/dropdown", methods=['POST'])
def initialize_dropdown():
    data = request.get_json()
    model_name = data['model_name']
    sentence_id = decode_text(data["id"])
    sentence = decode_text(data["sentence"])
    labels = decode_text(data["labels"])
    mode = decode_text(data["mode"])
    ner_labels = data["ner_labels"]

    data = NERSample(sample_id="",
                     sentence=sentence.split(' &#& '),
                     sentence_labels="",
                     labels=[l for l in labels.split(" ") if l in ner_labels],
                     ner_labels=ner_labels,
                     mode=mode)

    annotated_tokens, agreement = model.initialize_dropdown(data, sentence_id, model_name)
    true_mistakes = find_true_mistakes(agreement)
    output = {
        'annotated_tokens': annotated_tokens.tolist(),
        'tab1': model.model1.tab_name,
        'tab2': model.model2.tab_name,
        'tab3': model.model3.tab_name,
        'tab4': model.model4.tab_name,
        'tab5': model.model5.tab_name,
        'agreement': agreement,
        'true_mistakes': true_mistakes
    }

    return jsonify(output)



@app.route(base_route + "/predict", methods=['POST'])
def get_output():
    data = request.get_json()
    input_sample = data["sample"]
    model_name = data["model"]
    random_replacement = decode_text(input_sample["random"])
    sentence = decode_text(input_sample["sentence"]).split(' &#& ')
    if random_replacement !='':
        randomly_replce_tokens(sentence, int(random_replacement))
    sentence_labels = decode_text(input_sample["sentence_labels"])
    labels = decode_text(input_sample["gold_standard"])
    focus = decode_text(input_sample["focus"])
    mode = decode_text(input_sample["mode"])
    ner_labels = data["ner_labels"]

    sample = NERSample(sample_id=decode_text(input_sample["id"]),
                       sentence=sentence,
                       sentence_labels=sentence_labels,
                       labels=[l for l in labels.split(" ") if l in ner_labels],
                       ner_labels= ner_labels,
                       mode = mode)

    layer_outputs, layers, hidden_states, attentions, features, attentions_heads = generate_model_output(sample, model_name)
    annotated_layers = get_labels_for_tokens(features, layers, focus)
    annotated_heads = [get_labels_for_tokens(features, head, focus) for head in attentions_heads]
    layers_predicted_tags = get_predicted_tags(layer_outputs)
    layer_focus_prediction = get_focus_prediction(layer_outputs, focus, features)

    layers_similarity = [compute_similarity(focus, layer[0], features) for layer in hidden_states]


    output = {
        'hidden_states': layers,
        'annotated_hidden_states': annotated_layers,
        'prediction': layer_focus_prediction,
        'predicted_tags': layers_predicted_tags,
        'tokens':features.tokens.tolist(),
        'mistakes': [l['mistakes'] for l in layer_outputs],
        'layers_similarity': layers_similarity,
        'attentions_heads': attentions_heads,
        'annotated_heads': annotated_heads,
    }

    return jsonify(output)


@app.route(base_route + "/attention", methods=['POST'])
def get_attention():
    data = request.get_json()
    input_sample = data["sample"]
    model_name = data["model"]
    sentence = decode_text(input_sample["sentence"]).split(' &#& ')

    sentence_labels = decode_text(input_sample["sentence_labels"])
    labels = decode_text(input_sample["gold_standard"])
    focus = decode_text(input_sample["focus"])

    layer_nr = int(input_sample['attention_layer'])
    head_nr = int(input_sample['attention_head'])
    mode = decode_text(input_sample["mode"])
    ner_labels = data["ner_labels"]


    sample = NERSample(sample_id=decode_text(input_sample["id"]),
                       sentence=sentence,
                       sentence_labels=sentence_labels,
                       labels=[l for l in labels.split(" ") if l in ner_labels],
                       ner_labels = ner_labels,
                       mode = mode)

    layer_outputs, layers, hidden_states, attentions, features, attentions_heads = generate_model_output(sample, model_name)
    attention_tokens = [remove_padding(layer[0], features.tokens) for layer in attentions]
    attention_summary = extract_attention_summary(attention_tokens[layer_nr][head_nr], features.annotated_tokens)
    local_global = extract_attention_local(attention_tokens[layer_nr][head_nr], features.annotated_tokens)
    head_similarity = compute_head_similarity(focus, attentions[layer_nr][0][head_nr], features)
    fig = px.imshow(attention_tokens[layer_nr][head_nr],
                    labels=dict(x="Attend To That Token", y="This Token", color="Attention Weight"),
                    x=features.annotated_tokens,
                    y=features.annotated_tokens
                    )

    fig.layout.height = 700
    fig.layout.width = 700
    attention_heat = fig.to_html(full_html=False)

    output = {
        'attention_heat': attention_heat,
        'attention_summary': attention_summary,
        'local_global': local_global,
        'head_similarity': head_similarity
    }

    return jsonify(output)

@app.route(base_route + "/impact", methods=['POST'])
def get_impact():
    data = request.get_json()
    input_sample = data["sample"]
    model_name = data["model"]
    sentence = decode_text(input_sample["sentence"]).split(' &#& ')

    sentence_labels = decode_text(input_sample["sentence_labels"])
    labels = decode_text(input_sample["gold_standard"])
    mode = decode_text(input_sample["mode"])
    ner_labels = data["ner_labels"]



    sample = NERSample(sample_id=decode_text(input_sample["id"]),
                       sentence=sentence,
                       sentence_labels=sentence_labels,
                       labels=[l for l in labels.split(" ") if l in ner_labels],
                       ner_labels=ner_labels,
                       mode= mode)

    layer_outputs, hidden_states, attentions, features = model.tokenize_and_predict(sample, model_name)
    change_matrix = model.compute_training_impact(model_name, features)
    change_fig = px.imshow(change_matrix,
                    labels=dict(x="Heads", y="Layers", color="Similarity Score"),
                    )
    change_fig.layout.height = 700
    change_fig.layout.width = 700
    change_heatmap = change_fig.to_html(full_html=False)

    output = {
        'change_heatmap': change_heatmap,
    }

    return jsonify(output)

def find_true_mistakes(agreement):
    true_mistakes = []
    for a in agreement:
        if not a:
            true_mistakes.append(a)
    return true_mistakes

def randomly_replce_tokens(sentence, n):
    for i in range(n):
        sentence[random.randrange(0, len(sentence))] = '[MASK]'
    return sentence

def get_predicted_tags(layer_outputs):
    layers_perdicted_tags = []
    for layer_output in layer_outputs:
        labels = [pr[0] for pr in layer_output['prediction_dict'][0]]
        values = [pr[1] for pr in layer_output['prediction_dict'][0]]
        layers_perdicted_tags.append({'labels': labels, 'values': values})
    return layers_perdicted_tags

def get_focus_prediction(layer_outputs, focus, features):
    layer_focus_prediction = []
    for layer_output in layer_outputs:
        labels = []
        focus_prediction = layer_output['prediction'][:, features.annotated_tokens.tolist().index(focus), :]
        for i, p in enumerate(focus_prediction[0]):
            labels.append(features.inv_label_map[i])
        layer_focus_prediction.append({'focus_prediction': (softmax(focus_prediction) * 10)[0].tolist(), 'labels': labels})
    return  layer_focus_prediction


def generate_model_output(sample, model_name):
    layer_outputs, hidden_states, attentions, features = model.tokenize_and_predict(sample, model_name)
    start_time = current_milli_time()
    # build pca-layer list from hidden states
    layers = extract_reduced_layers(hidden_states, features)
    attention_tokens = [remove_padding(layer[0], features.tokens) for layer in attentions]
    attentions_heads = extract_attention_head(attention_tokens, features)
    end_time = current_milli_time()
    logger.info("Postprocessing Time: {} ms".format(end_time - start_time))
    return layer_outputs, layers, hidden_states, attentions, features, attentions_heads


def compute_head_similarity(focus, head, features):
    labels = []
    similarity = []
    distance = []
    head = head[:len(features.annotated_tokens), :len(features.annotated_tokens)]
    # for layer in hidden_states:
    for t in features.annotated_tokens.tolist():
        focus_vector = head[features.annotated_tokens.tolist().index(focus)]
        if t != focus:
            token_vector = head[features.annotated_tokens.tolist().index(t)]
            labels.append(t)
            similarity.append(1 - cosine(focus_vector, token_vector))
            distance.append(cosine(focus_vector, token_vector))
    return {'labels':labels , 'similarity':similarity, 'distance': distance }

def compute_similarity(focus, layer, features):
    labels = []
    similarity = []
    distance = []
    for t, l in zip(features.annotated_tokens.tolist(), layer[:len(features.annotated_tokens)]):
        focus_vector = layer[features.annotated_tokens.tolist().index(focus)]
        if t != focus:
            token_vector = layer[features.annotated_tokens.tolist().index(t)]
            labels.append(t)
            similarity.append(1 - cosine(focus_vector, token_vector))
            distance.append(cosine(focus_vector, token_vector))
    return {'labels':labels , 'similarity':similarity, 'distance': distance }


def extract_reduced_layers(layers, features):
    tokens = features.tokens
    reduced_layers = []
    token_vectors = [layer[0][:len(tokens)] for layer in layers]

    flat_list = torch.cat(token_vectors)
    layer_reduced = visualize.reduce(flat_list, "pca", 2)

    for i, layer in enumerate(layers):

        pca_result_x = layer_reduced[0][len(tokens) * i:len(tokens) * i + len(tokens)]
        pca_result_y = layer_reduced[1][len(tokens) * i:len(tokens) * i + len(tokens)]
        points = []
        for i, val in enumerate(pca_result_x):
            point = {
                'x': val,
                'y': pca_result_y[i],
                'label': tokens[i]
            }

            points.append(point)
        reduced_layers.append(points)
    return reduced_layers

def extract_attention_head(token_vectors, features):
    tokens = features.tokens
    reduced_heads = []
    layers = []
    for heads in token_vectors:
        # unstakc the tensor from 12 seq len seq len to list of seq len seq len then cat
        # the input is 12 48 48 which means for each word we have 48 columns represent it but when we flatten them it becomes 576
        flat_list = torch.cat(torch.unbind(heads))
        layer_reduced = visualize.reduce(flat_list, "pca", 2)

        for i, head in enumerate(heads):

            pca_result_x = layer_reduced[0][len(tokens) * i:len(tokens) * i + len(tokens)]
            pca_result_y = layer_reduced[1][len(tokens) * i:len(tokens) * i + len(tokens)]

            # build json with point information
            # TODO here they build the poistion infromation for each token all you need to do is to provide slices that distinguish different tags then map that in the js file
            points = []
            for i, val in enumerate(pca_result_x):
                point = {
                    'x': val,
                    'y': pca_result_y[i],
                    'label': tokens[i]
                }

                points.append(point)
            reduced_heads.append(points)
        layers.append(reduced_heads)
        reduced_heads = []
    return layers
def extract_attention_summary(head, annotated_tokens):
    labels = []
    produced_weights = []
    recieved_weights = []
    def get_row(_list, row_idx):
        return sum(np.array(_list[row_idx]))

    def get_col(_list, col_idx):
        return sum(np.array(_list[:, col_idx]))

    for i,  token in enumerate(annotated_tokens):
        if annotated_tokens[0] == '[CLS]' and annotated_tokens[-1] == '[SEP]':
            if token.split('_')[-1] != '-100' and token != '[CLS]' and token != '[SEP]':
                labels.append(token)
                produced_sum = get_row(head, i)
                recieved_sum = get_col(head, i)
                produced_weights.append(produced_sum/len(annotated_tokens))
                recieved_weights.append(recieved_sum/len(annotated_tokens))
        else:
            if token.split('_')[-1] != '-100' and token != '<s>' and token != '</s>':
                labels.append(token)
                produced_sum = get_row(head, i)
                recieved_sum = get_col(head, i)
                produced_weights.append(produced_sum/len(annotated_tokens))
                recieved_weights.append(recieved_sum/len(annotated_tokens))
    return {'labels': labels, 'analysis1': produced_weights, 'analysis2': recieved_weights}
    #

def extract_attention_local(head, annotated_tokens):
    labels = []
    local_att = []
    local_att_wieght = []
    global_att = []
    global_att_weight = []

    def get_row(_list, row_idx):
        return np.array(_list[row_idx])

    for i,  token in enumerate(annotated_tokens):
        if annotated_tokens[0] == '[CLS]' and annotated_tokens[-1] == '[SEP]':
            if token.split('_')[-1] != '-100' and token != '[CLS]' and token != '[SEP]':
                labels.append(token)
                produced_weights = get_row(head, i)
                for t,w in zip(annotated_tokens, produced_weights):
                    if len(t.split('_'))>1 and t.split('_')[1] ==token.split('_')[1]:
                        local_att_wieght.append(w)
                    else:
                        global_att_weight.append(w)
                local_att.append(sum(local_att_wieght))
                global_att.append(sum(global_att_weight))
                local_att_wieght=[]
                global_att_weight=[]
        else:
            if token.split('_')[-1] != '-100' and token != '<s>' and token != '</s>':
                labels.append(token)
                produced_weights = get_row(head, i)
                for t, w in zip(annotated_tokens, produced_weights):
                    if len(t.split('_')) > 1 and t.split('_')[1] == token.split('_')[1]:
                        local_att_wieght.append(w)
                    else:
                        global_att_weight.append(w)
                local_att.append(sum(local_att_wieght))
                global_att.append(sum(global_att_weight))
                local_att_wieght = []
                global_att_weight = []
    return {'labels': labels, 'analysis1': local_att, 'analysis2': global_att}


def remove_padding(layer, tokens):
    return layer[:, :len(tokens), :len(tokens)]


def get_labels_for_tokens(features, layers, focus):
        o_points = []
        loc_points = []
        org_points = []
        pers_points =[]
        misc_points =[]
        ignore_points = []
        focus_points = []
        annotated_layers = []

        for l in layers:
            for i, token in enumerate(l):
                if token['label'] == features.tokens[i]:
                    token_label = features.annotated_tokens[i].split('_')[-1]
                    annotated_token = '_'.join(features.annotated_tokens[i].split('_')[:2])
                    if annotated_token == '_'.join(focus.split('_')[:2]):
                        l[i]['label'] = annotated_token
                        updated_layer = l[i]
                        focus_points.append(updated_layer)
                        continue
                    if token_label == 'O':
                        l[i]['label']= annotated_token
                        updated_layer = l[i]
                        o_points.append(updated_layer)
                    elif token_label == 'B-LOC':
                        l[i]['label']= annotated_token
                        updated_layer = l[i]
                        loc_points.append(updated_layer)
                    elif token_label == 'I-LOC':
                        l[i]['label']= annotated_token
                        updated_layer = l[i]
                        loc_points.append(updated_layer)
                    elif token_label == 'B-PERS':
                        l[i]['label']= annotated_token
                        updated_layer = l[i]
                        pers_points.append(updated_layer)
                    elif token_label == 'I-PERS':
                        l[i]['label']= annotated_token
                        updated_layer = l[i]
                        pers_points.append(updated_layer)
                    elif token_label == 'B-PER':
                        l[i]['label'] = annotated_token
                        updated_layer = l[i]
                        pers_points.append(updated_layer)
                    elif token_label == 'I-PER':
                        l[i]['label'] = annotated_token
                        updated_layer = l[i]
                        pers_points.append(updated_layer)
                    elif token_label == 'B-ORG':
                        l[i]['label']= annotated_token
                        updated_layer = l[i]
                        org_points.append(updated_layer)
                    elif token_label == 'I-ORG':
                        l[i]['label']= annotated_token
                        updated_layer = l[i]
                        org_points.append(updated_layer)
                    elif annotated_token == 'B-MISC':
                        l[i]['label']= annotated_token
                        updated_layer = l[i]
                        misc_points.append(updated_layer)
                    elif token_label == 'I-MISC':
                        l[i]['label']= annotated_token
                        updated_layer = l[i]
                        misc_points.append(updated_layer)
                    else:
                        l[i]['label'] = annotated_token
                        updated_layer = l[i]
                        ignore_points.append(updated_layer)
            annotated_layers.append({'o_points': o_points,
                           'loc_points':loc_points,
                           'org_points':org_points,
                           'pers_points':pers_points,
                           'misc_points':misc_points,
                           'ignore_points':ignore_points,
                           'focus_points': focus_points
                           })
            o_points = []
            loc_points = []
            org_points = []
            pers_points = []
            misc_points = []
            ignore_points = []
            focus_points = []
        return annotated_layers








def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_folder", help="Base Directory")
    parser.add_argument("data1_dir", help="ANERCorp Directory")
    parser.add_argument("data2_dir", help="NERCorp Directory")
    parser.add_argument("model1_errors", help="First Model Errors")
    parser.add_argument("model2_errors", help="Second Model Errors")
    parser.add_argument("model3_errors", help="Third Model Errors")
    parser.add_argument("model4_errors", help="Fourth Model Errors")
    parser.add_argument("model5_errors", help="Fifth Model Errors")
    parser.add_argument("model1_name", help="First Model Name")
    parser.add_argument("model2_name", help="Second Model Name")
    parser.add_argument("model3_name", help="Third Model Name")
    parser.add_argument("model4_name", help="Fourth Model Name")
    parser.add_argument("model5_name", help="Fifth Model Name")
    parser.add_argument("tab1_name", help="First Model Tab")
    parser.add_argument("tab2_name", help="Second Model Tab")
    parser.add_argument("tab3_name", help="Third Model Tab")
    parser.add_argument("tab4_name", help="Fourth Model Tab")
    parser.add_argument("tab5_name", help="Fifth Model Tab")
    parser.add_argument("model1_type", help="First Model Type")
    parser.add_argument("model2_type", help="Second Model Type")
    parser.add_argument("model3_type", help="Third Model Type")
    parser.add_argument("model4_type", help="Fourth Model Type")
    parser.add_argument("model5_type", help="Fifth Model Type")
    parser.add_argument("model1_preprocessing", help="First Model Preprocessing")
    parser.add_argument("model2_preprocessing", help="Second Model Preprocessing")
    parser.add_argument("model3_preprocessing", help="Third Model Preprocessing")
    parser.add_argument("model4_preprocessing", help="Fourth Model Preprocessing")
    parser.add_argument("model5_preprocessing", help="Fifth Model Preprocessing")
    args = parser.parse_args()
    logger.debug("Init BERT models")
    model.init(args)
    logger.debug("Run app")
    waitress.serve(app.run("0.0.0.0", port=1337))


if __name__ == '__main__':
    run()



