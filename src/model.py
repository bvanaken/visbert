from collections import Counter

from transformers import BertModel, BertTokenizer, BertConfig
from arabert.preprocess import ArabertPreprocessor
from farasa.segmenter import FarasaSegmenter

import torch
from utils import current_milli_time
import logging
import os
from data_utils import SCDataset, SCModel, align_predictions, SecondSCDataset
import numpy as np
import json
from scipy.spatial import distance
import os
import uuid
from random import sample
from tqdm import tqdm

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

model1 = None
model2 = None
model3 = None
model4 = None



class BertNERModel:
    def __init__(self,  model_name: str, model_file: str, data_path: str, tab_name: str, model_type: str, cache_dir: str, device: str = "cpu"):
        self.model_name = model_name
        self.model_file = model_file
        self.model_data = self.read_arabic_data(data_path)
        self.model_labels = list(Counter([label for sentence in self.model_data for label in sentence[1]]).keys())
        self.tab_name = tab_name
        self.model_type = model_type
        self.cache_dir = cache_dir
        self.num_tag = len(self.model_labels)
        self.device = device

        self.model = self.load_model()
        self.ner_model, self.pretrained_ner_model = self.load_ner_model()
        self.tokenizer = self.load_tokenizer()
        self.preprocessor = self.load_preprocessor()

    def load_model(self):
        start_time = current_milli_time()
        config = BertConfig.from_pretrained(self.model_type, output_hidden_states=True, output_attentions=True, cache_dir=self.cache_dir)
        pretrained_weights = torch.load(self.model_file, map_location=torch.device('cpu'))
        model = BertModel.from_pretrained(self.model_type,state_dict=pretrained_weights,
                                                     config=config,
                                                     cache_dir=self.cache_dir)

        end_time = current_milli_time()
        logger.info("Model Loading Time: {} ms".format(end_time - start_time))
        return model

    def load_ner_model(self):
        pretrained_ner_model = SCModel(int(self.num_tag), self.model_type)
        ner_model = SCModel(int(self.num_tag), self.model_type)
        ner_model.load_state_dict(torch.load(self.model_file, map_location=torch.device('cpu')))
        return ner_model, pretrained_ner_model

    def load_tokenizer(self):
        return BertTokenizer.from_pretrained(self.model_type, cache_dir=self.cache_dir)

    def load_preprocessor(self):
        farasa_segmenter = FarasaSegmenter(interactive=True)
        return ArabertPreprocessor(self.model_type)

    def read_arabic_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = []
            sentence = []
            label = []
            for line in f:
                if len(line.split()) != 0:
                    if line.split()[0] == '.':
                        if len(sentence) > 0:
                            data.append((sentence, label))
                            sentence = []
                            label = []
                        continue
                    splits = line.split()
                    if 'TB' not in splits:
                        sentence.append(splits[0])
                        if len(splits) == 2:
                            label.append(splits[1])
                        else:
                            label.append(splits[-1])
        return data


def parse_model_output(ner_model, outputs,  sample, features):
        def softmax(x):
            f_x = np.exp(x) / np.sum(np.exp(x))
            return f_x
        #  TODO here we can parse the prediction where we have the logits of each token
        layer_logtis = []
        preds = None
        labels = None
        hidden_states = outputs[2]
        for layer in hidden_states:
            bo_tag = ner_model.bert_drop(layer)
            logits = ner_model.out_tag(bo_tag)
            if logits is not None:
                preds = logits if preds is None else torch.cat((preds, logits), dim=0)
            if features.labels is not None:
                labels = features.labels if labels is None else torch.cat((labels, features.labels), dim=0)
            preds = preds.detach().cpu().numpy()
            labels = labels.cpu().numpy()
            gold_standard_list, prediction_dict = align_predictions(preds, labels, features.inv_label_map)
            mistakes = identify_mistakes(gold_standard_list, prediction_dict)
            layer_logtis.append({"prediction": softmax(preds), "prediction_dict": prediction_dict, "mistakes": mistakes})
            preds = None
            labels = None
        return layer_logtis

def identify_mistakes(gold, prediction):
    mistakes = []
    for i, (t, p )in enumerate(zip(gold[0], prediction[0])):
        if t != p[0]:
            mistakes.append(f'Word ({i}) True: {t} -> Prediction: {p[0]}')
    return f'Number of mistakes: {len(mistakes)} Mistakes: {" # ".join(mistakes)}'




def initialize_dropdown(sample, model_name):
    flag = False
    if model_name == "squad":
        model = model1
    elif model_name == "hotpot":
        model = model2
    elif model_name == "babi":
        model = model3
        flag = True
    elif model_name == "v2SecondToken":
        model = model4
        flag = True
    else:
        raise Exception

    input_features = tokenize(sample, model.tokenizer, model.preprocessor, flag= flag)
    return input_features.annotated_tokens

def tokenize_and_predict(sample, model_name):
    flag = False
    if model_name == "squad":
        model = model1
    elif model_name == "hotpot":
        model = model2
    elif model_name == "babi":
        model = model3
        flag = True
    elif model_name == "v2SecondToken":
        model = model4
        flag = True
    else:
        raise Exception

    input_features = tokenize(sample, model.tokenizer, model.preprocessor, flag)

    with torch.no_grad():
        inputs = {'input_ids': input_features.input_ids,
                  'attention_mask': input_features.attention_mask,
                  'token_type_ids': input_features.token_type_ids
                  }

        start_time = current_milli_time()

        # Make Prediction
        if sample.mode == 'pretrained':
            outputs = model.pretrained_ner_model.bert(**inputs)
        else:
            outputs = model.model(**inputs)

        end_time = current_milli_time()
        logger.info("Prediction Time: {} ms".format(end_time - start_time))
        if sample.mode == 'pretrained':
            layer_outputs = parse_model_output(model.pretrained_ner_model, outputs, sample, input_features)
        else:
            layer_outputs = parse_model_output(model.ner_model, outputs, sample, input_features)
        return  layer_outputs, outputs[2], outputs[3], input_features


def tokenize(input_sample, tokenizer, preprocessor, flag = False):
    if flag:
        features_processing = SecondSCDataset(
            texts=[input_sample.sentence],
            tags=[input_sample.labels],
            ner_labels=input_sample.ner_labels,
            preprocessor=preprocessor,
            preprocess=True,
            tokenizer=tokenizer)
    else:
        features_processing = SCDataset(
            texts=[input_sample.sentence],
            tags=[input_sample.labels],
            ner_labels=input_sample.ner_labels,
            preprocessor=preprocessor,
            preprocess=True,
            tokenizer=tokenizer)

    features = features_processing.__getitem__(0)
    return features

def compute_training_impact(model_name, features):
    flag = False
    if model_name == "squad":
        model = model1
    elif model_name == "hotpot":
        model = model2
    elif model_name == "babi":
        model = model3
        flag = True
    elif model_name == "v2SecondToken":
        model = model4
        flag = True
    else:
        raise Exception
    training_impact = AttentionSimilarity(model.model_data,
                                          model.pretrained_ner_model.bert,
                                          model.ner_model.bert,
                                          10,
                                          model.tokenizer,
                                          model.preprocessor,
                                          features.tokens)
    impact_heatmap = training_impact.compute_similarity()
    similarities = []
    for sentence in training_impact.data:

        embedding_similarity = training_impact.compute_embedding_similarity(model.tokenizer.tokenize(model.preprocessor.preprocess(sentence)))
        similarities.append(embedding_similarity)

    return impact_heatmap, similarities



class AttentionSimilarity:
    def __init__(self, data, model1, model2, sample_size, tokeniser, preprocessor, tokens):
        self.data = data
        self.model1 = model1
        self.model2 = model2
        self.sample_size = sample_size
        self.tokenizer = tokeniser
        self.preprocessor = preprocessor
        self.tokens = tokens

    # def read_arabic_data(self, path):
    #     with open(path, 'r', encoding='utf-8') as f:
    #         data = []
    #         sentence = []
    #         for line in f:
    #             if len(line.split()) != 0:
    #                 if line.split()[0] == '.':
    #                     if len(sentence) > 0:
    #                         data.append(' '.join(sentence))
    #                         sentence = []
    #                     continue
    #                 splits = line.split()
    #                 if 'TB' not in splits:
    #                     sentence.append(splits[0])
    #     return data
    # def read_arabic_data(self, path):
    #     with open(path, 'r', encoding='utf-8') as f:
    #         data = []
    #         sentence = []
    #         label = []
    #         for line in f:
    #             if len(line.split()) != 0:
    #                 if line.split()[0] == '.':
    #                     if len(sentence) > 0:
    #                         data.append((sentence, label))
    #                         sentence = []
    #                         label = []
    #                     continue
    #                 splits = line.split()
    #                 if 'TB' not in splits:
    #                     sentence.append(splits[0])
    #                     if len(splits) == 2:
    #                         label.append(splits[1])
    #                     else:
    #                         label.append(splits[-1])
    #     return data

    def format_attention(self, attention):
        squeezed = []
        for layer_attention in attention:
            # 1 x num_heads x seq_len x seq_len
            if len(layer_attention.shape) != 4:
                raise ValueError(
                    "The attention tensor does not have the correct number of dimensions. Make sure you set "
                    "output_attentions=True when initializing your model.")
            squeezed.append(layer_attention.squeeze(0))
        # num_layers x num_heads x seq_len x seq_len
        return torch.stack(squeezed)

    def format_special_chars(self, tokens):
        return [t.replace('Ġ', ' ').replace('▁', ' ').replace('</w>', '') for t in tokens]

    def attention_matrix(self,
            attention=None,
            tokens=None,
            sentence_b_start=None,
            prettify_tokens=True,
            display_mode="dark",
            encoder_attention=None,
            decoder_attention=None,
            cross_attention=None,
            encoder_tokens=None,
            decoder_tokens=None,
    ):
        """Render model view

            Args:
                For self-attention models:
                    attention: list of ``torch.FloatTensor``(one for each layer) of shape
                        ``(batch_size(must be 1), num_heads, sequence_length, sequence_length)``
                    tokens: list of tokens
                    sentence_b_start: index of first wordpiece in sentence B if input text is sentence pair (optional)
                For encoder-decoder models:
                    encoder_attention: list of ``torch.FloatTensor``(one for each layer) of shape
                        ``(batch_size(must be 1), num_heads, encoder_sequence_length, encoder_sequence_length)``
                    decoder_attention: list of ``torch.FloatTensor``(one for each layer) of shape
                        ``(batch_size(must be 1), num_heads, decoder_sequence_length, decoder_sequence_length)``
                    cross_attention: list of ``torch.FloatTensor``(one for each layer) of shape
                        ``(batch_size(must be 1), num_heads, decoder_sequence_length, encoder_sequence_length)``
                    encoder_tokens: list of tokens for encoder input
                    decoder_tokens: list of tokens for decoder input
                For all models:
                    prettify_tokens: indicates whether to remove special characters in wordpieces, e.g. Ġ
                    display_mode: 'light' or 'dark' display mode
        """

        attn_data = []
        if attention is not None:
            if tokens is None:
                raise ValueError("'tokens' is required")
            if encoder_attention is not None or decoder_attention is not None or cross_attention is not None \
                    or encoder_tokens is not None or decoder_tokens is not None:
                raise ValueError("If you specify 'attention' you may not specify any encoder-decoder arguments. This"
                                 " argument is only for self-attention models.")
            attention = self.format_attention(attention)
            if sentence_b_start is None:
                attn_data.append(
                    {
                        'name': None,
                        'attn': attention.tolist(),
                        'left_text': tokens,
                        'right_text': tokens
                    }
                )
            else:
                slice_a = slice(0, sentence_b_start)  # Positions corresponding to sentence A in input
                slice_b = slice(sentence_b_start, len(tokens))  # Position corresponding to sentence B in input
                attn_data.append(
                    {
                        'name': 'All',
                        'attn': attention.tolist(),
                        'left_text': tokens,
                        'right_text': tokens
                    }
                )
                attn_data.append(
                    {
                        'name': 'Sentence A -> Sentence A',
                        'attn': attention[:, :, slice_a, slice_a].tolist(),
                        'left_text': tokens[slice_a],
                        'right_text': tokens[slice_a]
                    }
                )
                attn_data.append(
                    {
                        'name': 'Sentence B -> Sentence B',
                        'attn': attention[:, :, slice_b, slice_b].tolist(),
                        'left_text': tokens[slice_b],
                        'right_text': tokens[slice_b]
                    }
                )
                attn_data.append(
                    {
                        'name': 'Sentence A -> Sentence B',
                        'attn': attention[:, :, slice_a, slice_b].tolist(),
                        'left_text': tokens[slice_a],
                        'right_text': tokens[slice_b]
                    }
                )
                attn_data.append(
                    {
                        'name': 'Sentence B -> Sentence A',
                        'attn': attention[:, :, slice_b, slice_a].tolist(),
                        'left_text': tokens[slice_b],
                        'right_text': tokens[slice_a]
                    }
                )
        elif encoder_attention is not None or decoder_attention is not None or cross_attention is not None:
            if encoder_attention is not None:
                if encoder_tokens is None:
                    raise ValueError("'encoder_tokens' required if 'encoder_attention' is not None")
                encoder_attention = self.format_attention(encoder_attention)
                attn_data.append(
                    {
                        'name': 'Encoder',
                        'attn': encoder_attention.tolist(),
                        'left_text': encoder_tokens,
                        'right_text': encoder_tokens
                    }
                )
            if decoder_attention is not None:
                if decoder_tokens is None:
                    raise ValueError("'decoder_tokens' required if 'decoder_attention' is not None")
                decoder_attention = self.format_attention(decoder_attention)
                attn_data.append(
                    {
                        'name': 'Decoder',
                        'attn': decoder_attention.tolist(),
                        'left_text': decoder_tokens,
                        'right_text': decoder_tokens
                    }
                )
            if cross_attention is not None:
                if encoder_tokens is None:
                    raise ValueError("'encoder_tokens' required if 'cross_attention' is not None")
                if decoder_tokens is None:
                    raise ValueError("'decoder_tokens' required if 'cross_attention' is not None")
                cross_attention = self.format_attention(cross_attention)
                attn_data.append(
                    {
                        'name': 'Cross',
                        'attn': cross_attention.tolist(),
                        'left_text': decoder_tokens,
                        'right_text': encoder_tokens
                    }
                )
        else:
            raise ValueError("You must specify at least one attention argument.")

        # Generate unique div id to enable multiple visualizations in one notebook
        vis_id = 'bertviz-%s' % (uuid.uuid4().hex)

        # Compose html
        if len(attn_data) > 1:
            options = '\n'.join(
                f'<option value="{i}">{attn_data[i]["name"]}</option>'
                for i, d in enumerate(attn_data)
            )
            select_html = f'Attention: <select id="filter">{options}</select>'
        else:
            select_html = ""
        vis_html = f"""      
            <div id='%s'>
                <span style="user-select:none">
                    {select_html}
                </span>
                <div id='vis'></div>
            </div>
        """ % (vis_id)
        # they use loop because some times attentions are pointing to 2 sentences
        for d in attn_data:
            attn_seq_len_left = len(d['attn'][0][0])
            if attn_seq_len_left != len(d['left_text']):
                raise ValueError(
                    f"Attention has {attn_seq_len_left} positions, while number of tokens is {len(d['left_text'])} "
                    f"for tokens: {' '.join(d['left_text'])}"
                )
            attn_seq_len_right = len(d['attn'][0][0][0])
            if attn_seq_len_right != len(d['right_text']):
                raise ValueError(
                    f"Attention has {attn_seq_len_right} positions, while number of tokens is {len(d['right_text'])} "
                    f"for tokens: {' '.join(d['right_text'])}"
                )
            if prettify_tokens:
                d['left_text'] = self.format_special_chars(d['left_text'])
                d['right_text'] = self.format_special_chars(d['right_text'])

        params = {
            'attention': attn_data,
            'default_filter': "0",
            'display_mode': display_mode,
            'root_div_id': vis_id,
        }
        return params

    def compute_similarity(self):
        example = []

        for i in tqdm(range(len(sample(self.data, self.sample_size))), desc='Reading Data'):
            # index of the sentence = i zero to access the word list and 480 is the max length
            data = sample(self.data, self.sample_size)
            sentence_a = ' '.join(data[i][0])
            inputs = self.tokenizer.encode_plus(self.preprocessor.preprocess(sentence_a), return_tensors='pt', add_special_tokens=True)
            token_type_ids = inputs['token_type_ids']
            input_ids = inputs['input_ids']

            model1_attention = self.model1(input_ids, token_type_ids=token_type_ids)[-1]
            model2_attention = self.model2(input_ids, token_type_ids=token_type_ids)[-1]

            input_id_list = input_ids[0].tolist()  # Batch index 0
            attention_tokens = self.tokenizer.convert_ids_to_tokens(input_id_list)

            model1_att = self.attention_matrix(model1_attention, attention_tokens)
            model2_att = self.attention_matrix(model2_attention, attention_tokens)

            model1_mat = np.array(model1_att['attention'][0]['attn'])
            model2_mat = np.array(model2_att['attention'][0]['attn'])

            layer = []
            head = []

            for i in range(12):
                for j in range(12):
                    head.append(1 - distance.cosine(
                        model1_mat[i][j].flatten(),
                        model2_mat[i][j].flatten()
                    ))
                layer.append(head)
                head = []
            example.append(layer)
        similarity_matrix = np.array(example).mean(0)
        return similarity_matrix

    def compute_embedding_similarity(self, tokens):
        pretrained_embedding = self.model1.embeddings.word_embeddings.weight
        finetuned_embedding = self.model2.embeddings.word_embeddings.weight
        labels = []
        similarity = []
        for token in tokens:
            labels.append(token)
            similarity.append(1 - distance.cosine(
                pretrained_embedding[self.tokenizer.vocab[token]].detach().numpy(),
                finetuned_embedding[self.tokenizer.vocab[token]].detach().numpy()
            ))

        return {'labels': labels, 'similarity':similarity}


def init(args):
    global model1
    global model2
    global model3
    global model4

    m1: BertNERModel = BertNERModel(
        model_name=args.model1_name,
        model_file=os.path.join(args.base_folder, args.model1_name),
        data_path=os.path.join(args.base_folder, args.data1_dir),
        tab_name=args.tab1_name,
        model_type=args.model1_type,
        cache_dir=os.path.join(args.base_folder, "tmp")
    )
    model1 = m1
    logger.debug(f"Finished loading {args.model1_name}")

    # m2: BertNERModel = BertNERModel(
    #     model_path=args.model_dir,
    #     model_name=args.model2_name,
    #     model_file=os.path.join(args.model_dir, args.model2_name),
    #     tab_name=args.tab2_name,
    #     model_type=args.model3_type,
    #     cache_dir=os.path.join(args.model_dir, "tmp"),
    #     num_tag=args.num_tag)
    # model2 = m2
    # logger.debug(f"Finished loading {args.model2_name}")
    #
    # m3: BertNERModel = BertNERModel(
    #     model_path=args.model_dir,
    #     model_name=args.model3_name,
    #     model_file=os.path.join(args.model_dir, args.model3_name),
    #     tab_name=args.tab3_name,
    #     model_type=args.model1_type,
    #     cache_dir=os.path.join(args.model_dir, "tmp"),
    #     num_tag=args.num_tag)
    # model3 = m3
    # logger.debug(f"Finished loading {args.model3_name}")
    #
    # m4: BertNERModel = BertNERModel(
    #     model_path=args.model_dir,
    #     model_name=args.model4_name,
    #     model_file=os.path.join(args.model_dir, args.model4_name),
    #     tab_name=args.tab1_name,
    #     model_type=args.model3_type,
    #     cache_dir=os.path.join(args.model_dir, "tmp"),
    #     num_tag=args.num_tag)
    # model4 = m4
    # logger.debug(f"Finished loading {args.model4_name}")

