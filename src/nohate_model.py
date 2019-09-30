from pytorch_transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from data_utils import (convert_text_example_to_features)
import torch
from utils import current_milli_time
import logging
import os

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


def load_model(model_file, model_type, cache_dir):
    start_time = current_milli_time()

    # Load a pretrained model that has been fine-tuned
    config = BertConfig.from_pretrained(model_type, output_hidden_states=True, cache_dir=cache_dir)

    pretrained_weights = torch.load(model_file, map_location=torch.device('cpu'))
    model = BertForSequenceClassification.from_pretrained(model_type,
                                                          state_dict=pretrained_weights,
                                                          config=config,
                                                          cache_dir=cache_dir)

    end_time = current_milli_time()
    logger.info("Model Loading Time: {} ms".format(end_time - start_time))

    return model


def parse_model_output(output):
    logits = output[0]
    hidden_states = output[1]
    prediction = {}

    _, indices = torch.max(logits, 1)
    prediction["label"] = indices.item()

    softmax_result = softmax(logits)
    prediction["probability"] = softmax_result[0][prediction["label"]].item()

    return prediction, hidden_states


def tokenize_and_predict(sample, model_name):
    if model_name == "eng_large":
        model = eng_large_model
        tokenizer = large_tokenizer
    else:
        raise Exception

    example = InputExample(guid=0, text=sample["text"])

    input_features = tokenize(example, tokenizer)

    with torch.no_grad():
        inputs = {'input_ids': input_features.input_ids,
                  'attention_mask': input_features.input_mask,
                  'token_type_ids': input_features.segment_ids
                  }

        start_time = current_milli_time()

        # Make Prediction
        output = model(**inputs)

        end_time = current_milli_time()
        logger.info("Prediction Time: {} ms".format(end_time - start_time))

        # Parse Prediction
        prediction, hidden_states = parse_model_output(output)

        logger.info("Predicted Answer: {}".format(prediction["label"]))
        logger.info("Probability: {}".format(prediction["probability"]))

        return prediction, hidden_states, input_features


def tokenize(example, tokenizer):
    features = convert_text_example_to_features(example=example, tokenizer=tokenizer, max_seq_length=384)

    features.input_ids = torch.tensor([features.input_ids], dtype=torch.long)
    features.input_mask = torch.tensor([features.input_mask], dtype=torch.long)
    features.segment_ids = torch.tensor([features.segment_ids], dtype=torch.long)

    return features


def init(model_dir):
    global eng_large_model
    global large_tokenizer
    global softmax

    softmax = torch.nn.Softmax(dim=1)
    eng_large_model_file = os.path.join(model_dir, "nohate_eng_large.bin")
    cache_dir = os.path.join(model_dir, "tmp")

    eng_large_model = load_model(eng_large_model_file, 'bert-large-cased', cache_dir=cache_dir)
    large_tokenizer = BertTokenizer.from_pretrained('bert-large-cased', cache_dir=cache_dir, do_lower_case=False)
