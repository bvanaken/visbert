from pytorch_transformers import BertForQuestionAnswering, BertTokenizer, BertConfig
from data_utils import (read_squad_example, convert_example_to_features, parse_prediction, RawResult)
import torch
from utils import current_milli_time
import logging
import os

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

squad_model = None
hotpot_model = None
babi_model = None
base_tokenizer = None
large_tokenizer = None


def load_model(model_file, model_type, cache_dir):
    start_time = current_milli_time()

    # Load a pretrained model that has been fine-tuned
    config = BertConfig.from_pretrained(model_type, output_hidden_states=True, cache_dir=cache_dir)

    pretrained_weights = torch.load(model_file, map_location=torch.device('cpu'))
    model = BertForQuestionAnswering.from_pretrained(model_type,
                                                     state_dict=pretrained_weights,
                                                     config=config,
                                                     cache_dir=cache_dir)

    end_time = current_milli_time()
    logger.info("Model Loading Time: {} ms".format(end_time - start_time))

    return model


def parse_model_output(output, example, features):
    def to_list(tensor):
        return tensor.detach().cpu().tolist()

    result = RawResult(unique_id=1,
                       start_logits=to_list(output[0][0]),
                       end_logits=to_list(output[1][0]))

    nbest_predictions = parse_prediction(example, features, result)

    return nbest_predictions[0], output[2]  # top prediction, hidden states


def tokenize_and_predict(sample, model_name):

    if model_name == "squad":
        model = squad_model
        tokenizer = base_tokenizer
    elif model_name == "hotpot":
        model = hotpot_model
        tokenizer = large_tokenizer
    elif model_name == "babi":
        model = babi_model
        tokenizer = base_tokenizer
    else:
        raise Exception

    example = read_squad_example(sample)

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
        prediction, hidden_states = parse_model_output(output, example, input_features)

        logger.info("Predicted Answer: {}".format(prediction["text"]))
        logger.info("Start token: {}, End token: {}".format(prediction["start_index"], prediction["end_index"]))

        return prediction, hidden_states, input_features


def tokenize(example, tokenizer):
    features = convert_example_to_features(example=example,
                                           tokenizer=tokenizer,
                                           max_seq_length=384,
                                           doc_stride=128,
                                           max_query_length=64,
                                           is_training=False)

    features.input_ids = torch.tensor([features.input_ids], dtype=torch.long)
    features.input_mask = torch.tensor([features.input_mask], dtype=torch.long)
    features.segment_ids = torch.tensor([features.segment_ids], dtype=torch.long)
    features.cls_index = torch.tensor([features.cls_index], dtype=torch.long)
    features.p_mask = torch.tensor([features.p_mask], dtype=torch.float)

    return features


def init(model_dir):
    global squad_model
    global hotpot_model
    global babi_model
    global base_tokenizer
    global large_tokenizer

    squad_model_file = os.path.join(model_dir, "squad.bin")
    babi_model_file = os.path.join(model_dir, "babi.bin")
    hotpot_model_file = os.path.join(model_dir, "hotpot_distract.bin")
    cache_dir = os.path.join(model_dir, "tmp")

    squad_model = load_model(squad_model_file, 'bert-base-uncased', cache_dir=cache_dir)
    babi_model = load_model(babi_model_file, 'bert-base-uncased', cache_dir=cache_dir)
    base_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir, do_lower_case=True)

    hotpot_model = load_model(hotpot_model_file, 'bert-large-uncased', cache_dir=cache_dir)
    large_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', cache_dir=cache_dir, do_lower_case=True)
