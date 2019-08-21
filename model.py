from pytorch_transformers import BertForQuestionAnswering, BertTokenizer, BertConfig
from utils_squad import (read_squad_example, convert_example_to_features, parse_prediction, RawResult)
import torch
from utils import current_milli_time
import logging
import os

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

model_path = "/Users/betty/Projekte/Datexis/QA/baselines/bert/models/squad/pytorch_model.bin"
model = None
tokenizer = None


def load_model(model_file):
    start_time = current_milli_time()

    # Load a pretrained model that has been fine-tuned
    config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)

    pretrained_weights = torch.load(model_file, map_location=torch.device('cpu'))
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased',
                                                     state_dict=pretrained_weights,
                                                     config=config)

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


def predict(sample):
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

        return prediction, hidden_states, input_features.tokens


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


def init():
    global model
    global tokenizer
    model = load_model(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
