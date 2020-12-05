import os
from transformers.data.processors.utils import DataProcessor, InputExample
from transformers import PreTrainedTokenizer
from tqdm import tqdm
import random
import code
from typing import List, Optional, Union
import json
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids_a: List[int]
    input_ids_b: List[int]
    attention_mask_a: Optional[List[int]] = None
    token_type_ids_a: Optional[List[int]] = None
    attention_mask_b: Optional[List[int]] = None
    token_type_ids_b: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

def dual_encoder_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding_a = tokenizer.batch_encode_plus(
        [(example.text_a) for example in examples], max_length=max_length, pad_to_max_length=True,
    )
    batch_encoding_b = tokenizer.batch_encode_plus(
        [(example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {}
        for k in batch_encoding_a:
            inputs[k + "_a"] = batch_encoding_a[k][i]
            inputs[k + "_b"] = batch_encoding_b[k][i]
            
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features



class NatcatProcessor(DataProcessor):
    """Processor for the Natcat data set."""  
    def __init__(self):
        super(NatcatProcessor, self).__init__()

    def get_examples(self, filepath):
        """See base class."""
        """
            filepath: the file of article-category pairs 
        """
        examples = []
        i = 0
        with open(filepath) as fin:
            lines = fin.read().strip().split("\n")
            for line in tqdm(lines):
                line = line.strip().split("\t")

                pos_cat = line[0]
                neg_cats = line[1:-1]
                article = line[-1]
                for neg_cat in neg_cats:
                    examples.append(InputExample(guid=i, text_a=pos_cat, text_b=article, label='1'))
                    i += 1
                    examples.append(InputExample(guid=i, text_a=neg_cat, text_b=article, label='0'))
                    i += 1


        return examples 

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

class EvalProcessor:
    def __init__(self, cat_file_path):
        super(EvalProcessor, self).__init__()
        self.cats = []
        with open(cat_file_path) as fin:
            for line in fin:
                self.cats.append(line.strip())

    def get_examples(self, filepath):
        """See base class."""
        """
            filepath: the file of the evaluation dataset 
        """
        examples = []
        labels = []
        i = 0
        with open(filepath) as fin:
            lines = fin.read().strip().split("\n")
            for line in tqdm(lines):
                line = line.strip().split(",", 1)
                if line[0].startswith("'") or line[0].startswith('"'):
                    line[0] = line[0][1:-1]
                label = int(line[0]) - 1
                text = " ".join(line[1][1:-1].split()[:128])
                if text.strip() == "":
                    text = "N/A"
                for cat in self.cats:
                    i += 1
                    if label >= len(self.cats):
                        examples.append(InputExample(guid=i, text_a=cat, text_b=text, label=1))
                    else: 
                        if cat == self.cats[label]:
                            examples.append(InputExample(guid=i, text_a=cat, text_b=text, label=1))
                        else:
                            examples.append(InputExample(guid=i, text_a=cat, text_b=text, label=0))

        return examples

    def get_labels(self):
        return [0, 1]

processors = {
    "natcat": NatcatProcessor,
    "eval": EvalProcessor,
}

output_modes = {
    "natcat": "classification",
    "eval": "classification",
}

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(task_name, preds, labels):
    if task_name in ["natcat"]:
        return {"acc": simple_accuracy(preds, labels)}
    if task_name in ["eval"]:
        return {"acc": simple_accuracy(preds, labels)}

class DataFiles:
    def __init__(self, directory):
        self.all_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(".data")]
        self.todo_files = self.all_files

    def next(self):
        if len(self.todo_files) == 0:
            return None
        return self.todo_files.pop()

    def save(self, file_path):
        with open(file_path, "w") as fout:
            for f in self.todo_files:
                fout.write(f + "\n")

    def load(self, file_path):
        self.todo_files = []
        with open(file_path) as fin:
            for f in fin:
                self.todo_files.append(f.strip())


