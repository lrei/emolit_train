"""Eval model from huggingface on gold."""
import os
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from mbdataset import MLDatasetWithFloats
from processors import MultiLabelTSVProcessor
from mutils import write_metrics, perform_inference

model_dir = "lrei/roberta-base-emolit"
print(f"Model = {model_dir}")
output_dir = "/data"
os.makedirs(output_dir, exist_ok=True)


gold_file = "./data/emolit/gold.tsv"
SEQLEN = 40
DEVICE = 0

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
os.makedirs(output_dir, exist_ok=True)


# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model = model.to(dtype=torch.float16, device=DEVICE)
model.eval()
param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))
print('model size: {}'.format(param_size))
tokenizer = AutoTokenizer.from_pretrained(model_dir)


# Load data
id2label = model.config.id2label
print(id2label)
label2id = model.config.label2id
if label2id is None:
    label2id = {v: k for k, v in id2label.items()}  # type: ignore


processor_tst = MultiLabelTSVProcessor(data_file=gold_file)
le = MLDatasetWithFloats.create_label_encoder_from_id2label(
    id2label=id2label
)
target_names = le.classes_.tolist()
tst_dataset = MLDatasetWithFloats(processor_tst, tokenizer, SEQLEN, le=le)
preds = perform_inference(model, tst_dataset)
labels = tst_dataset.get_label_ids()
results_file = os.path.join(output_dir, "results.txt")
write_metrics(
    labels,
    preds,
    target_names=target_names,
    output_file=results_file,
    expand_neutral=None,
)

