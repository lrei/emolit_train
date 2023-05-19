"""Train classifier with soft targets."""

import os
import shutil
import logging
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
)

from processors import SoftMultiLabelTSVProcessor, MultiLabelTSVProcessor
from mbdataset import MultiSoftTrainDataset, MLDatasetWithFloats
from mutils import compute_metrics_soft, write_metrics
from loss import SoftBCETrainer


# Data
TRAIN_FILE = "trn.tsv"
VAL_FILE = "val.tsv"
DATA_DIR = "./data/emolit"
TST_FILE = "./data/emolit/gold.tsv"
output_dir = "./model"

# Params
MODEL_NAME = "roberta-large"
SEQLEN = 48
BS = 16
NUM_EPOCHS = 10
LOAD_BEST = True
os.makedirs(output_dir, exist_ok=True)

# logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


# Create the tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # ensure parallel
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
collator_fn = DataCollatorWithPadding(
    tokenizer, padding=True, pad_to_multiple_of=8
)
# Load the data
processor_trn = SoftMultiLabelTSVProcessor(
    data_file=os.path.join(DATA_DIR, TRAIN_FILE)
)
processor_val = SoftMultiLabelTSVProcessor(
    data_file=os.path.join(DATA_DIR, VAL_FILE)
)
processor_tst = MultiLabelTSVProcessor(data_file=TST_FILE)

train_dataset = MultiSoftTrainDataset(processor_trn, tokenizer, SEQLEN)
val_dataset = MultiSoftTrainDataset(
    processor_val, tokenizer, SEQLEN, mlb=train_dataset.mlb
)
tst_dataset = MLDatasetWithFloats(
    processor_tst, tokenizer, SEQLEN, le=train_dataset.get_label_encoder(),
)
target_names = train_dataset.get_target_names()
# train_dataset.print_random_examples(n=4)
# val_dataset.print_random_examples(n=4)

logger.info(f"Train: {len(train_dataset)} examples")
logger.info(f"Val: {len(val_dataset)} examples")
# Create the model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=train_dataset.num_labels,
    problem_type="multi_label_classification",
    label2id=train_dataset.label2id,
    id2label=train_dataset.id2label,
)

os.makedirs(output_dir, exist_ok=True)
training_args = TrainingArguments(
    output_dir=output_dir,
    do_train=True,
    per_device_train_batch_size=BS,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    num_train_epochs=NUM_EPOCHS,
    warmup_ratio=0.0,
    lr_scheduler_type="linear",
    weight_decay=0.0,
    fp16=True,
    logging_steps=2000,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=LOAD_BEST,
    do_eval=True,
    per_device_eval_batch_size=64,
    metric_for_best_model="f1_macro",
    fp16_full_eval=True,
    optim="adamw_torch",
    report_to="none",
    label_names=["labels"],
)
trainer = SoftBCETrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator_fn,
    compute_metrics=compute_metrics_soft,
    label_smoothing_factor=None,
)
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

res = trainer.evaluate(tst_dataset)
print(res)
res = trainer.predict(tst_dataset)
logits = res.predictions
labels = res.label_ids
logits = torch.FloatTensor(logits)
preds = (torch.sigmoid(logits) >= 0.5).float().numpy()
results_file = os.path.join(output_dir, "results.txt")
write_metrics(
    labels,
    preds,
    target_names=target_names,
    output_file=results_file,
    expand_neutral=None,
)

