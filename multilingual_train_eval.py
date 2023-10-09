"""Train and eval the classifier on the multilingual dataset."""

import os
import logging
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from processors import MultiLabelTSVProcessor
from mbdataset import MLDatasetWithFloats
from mutils import compute_metrics

TRN_SRC = "./data/emolit_multilingual/emolit.tsv"
GOLD_SRC = "/data/emolit_multilingual/gold.tsv"
output_dir = "./model/multilingual"
MODEL_NAME = "xlm-roberta-base"
SEQLEN = 64
BS=32
NUM_EPOCHS = 10

# logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
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
    load_best_model_at_end=True,
    do_eval=True,
    per_device_eval_batch_size=64,
    metric_for_best_model="f1_macro",
    fp16_full_eval=True,
    optim="adamw_torch",
    report_to="none",
    label_names=["labels"],
)

# load tokenizer
tkz = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
collator_fn = DataCollatorWithPadding(
    tkz, padding=True, pad_to_multiple_of=8
)

# load train dataset
proc_trn = MultiLabelTSVProcessor(TRN_SRC)
ds_trn = MLDatasetWithFloats(proc_trn, tkz, max_seq_length=SEQLEN)

# load eval dataset
proc_gold = MultiLabelTSVProcessor(GOLD_SRC)
ds_gold = MLDatasetWithFloats(proc_gold, tkz, max_seq_length=SEQLEN, le=ds_trn.get_label_encoder())

logger.info(f"Train: {len(ds_trn)} examples")
logger.info(f"Gold: {len(ds_gold)} examples")

# Create the model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=ds_trn.num_labels,
    problem_type="multi_label_classification",
    label2id=ds_trn.label2id,
    id2label=ds_trn.id2label,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_trn,
    eval_dataset=ds_gold,
    data_collator=collator_fn,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model(output_dir)
tkz.save_pretrained(output_dir)

print("Evaluating on gold dataset")
res = trainer.evaluate(ds_gold)
print(res)

# get the list of languages by reading the TSV and getting the values
# for the column "lang"
df = pd.read_csv(GOLD_SRC, sep="\t", usecols=["lang"])
langs = df["lang"].unique().tolist()
print(f"Evaluating for {len(langs)} languages: {langs}")
# for each language create it's own processor and dataset,
# then use trainer to evaluate
for lang in langs:
    print(f"Evaluating for language {lang}")
    # create a processor for this language
    proc_lang = MultiLabelTSVProcessor(GOLD_SRC, lang=lang)
    # create a dataset for this language
    ds_lang = MLDatasetWithFloats(proc_lang, tkz, max_seq_length=SEQLEN, le=ds_trn.get_label_encoder())
    # evaluate
    res = trainer.evaluate(ds_lang)
    print(res)
    print()