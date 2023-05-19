"""Utility functions."""
import json
import torch
from typing import List, Optional

import numpy as np

from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    accuracy_score,
)
from tqdm import tqdm


from mbdataset import TestDatasetForMappedOutputEval


def compute_metrics(eval_pred):
    """Compute metrics."""
    logits, labels = eval_pred

    logits = torch.FloatTensor(logits)
    preds = (torch.sigmoid(logits) >= 0.5).float().numpy()

    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    p_macro = precision_score(labels, preds, average="macro")
    p_micro = precision_score(labels, preds, average="micro")
    r_macro = recall_score(labels, preds, average="macro")
    r_micro = recall_score(labels, preds, average="micro")
    metrics = {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "p_macro": p_macro,
        "p_micro": p_micro,
        "r_macro": r_macro,
        "r_micro": r_micro,
    }

    return metrics


def compute_metrics_binary(eval_pred, pos_id=0):
    """Compute metrics."""
    logits, labels = eval_pred

    preds = logits.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary", pos_label=pos_id)
    p = precision_score(labels, preds, average="binary", pos_label=pos_id)
    r = recall_score(labels, preds, average="binary", pos_label=pos_id)
    metrics = {
        "acc": acc,
        "f1": f1,
        "p": p,
        "r": r,
    }

    return metrics


def compute_metrics_soft(eval_pred):
    """Compute metrics."""
    logits, labels = eval_pred

    # convert labels
    labels = torch.FloatTensor(labels)
    labels = (labels >= 0.5).float().numpy()

    # convert logits
    logits = torch.FloatTensor(logits)
    preds = (torch.sigmoid(logits) >= 0.5).float().numpy()

    # calculate metrics
    mse = mean_squared_error(labels, logits)
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    p_macro = precision_score(labels, preds, average="macro")
    p_micro = precision_score(labels, preds, average="micro")
    r_macro = recall_score(labels, preds, average="macro")
    r_micro = recall_score(labels, preds, average="micro")
    metrics = {
        "mse": mse,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "p_macro": p_macro,
        "p_micro": p_micro,
        "r_macro": r_macro,
        "r_micro": r_micro,
    }

    return metrics


def compute_metrics_with_logits_multiclass(eval_pred):
    """Compute metrics."""
    logits, labels = eval_pred

    preds = logits.argmax(axis=1)

    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    p_macro = precision_score(labels, preds, average="macro")
    p_micro = precision_score(labels, preds, average="micro")
    r_macro = recall_score(labels, preds, average="macro")
    r_micro = recall_score(labels, preds, average="micro")
    metrics = {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "p_macro": p_macro,
        "p_micro": p_micro,
        "r_macro": r_macro,
        "r_micro": r_micro,
    }

    return metrics


def compute_metrics_debug(eval_pred, mlb):
    logits, labels = eval_pred

    logits = torch.FloatTensor(logits)
    preds = (torch.sigmoid(logits) >= 0.5).float().numpy()

    sample_labels = labels[0:4]
    sample_labels = mlb.inverse_transform(sample_labels)
    sample_preds = preds[0:4]
    sample_preds = mlb.inverse_transform(sample_preds)
    print(f"Sample labels: {sample_labels}")
    print(f"Sample preds: {sample_preds}")

    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    p_macro = precision_score(labels, preds, average="macro")
    p_micro = precision_score(labels, preds, average="micro")
    r_macro = recall_score(labels, preds, average="macro")
    r_micro = recall_score(labels, preds, average="micro")
    metrics = {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "p_macro": p_macro,
        "p_micro": p_micro,
        "r_macro": r_macro,
        "r_micro": r_micro,
    }

    return metrics


def write_metrics(
    y_true,
    y_pred,
    output_file,
    target_names: Optional[List[str]] = None,
    expand_neutral: Optional[str] = None,
):
    """Compute and write final metrics.

    Expects softmax or sigmoid outputs.
    """
    if expand_neutral:
        if target_names:
            target_names = target_names + [expand_neutral]
        # expand y_true and y_pred to have a neutral class
        mask = ~np.any(y_true >= 0.5, axis=1)
        y_true = np.c_[y_true, np.zeros(y_true.shape[0])]
        y_true[mask, -1] = 1

        mask = ~np.any(y_pred >= 0.5, axis=1)
        y_pred = np.c_[y_pred, np.zeros(y_pred.shape[0])]
        y_pred[mask, -1] = 1

    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    p_macro = precision_score(y_true, y_pred, average="macro")
    p_micro = precision_score(y_true, y_pred, average="micro")
    r_macro = recall_score(y_true, y_pred, average="macro")
    r_micro = recall_score(y_true, y_pred, average="micro")

    metrics = {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "p_macro": p_macro,
        "p_micro": p_micro,
        "r_macro": r_macro,
        "r_micro": r_micro,
    }

    report = classification_report(
        y_true, y_pred, target_names=target_names, digits=4,
    )
    print(report)
    print(f"Results recorded in {output_file}")
    with open(output_file, "w") as fout:
        print(json.dumps(metrics), file=fout)
        print(report, file=fout)
    return metrics


def write_run_metrics(
    y_true,
    y_pred,
    output_file,
    target_names: Optional[List[str]] = None,
    expand_neutral: Optional[str] = None,
):
    """Compute and write final metrics.

    Expects softmax or sigmoid outputs.
    """
    if expand_neutral:
        if target_names:
            target_names = target_names + [expand_neutral]
        # expand y_true and y_pred to have a neutral class
        mask = ~np.any(y_true >= 0.5, axis=1)
        y_true = np.c_[y_true, np.zeros(y_true.shape[0])]
        y_true[mask, -1] = 1

        mask = ~np.any(y_pred >= 0.5, axis=1)
        y_pred = np.c_[y_pred, np.zeros(y_pred.shape[0])]
        y_pred[mask, -1] = 1

    f1_macro = f1_score(y_true, y_pred, average="macro")

    metrics = {
        "f1_macro": f1_macro,
    }

    print(f"Results recorded in {output_file}")
    with open(output_file, "a") as fout:
        print(json.dumps(metrics), file=fout)
    return metrics


def perform_mapped_inference(
    model, dataset: TestDatasetForMappedOutputEval, batch_size: int = 128,
):
    """Perform inference on a TestDatasetForMappedOutputEval with a given mapping.

    Args:
        model (torch.nn.Module): The model to use for inference.
        dataset (torch.utils.data.Dataset): The dataset to perform inference on.
        mapping (dict): The mapping from the original labels to the mapped labels.
        multilabel (bool, optional): Whether the dataset is multilabel or not.
        batch_size (int, optional): The batch size to use for inference.

    Mapping Format
    --------------
    mapped_label_id -> [original_label_id_1, original_label_id_2, ...]

    """
    # Set the model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    # collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    # get test properties
    mapping = dataset.get_mapping()
    multilabel = dataset.is_multilabel()

    # Create an empty list to store the predictions
    all_logits = []

    # Loop through the dataset in batches
    for i in tqdm(range(0, len(dataset), batch_size), dynamic_ncols=True):
        inputs = dataset[i : i + batch_size]
        # don't pass labels
        inputs.pop("labels", None)
        # send to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Pass the inputs through the model
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        # Extract the predicted labels from the output
        logits = outputs.logits.detach().float().cpu()
        all_logits.append(logits)
    all_logits = torch.vstack(all_logits)

    # map logits
    mapped_logits = torch.zeros((all_logits.shape[0], len(mapping)))
    for mapped_label_id, original_label_ids in mapping.items():
        mapped_logits[:, mapped_label_id] = torch.max(
            all_logits[:, original_label_ids], dim=1
        )[0]

    if multilabel:
        # Handle multilabel
        preds = (torch.sigmoid(mapped_logits) >= 0.5).float().numpy()
        return preds
    # Handle multiclass
    preds = torch.argmax(mapped_logits, dim=1).numpy()
    return preds


def compute_metrics_with_logits_multilabel(eval_pred):
    """Compute metrics."""
    logits, labels = eval_pred

    logits = torch.FloatTensor(logits)
    preds = (torch.sigmoid(logits) >= 0.5).float().numpy()

    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    p_macro = precision_score(labels, preds, average="macro")
    p_micro = precision_score(labels, preds, average="micro")
    r_macro = recall_score(labels, preds, average="macro")
    r_micro = recall_score(labels, preds, average="micro")
    metrics = {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "p_macro": p_macro,
        "p_micro": p_micro,
        "r_macro": r_macro,
        "r_micro": r_micro,
    }

    return metrics


def perform_inference(
    model, dataset, batch_size: int = 128, multilabel: bool = True,
):
    """Perform inference on a multilabel dataset.

    Args:
        model (torch.nn.Module): The model to use for inference.
        dataset (torch.utils.data.Dataset): The dataset to perform inference on.
        batch_size (int, optional): The batch size to use for inference.


    """
    # Set the model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    # collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create an empty list to store the predictions
    all_logits = []

    # Loop through the dataset in batches
    for i in tqdm(range(0, len(dataset), batch_size), dynamic_ncols=True):
        inputs = dataset[i : i + batch_size]
        # don't pass labels
        inputs.pop("labels", None)
        # send to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Pass the inputs through the model
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        # Extract the predicted labels from the output
        logits = outputs.logits.detach().float().cpu()
        all_logits.append(logits)
    all_logits = torch.vstack(all_logits)

    # Handle multilabel
    if multilabel:
        preds = (torch.sigmoid(all_logits) >= 0.5).float().numpy()
        return preds
    # Handle multiclass
    preds = torch.argmax(all_logits, dim=1).numpy()
    return preds


def write_metrics_simple(y_true, y_pred, output_file):
    """Compute and write final metrics.
    """

    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    metrics = {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }

    print(f"Results recorded in {output_file}")
    with open(output_file, "w") as fout:
        print(json.dumps(metrics), file=fout)
    return metrics
