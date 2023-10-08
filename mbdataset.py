"""Multilabel MultiTask Binary Dataset."""

import logging
from collections import Counter
from typing import List, Dict, Union, Optional

import torch
import numpy as np
from numpy.typing import NDArray
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast  # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer
from processors import (
    MultiLabelTSVProcessor,
    SoftMultiLabelTSVProcessor,
    MBExample,
    SoftExample,
)

logger = logging.getLogger(__name__)


class MLDatasetWithFloats(Dataset):
    """Multilabel Dataset."""

    id2label: Dict[int, str]
    label2id: Dict[str, int]
    num_labels: int
    mlb: MultiLabelBinarizer
    label_names: List[str]

    def __init__(
        self,
        processor: MultiLabelTSVProcessor,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_seq_length: int = 0,
        le: Optional[MultiLabelBinarizer] = None,
    ):
        """Initialize the dataset.

        Args:
            processors: List[MultiLabelTSVProcessor]. A list of processors
            tokenizer: PreTrainedTokenizer. The tokenizer to use.
            max_seq_length: int. The maximum length of the sequence.
            mask_labels: bool. Whether to use label masks.
        """
        self.max_seq_length = max_seq_length
        if max_seq_length <= 0:
            self.max_seq_length = tokenizer.max_len_single_sentence
        logger.info(f"Using max length={self.max_seq_length}")
        self.tokenizer = tokenizer

        examples = processor.get_examples()
        logger.info(f"Total Read: {len(examples)}")

        # go through examples and create statistics for sequence length
        """
        token_lengths = []
        for ex in examples:
            try:
                tokens = self.tokenizer(ex.text)["input_ids"]
                token_lengths.append(len(tokens))
            except:
                logger.error(f"Error with example: {ex}")
                continue
        
        # calculate average, max, and percentiles
        quartiles = np.percentile(token_lengths, [50, 97, 99])
        max_len = max(token_lengths)
        print("Median: %.3f" % quartiles[0])
        print("97: %.3f" % quartiles[1])
        print("99: %.3f" % quartiles[2])
        print("Max: %.3f" % max_len)
        """

        self._index_labels(examples, le)
        self.examples = examples
        self.features = self._featurize(examples)

    def _index_labels(
        self,
        examples: List[MBExample],
        le: Optional[MultiLabelBinarizer] = None,
    ):
        # create label index
        label_counts = Counter()
        for ex in examples:
            label_counts.update(ex.labels)
        label_set = [lbl for lbl, _ in label_counts.most_common()]

        # MultiLabelBinarizer
        if le:
            self.mlb = le
        else:
            self.mlb = MultiLabelBinarizer()
            self.mlb.fit([label_set])
        self.label_names = self.mlb.classes_.tolist()  # type: ignore
        logger.info(f"Classes: {self.label_names}")
        logger.info(f"Counts: {label_counts.most_common()}")

        # create samples_per_class
        self.samples_per_class = []
        for c in self.label_names:
            self.samples_per_class.append(label_counts[c])

        # id2label and label2id, num_classes
        self.id2label = {ii: lbl for ii, lbl in enumerate(self.label_names)}
        self.label2id = {lbl: ii for ii, lbl in enumerate(self.label_names)}
        self.num_labels = len(self.label_names)

    def _featurize_one(self, ex: MBExample):
        # tokenize
        features = self.tokenizer(
            text=ex.text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_length,
        )

        # create labels tensor
        labels = torch.Tensor(self.mlb.transform([ex.labels]))
        features["labels"] = labels

        return features

    def _featurize(self, exs: List[MBExample]):
        # tokenize
        features = self.tokenizer(
            text=[ex.text for ex in exs],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_length,
        )

        # create labels tensor
        labels = torch.Tensor(self.mlb.transform([ex.labels for ex in exs]))
        features["labels"] = labels

        return features

    def __len__(self):
        """Length of dataset corresponds to the number of examples."""
        return len(self.examples)

    def __getitem__(self, i):
        """Return the i-th example's features."""
        item = {k: self.features[k][i] for k in self.features.keys()}  # type: ignore

        return item

    def get_label_list(self) -> Union[List[List[str]], List[str]]:
        """Return the labels for the dataset."""
        labels = self.examples["labels"].numpy()  # type: ignore
        tuples = self.mlb.inverse_transform(labels)
        return [list(t) for t in tuples]
        # return [ex.labels for ex in self.examples]

    def get_target_names(self) -> List[str]:
        """Return the labels for the dataset."""
        return self.label_names

    def get_label_ids(self):
        """Return the labels for the dataset."""
        labels = self.features["labels"].numpy()  # type: ignore
        return labels

    def get_targets(self):
        """Return the labels for the dataset."""
        labels = self.features["labels"].numpy()  # type: ignore
        return labels

    def get_label_encoder(self) -> MultiLabelBinarizer:
        """Return the label encoder for the dataset."""
        return self.mlb

    @staticmethod
    def create_label_encoder_from_id2label(id2label: Dict[int, str]):
        """Create a label encoder from id2label."""
        # create label index
        label_set = [id2label[i] for i in range(len(id2label))]

        mlb = MultiLabelBinarizer()
        mlb.fit([label_set])
        return mlb


class MultiSoftTrainDataset(Dataset):
    """Multilabel Soft Dataset."""

    id2label: Dict[int, str]
    label2id: Dict[str, int]
    num_labels: int
    label_names: List[str]
    mlb: MultiLabelBinarizer

    def __init__(
        self,
        processor: Union[
            SoftMultiLabelTSVProcessor, List[SoftMultiLabelTSVProcessor],
        ],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_seq_length: int = 0,
        mlb: Optional[MultiLabelBinarizer] = None,
        make_hard: bool = False,
    ):
        """Initialize the dataset.

        Args:
            processors: List[MultiLabelTSVProcessor]. A list of processors
            tokenizer: PreTrainedTokenizer. The tokenizer to use.
            max_seq_length: int. The maximum length of the sequence.
            mask_labels: bool. Whether to use label masks.
        """
        self.max_seq_length = max_seq_length
        if max_seq_length <= 0:
            self.max_seq_length = tokenizer.max_len_single_sentence
        logger.info(f"Using max length={self.max_seq_length}")
        self.tokenizer = tokenizer
        self.make_hard = make_hard

        examples = []
        logger.info(f"Reading examples from processor")
        if isinstance(processor, list):
            for proc in processor:
                data_p = proc.get_examples()
                examples += data_p
                logger.info(f"Total Read: {len(examples)}")
            proc_labels = processor[0].get_labels()
        else:
            data_p = processor.get_examples()
            examples += data_p
            proc_labels = processor.get_labels()
        # logger.info(f"Read {len(data_p)} from processor={ii}")
        logger.info(f"Total Read: {len(examples)}")

        self._index_labels(proc_labels, mlb)
        self.examples = examples

        logger.info(f"tokenizing examples")
        self.features = self._featurize(examples)
        self.n = len(self.examples)

    def remove_examples(self, indices: List[int]):
        """Remove examples by indices."""
        self.examples = [
            ex for i, ex in enumerate(self.examples) if i not in indices
        ]
        self.features = self._featurize(self.examples)
        self.n = len(self.examples)

    def assign_scores(self, scores: NDArray, ignore_neutral: bool = True):
        """Assign scores to examples."""
        for i, ex in enumerate(self.examples):
            ex_scores = {
                self.id2label[j]: scores[i][j] for j in range(self.num_labels)
            }
            if ignore_neutral and sum(ex.scores.values()) == 0.0:
                continue
            ex.scores = ex_scores
            self.examples[i] = ex
        # update features
        self.features = self._featurize(self.examples)

    def save(self, path: str):
        """Save the dataset."""
        logger.info(f"Saving dataset to {path}")
        examples = []
        for ex in self.examples:
            row = {
                "tid": ex.guid,
                "text": ex.text,
            }
            if ex.split is not None:
                row["split"] = ex.split
            for label in ex.scores:
                row[label] = ex.scores[label]
            examples.append(row)
        import pandas as pd

        df = pd.DataFrame(examples)
        df.set_index("tid", inplace=True)
        df.to_csv(path, sep="\t")

    def calculate_token_lengths(self):
        """Calculate token lengths."""
        token_lengths = []
        for ex in self.examples:
            tokens = self.tokenizer(ex.text)["input_ids"]  # type: ignore
            token_lengths.append(len(tokens))

        quartiles = np.percentile(token_lengths, [50, 97, 99])
        max_len = max(token_lengths)
        print("Median: %.3f" % quartiles[0])
        print("97: %.3f" % quartiles[1])
        print("99: %.3f" % quartiles[2])
        print("Max: %.3f" % max_len)

    def _index_labels(
        self,
        processor_labels: Optional[List[str]] = None,
        mlb: Optional[MultiLabelBinarizer] = None,
    ):
        if processor_labels is None and mlb is None:
            raise ValueError("Must provide either processor_labels or mlb")

        # MultiLabelBinarizer
        if mlb is not None:
            logger.info("Using provided MultiLabelBinarizer")
            self.mlb = mlb
        else:
            logger.info("Creating MultiLabelBinarizer")
            self.mlb = MultiLabelBinarizer()
            self.mlb.fit([processor_labels])
            logger.info(f"Classes: {self.mlb.classes_}")

        # label names and id2label and label2id
        self.label_names = self.mlb.classes_.tolist()  # type: ignore

        # id2label and label2id, num_classes
        self.id2label = {ii: lbl for ii, lbl in enumerate(self.label_names)}
        logger.info(f"id2label: {self.id2label}")
        self.label2id = {lbl: ii for ii, lbl in enumerate(self.label_names)}
        self.num_labels = len(self.label_names)

    def _featurize_one(self, ex: SoftExample):
        """Convert a single example to features."""
        # tokenize
        features = self.tokenizer(
            text=ex.text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_length,
        )

        # create labels tensor
        labels = torch.zeros(self.num_labels, dtype=torch.float)
        for lbl, score in ex.scores.items():
            our_label_id = self.label2id[lbl]
            if self.make_hard:
                score = 1.0 if score >= 0.5 else 0.0
            labels[our_label_id] = score
        features["labels"] = labels.unsqueeze(0)

        return features

    def _featurize(self, exs: List[SoftExample]):
        """Convert a list of examples to features."""
        # tokenize
        features = self.tokenizer(
            text=[ex.text for ex in exs],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_length,
        )

        # create labels tensor
        all_labels = []
        for ex in exs:
            labels = torch.zeros(self.num_labels, dtype=torch.float)
            for lbl, score in ex.scores.items():
                our_label_id = self.label2id[lbl]
                if self.make_hard:
                    score = 1.0 if score >= 0.5 else 0.0
                labels[our_label_id] = score
            all_labels.append(labels)
        features["labels"] = torch.vstack(all_labels)

        return features

    def __len__(self):
        """Length of dataset corresponds to the number of examples."""
        return self.n

    def __getitem__(self, i):
        """Return the i-th example's features."""
        item = {k: self.features[k][i] for k in self.features.keys()}  # type: ignore
        return item

    def get_target_names(self):
        return self.label_names

    def get_label_list(self) -> Union[List[List[str]], List[str]]:
        """Return the labels for the dataset."""
        labels = self.get_targets()
        tuples = self.mlb.inverse_transform(labels)
        return [list(t) for t in tuples]

    def get_target_ids(self) -> List[List[int]]:
        """Return the labels for the dataset as lists of ints."""
        labels = self.get_targets()
        tuples = self.mlb.inverse_transform(labels)
        label_strings = [list(t) for t in tuples]
        label_ints = [
            [self.label2id[lbl] for lbl in lbls] for lbls in label_strings
        ]
        return label_ints

    def get_targets(self):
        """Get targets in multilabel format."""
        labels = self.get_soft_targets()
        labels = (labels >= 0.5).astype(float)
        return labels

    def get_soft_targets(self):
        """Return the labels for the dataset."""
        labels = self.features["labels"].numpy()  # type: ignore
        return labels

    def print_features(self, item):
        # text
        s = self.tokenizer.decode(item["input_ids"])
        print(f"{s}")

        # soft labels
        soft_labels = torch.unsqueeze(item["labels"], 0).numpy()  # type: ignore
        labels = (soft_labels >= 0.5).astype(float)
        labels = self.mlb.inverse_transform(labels)
        print(f"labels: {labels}")

        scores = dict(zip(self.mlb.classes_.tolist(), soft_labels[0].tolist()))
        print(f"{scores}")
        print("------------------------")

    def print_example(self, i):
        """Print the i-th example."""
        print(f"----- Example {i} -----")
        item = self.__getitem__(i)
        print(self.examples[i])
        self.print_features(item)

    def print_random_examples(self, n: int = 5):
        """Print the random n examples."""
        idx = torch.randperm(self.n)[:n].tolist()
        for i in idx:
            self.print_example(i)

    def print_featurize(self):
        idx = torch.randperm(self.n)[0]
        print(self.examples[idx])
        features = self._featurize_one(self.examples[idx])
        item = {k: features[k][0] for k in features.keys()}  # type: ignore
        self.print_features(item)

    def get_label_encoder(self) -> MultiLabelBinarizer:
        """Return the label encoder."""
        return self.mlb


class MLDataset(MLDatasetWithFloats):
    """Multilabel Dataset."""

    id2label: Dict[int, str]
    label2id: Dict[str, int]
    num_labels: int
    mlb: MultiLabelBinarizer
    label_names: List[str]

    
    def _featurize_one(self, ex: MBExample):
        raise NotImplementedError("Not implemented")

    def _featurize(self, exs: List[MBExample]):
        # tokenize
        features = self.tokenizer(
            text=[ex.text for ex in exs],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_length,
        )

        # create labels tensor
        labels = torch.FloatTensor(self.mlb.transform([ex.labels for ex in exs]))
        features["labels"] = labels

        return features

    def __len__(self):
        """Length of dataset corresponds to the number of examples."""
        return len(self.examples)

    def __getitem__(self, i):
        """Return the i-th example's features."""
        item = {k: self.features[k][i] for k in self.features.keys()}  # type: ignore

        return item

    def get_label_list(self) -> Union[List[List[str]], List[str]]:
        raise NotImplementedError("Not implemented")
