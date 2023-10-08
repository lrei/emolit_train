"""Processors read data from files."""
import json
import logging
import dataclasses
from dataclasses import dataclass
from collections import Counter
from typing import List, Optional, Dict

import pandas as pd
from transformers import DataProcessor  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class MBExample:
    """A single training/test example for multilabel sequence classification.

    Args:
        guid: Unique id for the example.
        text: string. The untokenized text of the sequence.
        labels: (Optional) List[string]. The labels of the example.
        label_list: (Optional) List[string]. The list of possible labels.
    """

    guid: str
    text: str
    labels: Optional[List[str]] = None
    label_list: Optional[List[str]] = None
    split: Optional[str] = None

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self))


@dataclass
class SoftExample:
    """A single training/test example for multilabel sequence classification.

    Args:
        guid: Unique id for the example.
        text: string. The untokenized text of the sequence.
        labels: (Optional) List[float]. The labels of the example.
        label_list: (Optional) List[string]. The corresponding list of labels.
    """

    guid: str
    text: str
    scores: Dict[str, float]
    split: Optional[str] = None

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self))


class MultiLabelTSVProcessor(DataProcessor):
    """Processor for Multilabel classification datasets stored as a TSV file.

    Args:
        data_file: string. The TSV file containing data.
        text_col: The column containing the text.
        ignore_cols: List[string]. A list of columns to ignore.
        index_col: string. The column of the file containing the id strings.
            If None, uses the row as index.
        random_state: int. The random state to use for the split.
    """

    label_list: List[str]

    def __init__(
        self,
        data_file,
        text_col: str = "text",
        lang: Optional[str] = None,
        lang_col: str = "lang",
        split: Optional[str] = None,
        split_col: Optional[str] = None,
        random_state: int = 1984,
    ):
        """See class."""
        super(MultiLabelTSVProcessor, self).__init__()
        self.data_file = data_file
        self.text_col = text_col
        self.random_state = random_state

        self.lang = lang
        self.lang_col = lang_col

        self.split_col = split_col
        self.split = split

        self.not_labels = [self.text_col, self.lang_col, self.split_col]
        self.not_labels = [c for c in self.not_labels if c is not None]
        self.not_labels += ["tid", "id", "split", "og_split", "text", "lang"]

    def _df_has_neutral(self, df):
        """Check that the dataframe has a neutral examples."""
        label_list = self._get_labels_in_data(df)
        df = df.astype({c: int for c in label_list})
        df = df[label_list].sum(axis=1)
        if any(df == 0):
            print(f"Found {sum(df==0)} neutral examples")
            return True
        return False

    def _get_labels_in_data(self, df):
        """Get the labels present in the dataframe."""
        label_list = df.columns.tolist()
        label_list = [
            lbl
            for lbl in label_list
            if lbl not in self.not_labels
        ]
        return label_list

    def _read_tsv(self) -> List[MBExample]:
        """Read data file."""
        logger.info(
            f"Reading from file={self.data_file}"
        )
        counter = Counter()
        df = pd.read_csv(
            self.data_file,
            sep="\t",  # type: ignore
        )  # type: ignore
        cols = df.columns.tolist()  # type: ignore
        cols = [c for c in cols if c not in self.not_labels]

        # filter by language
        if self.lang is not None:
            logger.info(f"Filtering by language: {self.lang}")
            df = df[df[self.lang_col] == self.lang]


        ids = df.index.tolist()  # type: ignore

        # get the labels present in the dataset file
        label_list = self._get_labels_in_data(df)
        if self._df_has_neutral(df):
            self.neutral = True
        self.label_list = label_list
        # logger.info(f"{self.__class__.__name__} Label list: {label_list}")

        # switch the data format to a list of dictionaries
        df = df.to_dict("records")  # type: ignore

        # convert to a list of MultilabelExample
        examples = []
        for guid, row in zip(ids, df):
            if guid is None:
                raise ValueError("No found")
            guid = str(guid)
            keys = [k for k in row.keys() if k not in self.not_labels]
            labels: List[str] = [str(k) for k in keys if row[k] == 1]

            # count labels
            count_labels = [k for k in labels]
            # in case no label is present, this is a neutral example
            if not count_labels:
                count_labels = ["neutral"]
            counter.update(count_labels)
            counter["n_examples"] += 1

            split = row[self.split_col] if self.split_col else None

            # mask
            labels_in_example = [k for k in label_list if row[k] >= 0]

            # create the MBExample
            ex = MBExample(
                guid=guid,
                text=row[self.text_col],
                labels=labels,
                label_list=labels_in_example,
                split=split,
            )
            # add it to the list
            examples.append(ex)

        logger.info(f"Read: {len(examples)} examples")
        for k in sorted(counter.keys()):
            logger.info(f"\t{k}:\t{counter[k]}")
        return examples

    def get_examples(self) -> List[MBExample]:
        """See base class."""
        return self._read_tsv()

    def get_labels(self) -> List[str]:
        """Return the list of labels."""
        return self.label_list


class SoftMultiLabelTSVProcessor(DataProcessor):
    """Processor for Multilabel classification datasets stored as a TSV file.

    Labels are expected to be soft labels (i.e. probabilities)
    Args:
        data_file: string. The TSV file containing data.
        text_col: The column containing the text.
        use_cols: List[string]. A list of columns to use.
        index_col: string. The column of the file containing the id strings.
            If None, uses the row as index.
    """

    labels: List[str]

    def __init__(
        self,
        data_file,
        text_col: str = "text",
        index_col: str = "tid",
        split_col: Optional[str] = None,
        use_cols: Optional[List[str]] = None,
    ):
        """See class."""
        super(SoftMultiLabelTSVProcessor, self).__init__()
        self.data_file = data_file
        self.text_col = text_col
        self.split_col = split_col
        self.use_cols = use_cols
        self.index_col = index_col

        if self.use_cols and split_col:
            self.use_cols = [
                self.index_col,
                self.text_col,
                split_col,
            ] + self.use_cols
        elif self.use_cols and not split_col:
            self.use_cols = [self.index_col, self.text_col] + self.use_cols

    def get_labels_in_df(self, df):
        """Get the labels present in the dataframe."""
        label_list = df.columns.tolist()
        label_list = [
            lbl
            for lbl in label_list
            if lbl != self.text_col
            and lbl != self.index_col
            and lbl != self.split_col
        ]
        self.labels = sorted(label_list)
        return self.labels

    def _read_tsv(self) -> List[SoftExample]:
        """Read data file."""
        logger.info(
            f"Reading from file={self.data_file} index_col={self.index_col} use_cols={self.use_cols}"
        )
        counter = Counter()
        df = pd.read_csv(
            self.data_file,
            sep="\t",  # type: ignore
            index_col=self.index_col,
            usecols=self.use_cols,
        )  # type: ignore

        ids = df.index.tolist()  # type: ignore

        # get the labels present in the dataset file
        label_list = self.get_labels_in_df(df)
        logger.info(f"Processor: Labels in File: {label_list}")

        # switch the data format to a list of dictionaries
        df = df.to_dict("records")  # type: ignore

        # convert to a list of MultilabelExample
        examples = []
        for guid, row in zip(ids, df):
            if guid is None:
                raise ValueError("Missing id in row")
            guid = str(guid)
            scores = {k: row[k] for k in label_list}
            # create the MBExample
            split = row[self.split_col] if self.split_col else None
            ex = SoftExample(
                guid=guid, text=row[self.text_col], scores=scores, split=split,
            )
            # add it to the list
            examples.append(ex)

        logger.info(f"Read: {counter}")
        return examples

    def get_examples(self) -> List[SoftExample]:
        """See base class."""
        return self._read_tsv()

    def get_labels(self) -> List[str]:
        """Return the list of labels."""
        return self.labels

