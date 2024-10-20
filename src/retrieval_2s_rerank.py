import json
import os
import pickle
import time
import torch
import logging
import scipy
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union, NoReturn
from tqdm.auto import tqdm

import argparse
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from torch.nn.functional import normalize

from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi, BM25Plus
from transformers import AutoTokenizer, AutoModel
from utils import set_seed

set_seed(42)
logger = logging.getLogger(__name__)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logging.info(f"[{name}] done in {time.time() - t0:.3f} s")


class TwoStageReranker:
    def __init__(
        self,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        logging.info(f"Lengths of contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.tokenize_fn = AutoTokenizer.from_pretrained(
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        )
        self.sparse_embeder = None
        self.sparse_embeder = TfidfVectorizer(
            tokenizer=self.tokenize_fn, ngram_range=(1, 2), max_features=50000,
        )
        self.dense_embeder = AutoModel.from_pretrained(
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        )
        self.sparse_embeds = None
        self.dense_embeds = None