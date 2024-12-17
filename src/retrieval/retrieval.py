from typing import Callable, List, NoReturn, Tuple, Optional

class Retrieval:
    def __init__(
        self,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        tokenize_fn=None,
        args=None,
     ):
        pass

    def retrieve(self, query_or_dataset, topk: Optional[int] = 10 , alpha: Optional[float] = 0.7):
        pass