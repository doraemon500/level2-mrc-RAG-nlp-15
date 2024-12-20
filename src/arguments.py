from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    # CurtisJeon/klue-roberta-large-korquad_v1_qa
    # uomnf97/klue-roberta-finetuned-korquad-v2
    model_name_or_path: str = field(
        default="uomnf97/klue-roberta-finetuned-korquad-v2",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    config_name_dpr: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    dense_model_name: Optional[str] = field(
        default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        metadata={
            "help": "Dense model for dense embedding"
        },
    )
    #################################################################################
    batch_size: int = field(
        default=16
    )
    
    num_epochs: int = field(
        default=3
    )

    #################################################################################


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={
            "help": "The name of the dataset to use."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached training and evaluation sets"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=64,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={
        "help": "Whether to run passage retrieval using sparse embedding."
        },
    )
    num_clusters: int = field(
        default=64, metadata={
            "help": "Define how many clusters to use for faiss."
        },
    )
    use_faiss: bool = field(
        default=True, metadata={
            "help": "Whether to build with faiss"
        },
    )
    dense_encoder_type: str = field(
        default = 'hybrid', metadata = {
            "help": "Whether to run passage retrieval using dense embedding."
        },
    )
    remove_char: bool = field(
        default=True, metadata={
            "help": "Whether to remove special character before embedding"
        },
    )
    data_path: str = field(
        default="../data/",
        metadata={
            "help": "The path of the data directory"
        },
    )
    context_path: str = field(
        default="wikipedia_documents.json",
        metadata={
            "help": "The name of the context file"
        },
    )
    alpha_retrieval: Optional[float] = field(
        default=0.7,
        metadata={
            "help": "Value for hybridizing embedding scores in HybridSearch"
        }
    )
    top_k_retrieval: int = field(
        default=20,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )