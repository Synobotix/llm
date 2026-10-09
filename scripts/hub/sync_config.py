"""
Configuration pour la synchronisation avec Hugging Face.
"""

HF_REPO_ID = "Mamy23/llm-dataset-gpt_classic_bitnet"
HF_REPO_TYPE = "dataset"


SYNC_FOLDERS = [

    ("data/processed", "data/processed"),

    ("data/tokenized", "data/tokenized"),

    ("tokenizer", "tokenizer"),

    ("checkpoints/bitnet", "checkpoints/bitnet"),

    ("documentation/result/bitnet", "documentation/result/bitnet"),
]
