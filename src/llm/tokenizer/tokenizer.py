from pathlib import Path

from tokenizers import Tokenizer

from llm.config.parameters import TOKENIZER_FILE


def load_tokenizer(tokenizer_file: Path | None = None) -> Tokenizer:
    """
    Charge le tokenizer depuis `tokenizer_file`
    (par défaut TOKENIZER_FILE).
    """
    tokenizer_file = tokenizer_file or TOKENIZER_FILE

    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer introuvable : {tokenizer_file}")

    return Tokenizer.from_file(str(tokenizer_file))


def get_vocab_size() -> int:
    return load_tokenizer().get_vocab_size()
