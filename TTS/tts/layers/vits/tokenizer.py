import json
import torch
from tokenizers import Tokenizer


class VoiceBpeTokenizer:
    def __init__(self, vocab_file=None, preprocess=None):
        self.tokenizer = None

        if vocab_file is not None:
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = json.load(f)

            self.language = vocab["model"]["language"] if "language" in vocab["model"] else None

            if preprocess is None:
                self.preprocess = "pre_tokenizer" in vocab and vocab["pre_tokenizer"]
            else:
                self.preprocess = preprocess

            self.tokenizer = Tokenizer.from_file(vocab_file)

    def encode(self, txt):
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")
        txt = txt.replace("[UNK]", "")
        return txt

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self):
        return max(self.tokenizer.get_vocab().values()) + 1
