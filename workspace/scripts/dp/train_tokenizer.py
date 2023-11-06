import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

is_train = True

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
START_TOKEN = "[START]"
STOP_TOKEN = "[STOP]"
SPACE_TOKEN = "[SPACE]"

train_files = [
    "data/tts/token/v1/ar/train.txt",
    "data/tts/token/v1/en/train.txt",
    "data/tts/token/v1/hi/train.txt"
]

tokenizer_path = "data/tts/token/v1/model"
os.makedirs(tokenizer_path, exist_ok=True)

if is_train:
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN, fuse_unk=False))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()

    trainer = trainers.BpeTrainer(
        vocab_size=4096,
        special_tokens=[PAD_TOKEN, START_TOKEN, STOP_TOKEN, SPACE_TOKEN, UNK_TOKEN],
        max_token_length=3,
    )

    tokenizer.train(files=train_files, trainer=trainer)
    tokenizer.save(os.path.join(tokenizer_path, "bpe_tokenizer.json"))

print("\nStarting testing the model...")

tokenizer = Tokenizer.from_file(os.path.join(tokenizer_path, "bpe_tokenizer.json"))
print(f"Vocab size: {tokenizer.get_vocab_size()}")

for fl in train_files:
    with open(fl, encoding="utf-8") as st:
        for index, line in enumerate(st):
            if index >= 2:
                break
            text = str(line).strip("\n").strip()
            in_text = text.replace(" ", "[SPACE]")
            encoded = tokenizer.encode(in_text)
            print(f"[{text}] | [{encoded.ids}] | [{encoded.tokens}]")
        print("\n")
