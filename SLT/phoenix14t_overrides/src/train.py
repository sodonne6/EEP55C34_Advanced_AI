import gzip
import pickle

import torch
import yaml
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from src.configuration import SLTConfig
from src.model import SLTModel


def load_data(sgn_path):
    f = gzip.open(sgn_path, "rb")
    folders = pickle.load(f)
    return folders


class SignLanguageDataset(Dataset):
    def __init__(self, pickle_path):
        data = load_data(pickle_path)
        self.signs = [torch.tensor(item["sign"], dtype=torch.float32) for item in data]
        self.texts = [item["text"] for item in data]

    def __len__(self):
        return len(self.signs)

    def __getitem__(self, idx):
        return {
            "sign": self.signs[idx],  # [seq_len, feature_dim]
            "text": self.texts[idx],  # str
        }


class SLTDataCollator:
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        # 1) Signs padding
        sign_features = [item["sign"] for item in batch]
        padded_signs = torch.nn.utils.rnn.pad_sequence(sign_features, batch_first=True)

        batch_size = len(sign_features)
        max_sign_len = padded_signs.size(1)
        attention_mask = torch.zeros(batch_size, max_sign_len, dtype=torch.long)
        for i, sign in enumerate(sign_features):
            attention_mask[i, : sign.size(0)] = 1

        # 2) Text -> labels WITHOUT CLS/SEP; append EOS manually
        texts = [item["text"] for item in batch]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_len - 1,   # leave room for EOS
            add_special_tokens=False,      # IMPORTANT
        )
        ids = enc["input_ids"]           # [B, L]
        pad = self.tokenizer.pad_token_id
        eos = self.tokenizer.sep_token_id  # treat SEP as EOS

        # Put EOS at first pad position (or overwrite last token if already full)
        lengths = (ids != pad).sum(dim=1)  # [B]
        labels = ids.clone()
        for i, L in enumerate(lengths.tolist()):
            if L < labels.size(1):
                labels[i, L] = eos
            else:
                labels[i, -1] = eos

        # Ignore padding for loss
        labels[labels == pad] = -100

        return {
            "input_values": padded_signs,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    # load config file
    with open("configs/config.yaml") as f:
        try:
            config_file = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise exc

    # 1. Setup Data
    dataset = SignLanguageDataset(config_file["dataset"]["pickle_path"]["train"])
    eval_dataset = SignLanguageDataset(config_file["dataset"]["pickle_path"]["dev"])

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config_file["model"]["pretrained_model_name"]
    )

    # 3. Model Configuration
    config = SLTConfig(
        input_dim=config_file["model"].get("input_dim", 832),
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = SLTModel(config)

    # 4. Trainer Setup
    training_args = TrainingArguments(**config_file["training_args"])

    collator = SLTDataCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()

    # Save model and tokenizer
    model.save_pretrained("./slt_final_model")
    tokenizer.save_pretrained("./slt_final_model")


if __name__ == "__main__":
    main()
