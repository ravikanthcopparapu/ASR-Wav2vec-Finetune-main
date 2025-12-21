import sys
sys.path.append("../")
import torch

from utils.feature import load_wav
from typing import Dict

class DefaultCollate:
    def __init__(self, processor, sr) -> None:
        self.processor = processor
        self.sr = sr

    def __call__(self, inputs) -> Dict[str, torch.tensor]:
        # unzip features and transcripts from the batch list
        features, transcripts = zip(*inputs)
        features, transcripts = list(features), list(transcripts)

        # 1) Process audio with the feature extractor
        batch = self.processor(
            features,
            sampling_rate=16000,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )

        # 2) Process text labels with the tokenizer directly
        #    (avoid as_target_processor + padding clash)
        labels_batch = self.processor.tokenizer(
            transcripts,
            padding=True,          # or "longest"
            return_tensors="pt",
        )

        # Mask out padding in labels with -100 so it’s ignored in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels

        return batch


class Dataset:
    def __init__(self, data, sr, preload_data, transform=None):
        self.data = data
        self.sr = sr
        self.transform = transform
        self.preload_data = preload_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        item = self.data.iloc[idx]
        if not self.preload_data:
            feature = load_wav(item["path"], sr=self.sr)
        else:
            feature = item["wav"]

        return feature, item["transcript"]
