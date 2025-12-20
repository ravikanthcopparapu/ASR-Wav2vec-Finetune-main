import argparse
import json
import torch
import toml
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor
)

from utils.metric import Metric
from dataloader.dataset import DefaultCollate
from utils.utils import initialize_module, set_seed


@torch.no_grad()
def main(args):
    # --------------------------------------------------
    # Load config
    # --------------------------------------------------
    config = toml.load(args.config)
    set_seed(config["meta"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------------------------------
    # Dataset config
    # --------------------------------------------------
    config["test_dataset"]["args"]["sr"] = config["meta"]["sr"]
    config["test_dataset"]["args"]["special_tokens"] = config["special_tokens"]

    # --------------------------------------------------
    # Load test dataset
    # --------------------------------------------------
    test_base_ds = initialize_module(
        config["test_dataset"]["path"],
        args=config["test_dataset"]["args"]
    )

    test_ds = test_base_ds.get_data()

    print("Number of test utterances:", len(test_ds))

    # --------------------------------------------------
    # Load vocab (generated during training)
    # --------------------------------------------------
    vocab_path = args.vocab
    assert os.path.exists(vocab_path), f"Missing vocab file: {vocab_path}"

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        **config["special_tokens"],
        word_delimiter_token="|"
    )

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        config["meta"]["pretrained_path"]
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    collate_fn = DefaultCollate(processor, config["meta"]["sr"])

    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = Wav2Vec2ForCTC.from_pretrained(
        config["meta"]["pretrained_path"],
        ctc_loss_reduction="sum",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)

    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Metric
    # --------------------------------------------------
    compute_metric = Metric(processor)

    total_loss = 0.0
    total_batches = 0
    all_wer = []

    # --------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------
    for batch in tqdm(test_dl, desc="Testing"):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        total_loss += loss.item()
        total_batches += 1

        wer = compute_metric(logits, batch["labels"])
        all_wer.append(wer)

    # --------------------------------------------------
    # Results
    # --------------------------------------------------
    avg_loss = total_loss / total_batches
    avg_wer = sum(all_wer) / len(all_wer)

    print("\n========== TEST RESULTS ==========")
    print(f"Average Test Loss : {avg_loss:.4f}")
    print(f"Average Test WER  : {avg_wer:.4f}")
    print("=================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR TEST")

    parser.add_argument(
        "-c", "--config",
        required=True,
        type=str,
        help="Path to config.toml"
    )

    parser.add_argument(
        "-t", "--checkpoint",
        required=True,
        type=str,
        help="Path to trained model checkpoint (.tar)"
    )

    parser.add_argument(
        "-v", "--vocab",
        default="vocab.json",
        type=str,
        help="Path to vocab.json generated during training"
    )

    args = parser.parse_args()
    main(args)
