import evaluate
import torch

class Metric:
    def __init__(self, processor):
        self.processor = processor
        # load WER metric using the `evaluate` library
        self.wer_metric = evaluate.load("wer")

    def __call__(self, logits, labels):
        # get predicted token ids
        preds = torch.argmax(logits, axis=-1)

        # replace ignore index (-100) with pad token id
        labels[labels == -100] = self.processor.tokenizer.pad_token_id

        # decode predictions and labels to strings
        pred_strs = self.processor.batch_decode(preds)
        # we do not want to group tokens when computing the metrics
        label_strs = self.processor.batch_decode(labels, group_tokens=False)

        # compute WER
        wer = self.wer_metric.compute(predictions=pred_strs, references=label_strs)
        return wer
