"""
training/train_summarizer.py
-----------------------------
Fine-tune facebook/bart-base on the IN-Abs dataset
(Indian Supreme Court abstractive summaries).

This script is for OFFLINE use only.

Dataset
-------
From: https://github.com/Law-AI/summarization
      IN-Abs: full judgment + headnotes pairs from Indian SC judgments
"""

from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "facebook/bart-base"
OUTPUT_DIR  = str(Path(__file__).parent.parent / "models" / "bart_legal_finetuned")
LOGGING_DIR = str(Path(__file__).parent.parent / "models" / "logs_bart")

LEARNING_RATE    = 3e-5
BATCH_SIZE       = 8
EPOCHS           = 3
MAX_INPUT_LEN    = 1024
MAX_TARGET_LEN   = 256
LABEL_SMOOTHING  = 0.1

rouge = evaluate.load("rouge")


def preprocess(examples, tokenizer):
    inputs = tokenizer(
        examples["document"],
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding=False,
    )
    targets = tokenizer(
        examples["summary"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding=False,
    )
    inputs["labels"] = targets["input_ids"]
    return inputs


def compute_metrics(eval_pred, tokenizer):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {"rouge-L": round(result["rougeL"], 4)}


def main():
    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = BartTokenizer.from_pretrained(BASE_MODEL)

    # Load IN-Abs dataset
    # Option 1: Load from HuggingFace if available
    # Option 2: Load from local CSV (document, summary columns)
    try:
        dataset = load_dataset("law-ai/IN-Abs")
        print("Loaded IN-Abs from HuggingFace.")
    except Exception:
        print("IN-Abs not on HuggingFace. Loading from local CSV...")
        import pandas as pd
        train_df = pd.read_csv("data/IN_Abs_train.csv")
        dev_df   = pd.read_csv("data/IN_Abs_dev.csv")
        from datasets import DatasetDict
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(dev_df),
        })

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        lambda ex: preprocess(ex, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print(f"Loading model: {BASE_MODEL}")
    model = BartForConditionalGeneration.from_pretrained(BASE_MODEL)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        label_smoothing_factor=LABEL_SMOOTHING,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rouge-L",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        logging_dir=LOGGING_DIR,
        logging_steps=100,
        push_to_hub=False,
        report_to="none",
        fp16=True,   # RTX 3050 supports fp16
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda ep: compute_metrics(ep, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\nStarting fine-tuning — {EPOCHS} epochs...")
    trainer.train()

    print(f"\nSaving best model to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
