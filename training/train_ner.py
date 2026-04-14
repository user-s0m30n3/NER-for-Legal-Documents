"""
training/train_ner.py
---------------------
Fine-tune law-ai/InLegalBERT on the EkStep Legal NER dataset.

This script is for OFFLINE use only — not required for pipeline inference.
Run this to produce a fine-tuned NER checkpoint that can replace the
spaCy model in Phase 2 of the pipeline (Track B).

Usage
-----
python training/train_ner.py

Dataset
-------
opennyaiorg/InLegalNER on HuggingFace Datasets
Label set: PETITIONER, RESPONDENT, JUDGE, COURT, LAWYER, DATE, ORG, GPE,
           STATUTE, PROVISION, PRECEDENT, CASE_NUMBER, WITNESS, OTHER_PERSON
"""

import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "law-ai/InLegalBERT"
DATASET_NAME = "opennyaiorg/InLegalNER"
OUTPUT_DIR   = str(Path(__file__).parent.parent / "models" / "inlegalbert_ner_finetuned")
LOGGING_DIR  = str(Path(__file__).parent.parent / "models" / "logs")

LEARNING_RATE = 2e-5
BATCH_SIZE    = 16
EPOCHS        = 5
MAX_SEQ_LEN   = 512

# EkStep label set (BIO scheme)
LABELS = [
    "O",
    "B-COURT", "I-COURT",
    "B-PETITIONER", "I-PETITIONER",
    "B-RESPONDENT", "I-RESPONDENT",
    "B-JUDGE", "I-JUDGE",
    "B-LAWYER", "I-LAWYER",
    "B-DATE", "I-DATE",
    "B-ORG", "I-ORG",
    "B-GPE", "I-GPE",
    "B-STATUTE", "I-STATUTE",
    "B-PROVISION", "I-PROVISION",
    "B-PRECEDENT", "I-PRECEDENT",
    "B-CASE_NUMBER", "I-CASE_NUMBER",
    "B-WITNESS", "I-WITNESS",
    "B-OTHER_PERSON", "I-OTHER_PERSON",
]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenize text and align BIO labels to subword tokens.
    For each word, labels are copied to the first subword;
    subsequent subwords get label -100 (ignored in loss).
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_SEQ_LEN,
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(labels[word_id])
            else:
                label_ids.append(-100)  # subword token
            prev_word_id = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def compute_metrics(eval_pred):
    """Compute seqeval F1, precision, recall on NER predictions."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_preds = []
    true_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        p_row, l_row = [], []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                p_row.append(ID2LABEL[p])
                l_row.append(ID2LABEL[l])
        true_preds.append(p_row)
        true_labels.append(l_row)

    return {
        "f1":        f1_score(true_labels, true_preds),
        "precision": precision_score(true_labels, true_preds),
        "recall":    recall_score(true_labels, true_preds),
    }


def main():
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    print(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Tokenizing dataset...")
    tokenized_ds = dataset.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    print(f"Loading model: {BASE_MODEL}")
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=LOGGING_DIR,
        logging_steps=50,
        push_to_hub=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"\nStarting training — {EPOCHS} epochs...")
    trainer.train()

    print(f"\nSaving best model to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nEvaluation on test set:")
    results = trainer.evaluate(tokenized_ds["test"])
    print(results)


if __name__ == "__main__":
    main()
