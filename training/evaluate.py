"""
training/evaluate.py
---------------------
Evaluate NER (seqeval F1) and summarization (ROUGE-L) on test sets.
"""

import json
import argparse
from pathlib import Path


def evaluate_ner(predictions_file: str, labels_file: str):
    """Evaluate NER model using seqeval F1."""
    from seqeval.metrics import (
        f1_score, precision_score, recall_score, classification_report
    )
    with open(predictions_file) as f:
        preds = json.load(f)
    with open(labels_file) as f:
        labels = json.load(f)

    print("=== NER Evaluation (seqeval) ===")
    print(f"F1       : {f1_score(labels, preds):.4f}")
    print(f"Precision: {precision_score(labels, preds):.4f}")
    print(f"Recall   : {recall_score(labels, preds):.4f}")
    print("\nPer-entity breakdown:")
    print(classification_report(labels, preds))


def evaluate_summaries(predictions_file: str, references_file: str):
    """Evaluate summaries using ROUGE-L."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    with open(predictions_file) as f:
        preds = [line.strip() for line in f]
    with open(references_file) as f:
        refs = [line.strip() for line in f]

    r1 = r2 = rl = 0.0
    n = len(preds)
    for p, r in zip(preds, refs):
        scores = scorer.score(r, p)
        r1 += scores['rouge1'].fmeasure
        r2 += scores['rouge2'].fmeasure
        rl += scores['rougeL'].fmeasure

    print("=== Summarization Evaluation (ROUGE) ===")
    print(f"ROUGE-1: {r1/n:.4f}")
    print(f"ROUGE-2: {r2/n:.4f}")
    print(f"ROUGE-L: {rl/n:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['ner', 'summarization'], required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--references', required=True)
    args = parser.parse_args()

    if args.task == 'ner':
        evaluate_ner(args.predictions, args.references)
    else:
        evaluate_summaries(args.predictions, args.references)
