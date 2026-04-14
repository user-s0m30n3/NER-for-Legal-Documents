"""
run_pipeline.py
---------------
CLI test harness for the VeriSum-Legal pipeline.

Usage
-----
# Single PDF:
python run_pipeline.py --pdf data/sample_judgments/judgment1.pdf

# All PDFs in the sample_judgments folder:
python run_pipeline.py --all

# Quick NER only (skip summarization):
python run_pipeline.py --pdf data/sample_judgments/judgment1.pdf --ner-only

# Use doc-level NER (faster, slightly less accurate):
python run_pipeline.py --pdf data/sample_judgments/judgment1.pdf --fast
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Make pipeline importable without installing the package ───────────────────
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.chunker import extract_text_from_pdf, preprocess_text, sliding_window_chunks
from pipeline.ner_infer import load_ner_models, run_ner, aggregate_entities
from pipeline.summarizer_infer import load_summarizer, generate_summary
from pipeline.verifier import verify_summary, extract_provisions_from_text

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def run_single(
    pdf_path: str,
    legal_nlp,
    preamble_nlp,
    bart_tokenizer,
    bart_model,
    bart_device: str,
    ner_only: bool = False,
    fast: bool = False,
) -> dict:
    """
    Run the full VeriSum-Legal pipeline on a single PDF.

    Returns
    -------
    dict
        Result with keys: file, entities, summary, verification, timing
    """
    pdf_path = Path(pdf_path)
    timing = {}
    result = {"file": str(pdf_path.name)}

    print_section(f"Processing: {pdf_path.name}")

    # ── Phase 1: Extract & Preprocess ─────────────────────────────────────────
    t0 = time.time()
    print("[1/4] Extracting text from PDF...")
    raw_text = extract_text_from_pdf(str(pdf_path))
    clean_text = preprocess_text(raw_text)
    timing['extraction_sec'] = round(time.time() - t0, 2)

    char_count = len(clean_text)
    est_tokens = char_count // 4  # rough estimate: ~4 chars/token
    print(f"      -> {char_count:,} characters | ~{est_tokens:,} tokens")

    # ── Phase 2: NER ──────────────────────────────────────────────────────────
    t1 = time.time()
    run_type = 'doc' if fast else 'sent'
    print(f"[2/4] Running NER (mode={run_type})...")
    entity_list = run_ner(clean_text, legal_nlp, preamble_nlp, run_type=run_type)
    entities = aggregate_entities(entity_list)
    timing['ner_sec'] = round(time.time() - t1, 2)

    # Count unique named entities
    total_ents = sum(len(v) for v in entities.values())
    print(f"      -> {len(entity_list)} raw spans | {total_ents} unique entities extracted")

    # Print entity summary
    for etype, vals in entities.items():
        if vals:
            preview = ", ".join(f'"{v}"' for v in vals[:3])
            more = f" (+{len(vals)-3} more)" if len(vals) > 3 else ""
            print(f"      {etype:15s}: {preview}{more}")

    result['entities'] = entities
    result['timing'] = timing

    if ner_only:
        result['summary'] = None
        result['verification'] = None
        return result

    # ── Phase 3: Summarize ────────────────────────────────────────────────────
    t2 = time.time()
    print(f"\n[3/4] Generating summary (BART on {bart_device.upper()})...")
    summary = generate_summary(
        clean_text, entities, bart_tokenizer, bart_model, bart_device
    )
    timing['summarization_sec'] = round(time.time() - t2, 2)
    print(f"      -> {len(summary.split())} words generated in {timing['summarization_sec']}s")
    print(f"\n--- SUMMARY ---\n{summary}\n--- END ---")

    result['summary'] = summary

    # ── Phase 4: Verify ───────────────────────────────────────────────────────
    t3 = time.time()
    print("\n[4/4] Running hallucination verification...")
    verification = verify_summary(entities, summary, clean_text)
    timing['verification_sec'] = round(time.time() - t3, 2)

    status = verification['overall_status']
    status_icon = {'VERIFIED': '[PASS]', 'WARNING': '[WARN]', 'CRITICAL': '[FAIL]'}.get(status, '?')
    print(f"      -> Status: {status_icon} {status}")

    if verification['flags']:
        for flag in verification['flags']:
            print(f"      {flag}")

    timing['total_sec'] = round(time.time() - t0, 2)
    result['verification'] = verification
    result['timing'] = timing

    return result


def save_result(result: dict, pdf_name: str):
    """Save result as JSON to the output/ directory."""
    stem = Path(pdf_name).stem
    out_path = OUTPUT_DIR / f"{stem}_result.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Result saved -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="VeriSum-Legal CLI — Indian Court Judgment Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pdf', type=str, help='Path to a single PDF file')
    group.add_argument('--all', action='store_true',
                       help='Process all PDFs in data/sample_judgments/')
    parser.add_argument('--ner-only', action='store_true',
                        help='Run NER only, skip summarization and verification')
    parser.add_argument('--fast', action='store_true',
                        help='Use doc-level NER (faster but slightly less accurate)')
    parser.add_argument('--no-quantize', action='store_true',
                        help='Disable 4-bit quantization (use full precision)')
    args = parser.parse_args()

    # ── Load models (once) ────────────────────────────────────────────────────
    print_section("Loading Models")

    print("[NER] Loading en_legal_ner_trf + en_core_web_sm...")
    t = time.time()
    legal_nlp, preamble_nlp = load_ner_models()
    print(f"      -> NER models loaded in {round(time.time()-t, 1)}s")

    bart_tokenizer = bart_model = bart_device = None
    if not args.ner_only:
        print("[BART] Loading facebook/bart-base...")
        t = time.time()
        bart_tokenizer, bart_model, bart_device = load_summarizer(
            quantize=not args.no_quantize
        )
        print(f"      -> BART loaded in {round(time.time()-t, 1)}s")

    # ── Determine which PDFs to process ──────────────────────────────────────
    if args.all:
        sample_dir = Path(__file__).parent / "data" / "sample_judgments"
        pdf_files = sorted(sample_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"\n[!] No PDF files found in {sample_dir}")
            print("    -> Put your judgment PDFs there and re-run.")
            sys.exit(1)
    else:
        pdf_files = [Path(args.pdf)]
        if not pdf_files[0].exists():
            print(f"[!] File not found: {pdf_files[0]}")
            sys.exit(1)

    # ── Process each PDF ──────────────────────────────────────────────────────
    all_results = []
    for pdf_path in pdf_files:
        try:
            result = run_single(
                pdf_path=str(pdf_path),
                legal_nlp=legal_nlp,
                preamble_nlp=preamble_nlp,
                bart_tokenizer=bart_tokenizer,
                bart_model=bart_model,
                bart_device=bart_device,
                ner_only=args.ner_only,
                fast=args.fast,
            )
            save_result(result, pdf_path.name)
            all_results.append(result)

            print(f"\n[TIMING] breakdown:")
            for k, v in result.get('timing', {}).items():
                print(f"   {k:25s}: {v}s")

        except Exception as e:
            print(f"\n[ERROR] Failed to process {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary table across all PDFs ────────────────────────────────────────
    if len(all_results) > 1:
        print_section("Batch Summary")
        for r in all_results:
            status = r.get('verification', {})
            if status:
                s = status.get('overall_status', 'N/A')
            else:
                s = 'NER-ONLY'
            t_total = r.get('timing', {}).get('total_sec', '?')
            print(f"  {r['file']:40s} | {s:10s} | {t_total}s")


if __name__ == '__main__':
    main()
