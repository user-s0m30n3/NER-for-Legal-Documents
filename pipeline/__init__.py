"""
VeriSum-Legal pipeline package.

Expose top-level entry points for easy imports.
"""
from .chunker import extract_text_from_pdf, preprocess_text, sliding_window_chunks
from .ner_infer import load_ner_models, run_ner, aggregate_entities
from .summarizer_infer import load_summarizer, generate_summary
from .verifier import verify_summary, extract_provisions_from_text

__all__ = [
    "extract_text_from_pdf",
    "preprocess_text",
    "sliding_window_chunks",
    "load_ner_models",
    "run_ner",
    "aggregate_entities",
    "load_summarizer",
    "generate_summary",
    "verify_summary",
    "extract_provisions_from_text",
]
