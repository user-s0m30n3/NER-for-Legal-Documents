"""
pipeline/chunker.py
-------------------
PDF ingestion, text preprocessing, and sliding-window chunking.

Functions
---------
extract_text_from_pdf(pdf_path) -> str
preprocess_text(text) -> str
sliding_window_chunks(text, tokenizer, max_tokens, overlap) -> List[Dict]
"""

import re
import pdfplumber
from typing import List, Dict, Optional


# ── Patterns that appear as noise in Indian Kanoon / court PDFs ──────────────
_NOISE_PATTERNS = [
    re.compile(r'Indian Kanoon\s*-\s*http[s]?://\S+', re.IGNORECASE),
    re.compile(r'Digitally Signed By\s*:.*', re.IGNORECASE),
    re.compile(r'Signing Date\s*:.*', re.IGNORECASE),
    re.compile(r'Reason\s*:.*', re.IGNORECASE),
    re.compile(r'Location\s*:.*', re.IGNORECASE),
    # Repeated page numbers like "Page 1 of 23" or "1/23"
    re.compile(r'\bPage\s+\d+\s+of\s+\d+\b', re.IGNORECASE),
    re.compile(r'\b\d+\s*/\s*\d+\b'),
]

# ── Sentence boundary punctuation for chunk snapping ─────────────────────────
_SENT_ENDS = re.compile(r'[.!?]\s')


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract full text from a PDF using pdfplumber.

    Handles:
    - Multi-page documents
    - Broken hyphenation across lines (re-joins words)
    - Garbled Unicode (replaces with U+FFFD then strips)
    - Preserves paragraph structure (double newlines)

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to the PDF file.

    Returns
    -------
    str
        Raw extracted text, pages joined with newline.
    """
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
    except Exception as e:
        raise IOError(f"Failed to read PDF '{pdf_path}': {e}")

    if not pages:
        raise ValueError(f"No extractable text found in '{pdf_path}'. "
                         "The PDF may be scanned/image-based.")

    raw = "\n".join(pages)

    # Re-join broken hyphenation: "arbitra-\ntion" → "arbitration"
    raw = re.sub(r'-\n(\w)', r'\1', raw)

    # Strip garbled Unicode replacement characters
    raw = raw.replace('\ufffd', '')

    return raw


# ── OCR and Artifact Cleaning ───────────────────────────────────────────────
_OCR_JUNK_RE = re.compile(
    r'\b(?:[A-Z][\.\-\']\s?){2,}[A-Z]\b'   # T.T.'T. or G.G.-T. style noise
    r'|\b(?:[a-z][\.\-\']\s?){2,}[a-z]\b'
)

def clean_ocr_noise(text: str) -> str:
    """Remove common OCR artifacts and repetitive non-semantic character patterns."""
    text = _OCR_JUNK_RE.sub('', text)
    # Also collapse excessive punctuation artifacts
    text = re.sub(r'\.{4,}', '...', text)
    return text.strip()

def preprocess_text(text: str) -> str:
    """
    Clean text extracted from Indian court judgment PDFs.

    Removes:
    - Indian Kanoon headers / footers
    - Digital signature watermark lines
    - Repeated page-number patterns

    Normalizes:
    - Multiple blank lines → single blank line (preserves paragraphs)
    - Intra-paragraph newlines → single space (sentences flow)
    - Tabs and irregular whitespace

    Parameters
    ----------
    text : str
        Raw text from extract_text_from_pdf().

    Returns
    -------
    str
        Cleaned text with paragraph structure intact.
    """
    lines = text.split('\n')
    cleaned = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append('')  # preserve paragraph breaks
            continue

        # Remove noise patterns
        skip = False
        for pattern in _NOISE_PATTERNS:
            if pattern.search(stripped):
                skip = True
                break
        if skip:
            continue

        cleaned.append(stripped)

    # Join and collapse multiple blank lines to one
    joined = '\n'.join(cleaned)
    joined = clean_ocr_noise(joined)  # Apply OCR artifacts cleaning
    joined = re.sub(r'\n{3,}', '\n\n', joined)

    # Normalize tabs and non-breaking spaces
    joined = joined.replace('\t', ' ').replace('\xa0', ' ')

    # Collapse runs of spaces within a line
    joined = re.sub(r'[ ]{2,}', ' ', joined)

    return joined.strip()


def sliding_window_chunks(
    text: str,
    tokenizer,
    max_tokens: int = 450,
    overlap: int = 50,
) -> List[Dict]:
    """
    Split text into overlapping chunks aligned to sentence boundaries.

    The function tokenizes the full text, divides it into windows of
    `max_tokens` with `overlap` tokens between consecutive windows,
    then snaps each boundary back to the nearest sentence-end so that
    sentences are never split across chunks.

    Parameters
    ----------
    text : str
        Preprocessed judgment text.
    tokenizer :
        A HuggingFace tokenizer (used for token counting only).
    max_tokens : int
        Maximum tokens per chunk (default 450, leaving room for special tokens).
    overlap : int
        Token overlap between consecutive chunks (default 50).

    Returns
    -------
    List[Dict]
        Each dict has keys:
        - chunk_id   : int
        - text       : str   (decoded chunk text)
        - token_start: int   (token index in the full sequence)
        - token_end  : int   (token index in the full sequence)
    """
    # Tokenize entire text (no special tokens — we just need offsets)
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    token_ids = encoding['input_ids']
    offsets = encoding['offset_mapping']  # list of (char_start, char_end) per token
    total_tokens = len(token_ids)

    if total_tokens == 0:
        return []

    stride = max_tokens - overlap
    chunks = []
    chunk_id = 0
    token_start = 0

    while token_start < total_tokens:
        token_end = min(token_start + max_tokens, total_tokens)

        # ── Snap token_end back to nearest sentence boundary ─────────────────
        if token_end < total_tokens:
            # Get the char range for this chunk
            chunk_char_end = offsets[token_end - 1][1]
            chunk_char_start = offsets[token_start][0]
            chunk_text_raw = text[chunk_char_start:chunk_char_end]

            # Find last sentence boundary in the chunk
            last_sent_end = None
            for m in _SENT_ENDS.finditer(chunk_text_raw):
                last_sent_end = chunk_char_start + m.end()

            if last_sent_end is not None:
                # Snap token_end to the token covering last_sent_end
                for i in range(token_end - 1, token_start, -1):
                    if offsets[i][1] <= last_sent_end:
                        token_end = i + 1
                        break

        # ── Extract the char span for this token window ───────────────────────
        char_start = offsets[token_start][0]
        char_end   = offsets[token_end - 1][1]
        chunk_text = text[char_start:char_end]

        chunks.append({
            'chunk_id':    chunk_id,
            'text':        chunk_text,
            'token_start': token_start,
            'token_end':   token_end,
        })

        chunk_id += 1
        token_start += stride

        # If the remaining tokens are fewer than `overlap`, we're done
        if token_start >= total_tokens:
            break

    return chunks


def get_chunk_summary(chunks: List[Dict]) -> Dict:
    """Return quick stats about the chunked document."""
    if not chunks:
        return {'num_chunks': 0, 'total_tokens': 0, 'avg_tokens_per_chunk': 0}
    total = chunks[-1]['token_end']
    return {
        'num_chunks': len(chunks),
        'total_tokens': total,
        'avg_tokens_per_chunk': round(total / len(chunks), 1),
    }
