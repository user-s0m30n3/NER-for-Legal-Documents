"""
pipeline/summarizer_infer.py
-----------------------------
Abstractive summarization using facebook/bart-base with entity injection.

The entity prefix steers BART to mention the most important legal entities
in its generated summary, improving factual grounding.

Functions
---------
load_summarizer()  -> Tuple[BartTokenizer, BartForConditionalGeneration, str]
build_entity_prefix(entities) -> str
generate_summary(text, entities, model, tokenizer, device) -> str
"""

import torch
import warnings
from typing import Dict, List, Tuple

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    BitsAndBytesConfig,
)


MODEL_NAME = "facebook/bart-large-cnn"

# Maximum tokens BART can handle (hard architectural limit)
BART_MAX_TOKENS = 1024

# Token budget reserved for the entity prefix
PREFIX_TOKEN_BUDGET = 150

# Available tokens for the judgment text
TEXT_TOKEN_BUDGET = BART_MAX_TOKENS - PREFIX_TOKEN_BUDGET - 5  # 5 safety margin


def load_summarizer(quantize: bool = True) -> Tuple:
    """
    Load facebook/bart-base for conditional generation.

    GPU Strategy (RTX 3050):
    - If CUDA available + bitsandbytes installed: load in 4-bit (fastest, lowest VRAM)
    - If CUDA available but bitsandbytes fails: load in float16 on GPU
    - If no CUDA: load in float32 on CPU

    Parameters
    ----------
    quantize : bool
        Whether to attempt 4-bit quantization (requires GPU).

    Returns
    -------
    Tuple[BartTokenizer, BartForConditionalGeneration, str]
        (tokenizer, model, device_string)
    """
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Summarizer] Loading {MODEL_NAME} on {device.upper()}...")

    model = None

    if device == "cuda" and quantize:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = BartForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
            )
            print("[Summarizer] Loaded in 4-bit quantized mode (GPU).")
        except Exception as e:
            warnings.warn(f"[Summarizer] 4-bit quantization failed ({e}). "
                          "Falling back to float16 on GPU.")
            model = None

    if model is None and device == "cuda":
        try:
            model = BartForConditionalGeneration.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float16
            ).to(device)
            print("[Summarizer] Loaded in float16 mode (GPU).")
        except Exception as e:
            warnings.warn(f"[Summarizer] float16 GPU load failed ({e}). "
                          "Falling back to CPU float32.")
            model = None
            device = "cpu"

    if model is None:
        model = BartForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32
        )
        device = "cpu"
        print("[Summarizer] Loaded in float32 mode (CPU). "
              "Summarization will be slow (~60-120s per document).")

    model.eval()
    return tokenizer, model, device


def build_entity_prefix(entities: Dict[str, List[str]]) -> str:
    """
    Build the entity injection prefix for BART input.

    Takes the top values from key entity types and
    formats them as a structured natural-language prefix to guide the summary.

    Parameters
    ----------
    entities : Dict[str, List[str]]
        Output of aggregate_entities() from ner_infer.py.

    Returns
    -------
    str
        The formatted prefix string.
    """
    parts = []
    
    # Use only the absolute most critical entities for steering to avoid confusing CNN models
    for etype in ['JUDGE', 'COURT', 'PETITIONER', 'RESPONDENT']:
        vals = entities.get(etype, [])
        if vals:
            parts.append(f"{etype} was {vals[0].strip()}")

    if not parts:
        return "Summary of the following text, ensuring the final court order and outcome are included: "

    prefix = "Context: " + ", ".join(parts) + ". Provide a legal abstract including the final outcome/order: "
    return prefix


def generate_summary(
    text: str,
    entities: Dict[str, List[str]],
    tokenizer: BartTokenizer,
    model: BartForConditionalGeneration,
    device: str,
    max_length: int = 400,
    min_length: int = 200,
) -> str:
    """
    Generate an abstractive summary of the judgment with entity injection.

    Process:
    1. Build entity prefix from top NER entities
    2. Count prefix tokens
    3. Truncate judgment text using Lead + Tail strategy to capture both 
       factual background (head) and final order (tail).
    4. Concatenate prefix + head + tail
    5. Run beam search generation

    Parameters
    ----------
    text : str
        Preprocessed full judgment text.
    entities : Dict[str, List[str]]
        Aggregated entities from aggregate_entities().
    tokenizer : BartTokenizer
    model : BartForConditionalGeneration
    device : str
        "cuda" or "cpu"
    max_length : int
        Maximum length of generated summary in tokens.
    min_length : int
        Minimum length of generated summary in tokens.

    Returns
    -------
    str
        Decoded summary string.
    """
    # 1. Build entity prefix
    prefix = build_entity_prefix(entities)

    # 2. Count prefix tokens
    prefix_ids = tokenizer(
        prefix,
        add_special_tokens=False,
        return_tensors=None,
    )['input_ids']
    prefix_token_count = len(prefix_ids)

    # 3. Calculate available tokens for the text (Head + Tail)
    available = BART_MAX_TOKENS - prefix_token_count - 10  # 10 for special tokens/separator
    
    # Lead+Tail strategy: Take first 60% and last 40% of the budget
    head_budget = int(available * 0.6)
    tail_budget = available - head_budget

    # Tokenize full text
    full_ids = tokenizer(text, add_special_tokens=False)['input_ids']
    
    if len(full_ids) <= available:
        selected_ids = full_ids
    else:
        head_ids = full_ids[:head_budget]
        tail_ids = full_ids[-tail_budget:]
        # Joined with a separator token [SEP] or similar if needed, 
        # but BART just needs the sequence.
        selected_ids = head_ids + tail_ids

    truncated_text = tokenizer.decode(
        selected_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # 4. Build the full input
    full_input = prefix + truncated_text

    # 5. Tokenize combined input
    inputs = tokenizer(
        full_input,
        max_length=BART_MAX_TOKENS,
        truncation=True,
        return_tensors="pt",
    )

    # Move to device
    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    # 6. Generate summary
    with torch.no_grad():
        output_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    # 7. Decode
    summary = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    
    # 8. Post-generation cleaning (strip T.T.'T. artifacts etc.)
    from pipeline.chunker import clean_ocr_noise
    summary = clean_ocr_noise(summary)
    
    return summary.strip()
