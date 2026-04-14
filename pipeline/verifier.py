"""
pipeline/verifier.py
--------------------
Deterministic hallucination detection — pure Python, no ML.

Functions
---------
extract_provisions_from_text(text) -> Set[str]
verify_summary(ner_entities, summary, original_text) -> Dict
"""

import re
from typing import Dict, List, Set


# ── Regex patterns to find legal provisions anywhere in text ─────────────────
_PROVISION_PATTERNS = [
    re.compile(r'[Ss]ection\s+\d+[A-Z]?(?:\(\d+\))?(?:\([a-zA-Z]\))?'),
    re.compile(r'[Aa]rticle\s+\d+[A-Z]?(?:\(\d+\))?'),
    re.compile(r'[Cc]lause\s+\d+[A-Z]?(?:\(\w+\))?'),
    re.compile(r'[Rr]ule\s+\d+[A-Z]?'),
    re.compile(r'[Oo]rder\s+[IVXLC]+\s+[Rr]ule\s+\d+'),
]

# ── Citation patterns ─────────────────────────────────────────────────────────
_CITATION_PATTERN = re.compile(
    r'\(\d{4}\)\s+\d+\s+[A-Z]+\s+\d+'          # (2014) 2 SCC 62
    r'|AIR\s+\d{4}\s+[A-Z]+\s+\d+'             # AIR 2003 SC 2629
    r'|\d{4}\s+INSC\s+\d+'                      # 2026 INSC 336
)


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for fuzzy comparison."""
    return re.sub(r'\s+', ' ', text.strip().lower())


def extract_provisions_from_text(text: str) -> Set[str]:
    """
    Extract all provision mentions from a block of text using regex.

    Used as a deterministic ground-truth set for hallucination checking.
    Returns normalized (lowercase, whitespace-collapsed) strings.

    Parameters
    ----------
    text : str
        Any block of text — original judgment or generated summary.

    Returns
    -------
    Set[str]
        Normalized provision strings found in the text.
    """
    found = set()
    for pattern in _PROVISION_PATTERNS:
        for m in pattern.finditer(text):
            found.add(_normalize(m.group()))
    return found


def _provisions_in_ner(ner_entities: Dict) -> Set[str]:
    """Normalize all PROVISION entities from NER output."""
    return {_normalize(p) for p in ner_entities.get('PROVISION', [])}


def _entity_in_summary(entity_list: List[str], summary: str) -> bool:
    """Check if any entity in the list appears as a substring in the summary."""
    summary_lower = summary.lower()
    for entity in entity_list:
        # Partial match: at least one significant word must appear
        significant = [w for w in entity.strip().split() if len(w) > 3]
        if significant and any(w.lower() in summary_lower for w in significant):
            return True
        if entity.strip().lower() in summary_lower:
            return True
    return False


def verify_summary(
    ner_entities: Dict,
    summary: str,
    original_text: str,
) -> Dict:
    """
    Run hallucination and completeness checks on the generated summary.

    Three Checks
    ------------
    1. Provision hallucination:
       Extract provisions from summary via regex. For each, check if it
       exists in ner_entities['PROVISION'] OR in the original text regex set.
       Flag any that don't appear in either source as HALLUCINATED.

    2. Entity presence:
       For JUDGE, PETITIONER, RESPONDENT — verify at least one NER value
       appears (substring match) in the summary. Flag missing ones.

    3. Court/case consistency:
       Check if the COURT name from NER appears in the summary.

    Parameters
    ----------
    ner_entities : Dict
        Output of aggregate_entities() from ner_infer.py.
    summary : str
        Generated summary from summarizer_infer.py.
    original_text : str
        Full preprocessed judgment text.

    Returns
    -------
    Dict with keys:
        hallucinated_provisions : List[str]
        missing_entities        : List[str]
        verified_provisions     : List[str]
        overall_status          : str  ("CRITICAL" | "WARNING" | "VERIFIED")
        flags                   : List[str]  (human-readable messages)
    """
    flags: List[str] = []
    hallucinated: List[str] = []
    missing: List[str] = []
    verified: List[str] = []

    # ── Ground truth: regex provisions from the original document ─────────────
    source_provisions = extract_provisions_from_text(original_text)
    ner_provisions    = _provisions_in_ner(ner_entities)
    all_valid         = source_provisions | ner_provisions

    # ── Check 1: Provision hallucination ─────────────────────────────────────
    summary_provisions = extract_provisions_from_text(summary)
    for prov in summary_provisions:
        if prov in all_valid:
            verified.append(prov)
        else:
            hallucinated.append(prov)
            flags.append(
                f"[FAIL] '{prov}' mentioned in summary but NOT found in original document"
            )

    # ── Check 2: Key entity presence ─────────────────────────────────────────
    for entity_type in ['JUDGE', 'PETITIONER', 'RESPONDENT']:
        entity_vals = ner_entities.get(entity_type, [])
        if entity_vals and not _entity_in_summary(entity_vals, summary):
            missing.append(entity_type)
            flags.append(
                f"[WARN] {entity_type} '{entity_vals[0]}' found by NER but not mentioned in summary"
            )

    # ── Check 3: Court/case consistency ──────────────────────────────────────
    court_vals = ner_entities.get('COURT', [])
    if court_vals and not _entity_in_summary(court_vals, summary):
        missing.append('COURT')
        flags.append(
            f"[WARN] COURT '{court_vals[0]}' found by NER but not mentioned in summary"
        )

    # ── Overall status ────────────────────────────────────────────────────────
    if hallucinated:
        status = 'CRITICAL'
    elif missing:
        status = 'WARNING'
    else:
        status = 'VERIFIED'

    return {
        'hallucinated_provisions': hallucinated,
        'missing_entities':        missing,
        'verified_provisions':     verified,
        'overall_status':          status,
        'flags':                   flags,
    }
