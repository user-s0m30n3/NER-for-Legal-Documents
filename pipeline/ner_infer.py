"""
pipeline/ner_infer.py
---------------------
Named Entity Recognition for Indian court judgments.

Uses opennyaiorg/en_legal_ner_trf (spaCy/RoBERTa) as NER engine.
Adapts postprocessing logic from Legal-NLP-EkStep/legal_NER (Apache-2.0).

Functions
---------
load_ner_models()  -> Tuple[spacy.Language, spacy.Language]
run_ner(text, legal_nlp, preamble_nlp) -> List[Dict]
aggregate_entities(entity_list) -> Dict[str, List[str]]
"""

import re
import spacy
import warnings
from typing import List, Dict, Tuple, Set, Optional

try:
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    pass

# ── Label remapping: EkStep labels → project spec labels ─────────────────────
LABEL_MAP = {
    'ORG':          'ORGANIZATION',
    'GPE':          'LOCATION',
    'LAWYER':       'LAWYER',        # keep — useful context even if not in spec
    'WITNESS':      'WITNESS',
    'OTHER_PERSON': 'OTHER_PERSON',
}

# Labels to include in the final aggregated output
CANONICAL_LABELS = {
    'PETITIONER', 'RESPONDENT', 'JUDGE', 'COURT',
    'STATUTE', 'PROVISION', 'CASE_NUMBER', 'DATE',
    'PRECEDENT', 'ORGANIZATION', 'LOCATION',
    # Regex-supplemented
    'CITATION', 'AMOUNT', 'LEGAL_TERM',
}

# Hallucinations or jurisdictional terms to filter out
BLACKLIST_ENTITIES = {
    'suzana', 'state of goa', 'rs,' 
}

# Entity color map (for future UI — stored here for single source of truth)
ENTITY_COLORS = {
    'COURT':        '#bbabf2',
    'PETITIONER':   '#f570ea',
    'RESPONDENT':   '#cdee81',
    'JUDGE':        '#fdd8a5',
    'LAWYER':       '#f9d380',
    'WITNESS':      'violet',
    'STATUTE':      '#faea99',
    'PROVISION':    '#ffff00',
    'CASE_NUMBER':  '#fbb1cf',
    'PRECEDENT':    '#fad6d6',
    'DATE':         '#b1ecf7',
    'OTHER_PERSON': '#b0f6a2',
    'ORGANIZATION': '#a57db5',
    'LOCATION':     '#7fdbd4',
    'CITATION':     '#c8e6c9',
    'AMOUNT':       '#ffe0b2',
    'LEGAL_TERM':   '#b3e5fc',
}

# ── Statute acronym normalization (adapted from EkStep check_stat()) ──────────
# Maps regex patterns → canonical statute names
_STATUTE_ALIAS_PATTERNS = [
    (re.compile(r'(?i)\b(((criminal|cr)\.?\s*(procedure|p)\.?\s*(c|code)\.*)|(code\s*of\s*criminal\s*procedure))\s*'),
     'Criminal Procedure Code'),
    (re.compile(r'(?i)\b((i\.?\s*p\.?\s*c\.?)|(indian\s+penal\s+code))\b'),
     'Indian Penal Code'),
    (re.compile(r'(?i)\b(constitution(\s+of\s+india)?)\b'),
     'Constitution of India'),
    (re.compile(r'(?i)\b((i\.?\s*t\.?\s*|income\s*-?\s*tax\s+)act)\b'),
     'Income Tax Act'),
    (re.compile(r'(?i)\b((m\.?\s*v\.?\s*)|(motor\s*-?vehicle(s)?\s+)act)\b'),
     'Motor Vehicles Act'),
    (re.compile(r'(?i)\b((i\.?\s*d\.?\s*)|(industrial\s*-?dispute(s)?\s+)act)\b'),
     'Industrial Disputes Act'),
    (re.compile(r'(?i)\b(sarfaesi|securitisation\s+and\s+reconstruction\s+of\s+financial\s+assets)\b'),
     'SARFAESI Act'),
    (re.compile(r'(?i)\bndps\b'),
     'NDPS Act'),
    (re.compile(r'(?i)\b(arbitration\s+(and\s+conciliation\s+)?act)\b'),
     'Arbitration and Conciliation Act, 1996'),
]

def _normalize_statute(name: str) -> str:
    """Resolve common statute acronyms to canonical names."""
    for pattern, canonical in _STATUTE_ALIAS_PATTERNS:
        if pattern.search(name):
            return canonical
    return name.strip()

# ── Regex extractors for entities not covered by the spaCy model ──────────────
_CITATION_RE = re.compile(
    r'\(\d{4}\)\s+\d+\s+[A-Z]+\s+\d+'         # (2014) 2 SCC 62
    r'|AIR\s+\d{4}\s+[A-Z]+\s+\d+'            # AIR 2003 SC 2629
    r'|\d{4}\s+INSC\s+\d+'                     # 2026 INSC 336
    r'|\d{4}\s+SCC\s+\(Cri\)\s+\d+'           # 2014 SCC (Cri) 123
    r'|\d{4}:[A-Z\-]+:\d{1,5}'                # 2026:BHC-GOA:774
    r'|\d{4}\s+SCC\s+OnLine\s+[A-Z]+\s+\d+'   # 2016 SCC OnLine HP 379
)

_CASE_NUMBER_RE = re.compile(
    r'(?i)\b(?:FIRST\s+APPEAL|MISCELLANEOUS\s+CIVIL\s+APPLICATION|SUIT|W\.?P\.?|WRIT\s+PETITION|C\.?A\.?|CIVIL\s+APPEAL|APPEAL|S\.?L\.?P\.?|F\.?A\.?|M\.?C\.?A\.?|CP)\s*(?:NO\.?\s*)?\d+\s*(?:OF|/|–)\s*\d{4}\b'
)

_AMOUNT_RE = re.compile(
    r'\b(?:Rs\.?|₹|Rupees)\s*\d+[\d,]*(?:/-)?(?:\s*(?:crore|lakh|thousand|hundred))?\b',
    re.IGNORECASE
)

# Curated legal term set commonly found in Indian judgments
_LEGAL_TERMS: Set[str] = {
    "Extension of Time", "Statement of Claim", "Hindrance Register",
    "Arbitral Award", "Counter Claim", "Notice Inviting Tender",
    "Letter of Acceptance", "Scope of Work", "Bill of Quantities",
    "Prolongation Cost", "Liquidated Damages", "Force Majeure",
    "Ex Parte", "Ad Interim", "Status Quo", "Sine Die",
    "Prima Facie", "Inter Alia", "Res Judicata", "Sub Judice",
    "Caveat", "Locus Standi", "Mens Rea", "Actus Reus",
    "Habeas Corpus", "Mandamus", "Certiorari", "Prohibition",
    "Bail", "Anticipatory Bail", "Regular Bail", "Remand",
    "FIR", "Charge Sheet", "Cognizance", "Summons",
    "Preliminary Objection", "Affidavit", "Interlocutory Application",
    "Injuria Grave", "Divorce", "Ill-treatment", "Cruelty",
}

# Terms that should be filtered out in Civil cases to prevent 'logic leaks'
_CRIMINAL_ONLY_TERMS: Set[str] = {
    "FIR", "Bail", "Anticipatory Bail", "Regular Bail", "Remand",
    "Charge Sheet", "Cognizance", "NDPS", "POCSO", "IPC", "CrPC"
}

_PARTY_NOISE_RE = re.compile(
    r'\b(?:and\s+)?(?:anr|ors|another|others|etc|alias|and\s+another|and\s+others)\.?\b',
    re.IGNORECASE
)

def _clean_party_name(name: str) -> str:
    """Strip legal suffixes like 'Anr', 'Ors' and header artifacts."""
    # Remove leading/trailing punctuation and whitespace
    name = name.strip('., :')
    # Remove noise suffixes
    name = _PARTY_NOISE_RE.sub('', name)
    # Final cleanup of multiple spaces or trailing commas
    name = re.sub(r'\s+', ' ', name).strip(' ,')
    return name

def _detect_jurisdiction(entities: Dict[str, List[str]], text: str) -> str:
    """
    Heuristic to determine if the case is Civil or Criminal.
    Defaults to 'Civil' unless strong criminal markers are found.
    """
    markers = entities.get('CASE_NUMBER', []) + entities.get('STATUTE', [])
    text_sample = text[:2000].lower()
    
    crim_keywords = ['criminal', 'ipc', 'crpc', 'fir', 'bail', 'ndps', 'pocso', 'accused', 'convicted']
    civil_keywords = ['civil', 'matrimonial', 'suit', 'f.a.', 'm.c.a.', 'divorce', 'service', 'petitioner']
    
    crim_score = sum(1 for k in crim_keywords if k in text_sample)
    civil_score = sum(1 for k in civil_keywords if k in text_sample)
    
    # Check Case Number prefixes specifically
    for cn in entities.get('CASE_NUMBER', []):
        if re.search(r'(?i)\b(cr\.?|crl\.?)\b', cn):
            crim_score += 5
        if re.search(r'(?i)\b(civil|f\.?a\.?|m\.?c\.?a\.?|suit|w\.?p\.?)\b', cn):
            civil_score += 5
            
    return 'CRIMINAL' if crim_score > civil_score else 'CIVIL'

def _extract_regex_entities(text: str) -> List[Dict]:
    """Extract CITATION, AMOUNT, and LEGAL_TERM entities via regex."""
    entities = []

    for m in _CITATION_RE.finditer(text):
        entities.append({
            'entity_type': 'CITATION',
            'entity_text': m.group().strip(),
            'start_char':  m.start(),
            'end_char':    m.end(),
            'confidence':  1.0,   # deterministic
        })

    for m in _AMOUNT_RE.finditer(text):
        entities.append({
            'entity_type': 'AMOUNT',
            'entity_text': m.group().strip(),
            'start_char':  m.start(),
            'end_char':    m.end(),
            'confidence':  1.0,
        })

    for m in _CASE_NUMBER_RE.finditer(text):
        entities.append({
            'entity_type': 'CASE_NUMBER',
            'entity_text': m.group().strip(),
            'start_char':  m.start(),
            'end_char':    m.end(),
            'confidence':  1.0,
        })

    for term in _LEGAL_TERMS:
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        for m in pattern.finditer(text):
            entities.append({
                'entity_type': 'LEGAL_TERM',
                'entity_text': m.group().strip(),
                'start_char':  m.start(),
                'end_char':    m.end(),
                'confidence':  1.0,
            })

    return entities


# ── Preamble/judgment split (adapted from EkStep legal_ner.py) ────────────────
# The preamble is the header block before the judgment body begins.
# Keyword markers that typically end preambles in Indian judgments:
_PREAMBLE_END_KEYWORDS = [
    re.compile(r'\bJ\s*U\s*D\s*G\s*M\s*E\s*N\s*T\b', re.IGNORECASE),
    re.compile(r'\bO\s*R\s*D\s*E\s*R\b', re.IGNORECASE),
    re.compile(r'\bC\s*O\s*M\s*M\s*O\s*N\s+O\s*R\s*D\s*E\s*R\b', re.IGNORECASE),
    re.compile(r'\b(per\s+)?[A-Z][a-z]+,?\s*J\.?\s*:', re.IGNORECASE),
    re.compile(r'\bHEAD\s*NOTE\b', re.IGNORECASE),
]

def _split_preamble_judgment(text: str) -> Tuple[str, str]:
    """
    Split judgment into preamble (header) and body.

    The preamble contains party names, judge names, case numbers, advocates.
    The judgment body contains statutes, provisions, precedents, dates.

    Returns (preamble_text, judgment_text).
    """
    lines = text.split('\n')
    split_line = None

    for i, line in enumerate(lines):
        for kw in _PREAMBLE_END_KEYWORDS:
            if kw.search(line):
                split_line = i
                break
        if split_line is not None:
            break

    if split_line is None or split_line == 0:
        # Fallback: use first 30% as preamble
        split_line = max(1, len(lines) // 3)

    preamble = '\n'.join(lines[:split_line])
    judgment = '\n'.join(lines[split_line:])
    return preamble, judgment


def load_ner_models() -> Tuple:
    """
    Load the legal NER spaCy model and preamble-splitting model.

    Returns
    -------
    Tuple[spacy.Language, spacy.Language]
        (legal_nlp, preamble_nlp)

    Raises
    ------
    OSError
        If models are not installed. Print install instructions.
    """
    install_msg = (
        "\n\nPlease install the NER models:\n"
        "  pip install https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl\n"
        "  pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.2.0/en_core_web_sm-3.2.0-py3-none-any.whl\n"
    )
    try:
        legal_nlp = spacy.load('en_legal_ner_trf')
    except OSError:
        raise OSError(f"Model 'en_legal_ner_trf' not found.{install_msg}")

    try:
        preamble_nlp = spacy.load('en_core_web_sm')
    except OSError:
        raise OSError(f"Model 'en_core_web_sm' not found.{install_msg}")

    return legal_nlp, preamble_nlp


def _get_sentence_doc(judgment_doc, legal_nlp) -> spacy.tokens.Doc:
    """
    Run NER sentence-by-sentence on the judgment body.
    More accurate than doc-level; adapated from EkStep get_sentence_docs().
    """
    all_ents = []
    sent_docs = []

    for sent in judgment_doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        sent_doc = legal_nlp(sent_text)
        sent_docs.append(sent_doc)

    if not sent_docs:
        return legal_nlp('')

    combined = spacy.tokens.Doc.from_docs(sent_docs)
    return combined


def run_ner(
    text: str,
    legal_nlp,
    preamble_nlp,
    run_type: str = 'sent',
    do_postprocess: bool = True,
) -> List[Dict]:
    """
    Run Named Entity Recognition on a full judgment text.

    Steps:
    1. Split into preamble and judgment body
    2. Run legal NER on preamble (gets PETITIONER, RESPONDENT, JUDGE, COURT)
    3. Run legal NER on judgment body (gets STATUTE, PROVISION, PRECEDENT, etc.)
    4. Merge preamble + body docs
    5. Apply postprocessing (statute normalization, precedent clustering)
    6. Supplement with regex entities (CITATION, AMOUNT, LEGAL_TERM)

    Parameters
    ----------
    text : str
        Preprocessed full judgment text.
    legal_nlp : spacy.Language
        Loaded en_legal_ner_trf model.
    preamble_nlp : spacy.Language
        Loaded en_core_web_sm model (for sentence splitting).
    run_type : str
        'sent' for sentence-by-sentence (accurate) or 'doc' for whole-doc (fast).
    do_postprocess : bool
        Whether to apply statute normalization and precedent clustering.

    Returns
    -------
    List[Dict]
        Each dict: {entity_type, entity_text, start_char, end_char, confidence}
    """
    # 1. Split preamble / judgment body
    preamble_text, judgment_text = _split_preamble_judgment(text)

    # Clean mid-sentence newlines from judgment body (EkStep pattern)
    judgment_text = re.sub(r'(\w[ -]*)(\n+)', r'\1 ', judgment_text)

    # 2. Process preamble
    doc_preamble = legal_nlp(preamble_text)

    # 3. Process judgment body
    if run_type == 'sent':
        judgment_doc = preamble_nlp(judgment_text)
        doc_judgment = _get_sentence_doc(judgment_doc, legal_nlp)
    else:
        doc_judgment = legal_nlp(judgment_text)

    # 4. Combine preamble and judgment docs
    try:
        combined_doc = spacy.tokens.Doc.from_docs([doc_preamble, doc_judgment])
    except Exception:
        # Fallback: just use judgment entities + preamble entities separately
        combined_doc = doc_judgment

    # 5. Convert spaCy entities → dicts, applying label remapping
    preamble_offset = 0
    judgment_offset = len(preamble_text) + 1  # +1 for the joining newline

    entities: List[Dict] = []

    for ent in doc_preamble.ents:
        label = LABEL_MAP.get(ent.label_, ent.label_)
        entities.append({
            'entity_type': label,
            'entity_text': ent.text.strip(),
            'start_char':  ent.start_char,
            'end_char':    ent.end_char,
            'confidence':  round(getattr(ent._, 'score', 1.0), 4),
        })

    for ent in doc_judgment.ents:
        label = LABEL_MAP.get(ent.label_, ent.label_)
        entity_text = ent.text.strip()

        # Normalize statute names
        if label == 'STATUTE' and do_postprocess:
            entity_text = _normalize_statute(entity_text)

        entities.append({
            'entity_type': label,
            'entity_text': entity_text,
            'start_char':  ent.start_char + judgment_offset,
            'end_char':    ent.end_char + judgment_offset,
            'confidence':  round(getattr(ent._, 'score', 1.0), 4),
        })

    # 6. Regex supplement
    regex_entities = _extract_regex_entities(text)
    entities.extend(regex_entities)

    return entities


def _levenshtein(s1: str, s2: str) -> int:
    """Simple Levenshtein distance (adapted from EkStep calculate_lev)."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _cluster_precedents(precedents: List[str], threshold: int = 10) -> List[str]:
    """
    Deduplicate precedents using Levenshtein distance.
    Adapted from EkStep create_precedent_clusters().
    Returns the longest representative from each cluster.
    """
    if not precedents:
        return []

    sorted_precs = sorted(precedents, key=len, reverse=True)
    used = [False] * len(sorted_precs)
    clusters = []

    for i, p in enumerate(sorted_precs):
        if used[i]:
            continue
        cluster = [p]
        for j in range(i + 1, len(sorted_precs)):
            if used[j]:
                continue
            dist = _levenshtein(p.lower(), sorted_precs[j].lower())
            if dist <= threshold:
                used[j] = True
                cluster.append(sorted_precs[j])
        clusters.append(max(cluster, key=len))  # keep the longest form

    return clusters


def aggregate_entities(entity_list: List[Dict]) -> Dict[str, List[str]]:
    """
    Group extracted entities by type, deduplicate, and sort by frequency.

    Special handling:
    - STATUTE: acronym normalization applied
    - PRECEDENT: Levenshtein clustering to resolve supra/short-form references
    - All types: case-insensitive deduplication, keeping original casing of
      the most-frequent variant

    Parameters
    ----------
    entity_list : List[Dict]
        Output of run_ner().

    Returns
    -------
    Dict[str, List[str]]
        Keys are entity type labels. Values are deduplicated, frequency-sorted
        lists. Empty list (never None) for types with no entities found.
    """
    # Initialize all canonical types with empty lists
    result: Dict[str, List[str]] = {label: [] for label in CANONICAL_LABELS}

    # Group by type
    grouped: Dict[str, Dict[str, int]] = {}  # type -> {text_lower: count}
    casing: Dict[str, Dict[str, str]] = {}   # type -> {text_lower: best_casing}

    for ent in entity_list:
        etype = ent.get('entity_type', '')
        etext = ent.get('entity_text', '').strip()

        if not etext or etype not in CANONICAL_LABELS:
            continue
            
        # Filter blacklist
        if etext.lower() in BLACKLIST_ENTITIES:
            continue

        key = etext.lower()
        if etype not in grouped:
            grouped[etype] = {}
            casing[etype] = {}

        grouped[etype][key] = grouped[etype].get(key, 0) + 1

        # Keep the longest casing variant (more likely to be complete)
        if key not in casing[etype] or len(etext) > len(casing[etype][key]):
            casing[etype][key] = etext

    # Build initial output
    for etype, counts in grouped.items():
        sorted_keys = sorted(counts, key=lambda k: counts[k], reverse=True)
        texts = [casing[etype][k] for k in sorted_keys]
        result[etype] = texts

    # Post-aggregation cleanup: Merging PRECEDENT with nearby CITATION or other PRECEDENT
    if result.get('PRECEDENT'):
        # Sort all relevant spans by character position
        relevant_spans = [e for e in entity_list if e['entity_type'] in ('PRECEDENT', 'CITATION')]
        relevant_spans.sort(key=lambda x: x['start_char'])
        
        merged_precs = []
        skip_indices = set()
        
        for i in range(len(relevant_spans)):
            if i in skip_indices:
                continue
                
            span = relevant_spans[i]
            if span['entity_type'] != 'PRECEDENT':
                continue
                
            current_text = span['entity_text']
            current_end = span['end_char']
            
            # Look ahead for nearby citations or precedents to merge
            # User requirement: link names like "Veepul Lakhanpal" to citations like "2016 SCC..."
            j = i + 1
            while j < len(relevant_spans):
                next_span = relevant_spans[j]
                # Increase gap to 60 to handle OCR noise or mid-sentence fragments
                if 0 <= (next_span['start_char'] - current_end) <= 60:
                    current_text += ", " if not current_text.lower().endswith("v/s") and not current_text.endswith(",") else " "
                    current_text += next_span['entity_text']
                    current_end = next_span['end_char']
                    skip_indices.add(j)
                    j += 1
                else:
                    break
            
            merged_precs.append(current_text)
            
        if merged_precs:
            final_precs = []
            seen = set()
            for p in merged_precs:
                # Cleanup formatting
                p = re.sub(r',\s*,', ',', p)
                p = p.strip(', ')
                if p.lower() not in seen:
                    final_precs.append(p)
                    seen.add(p.lower())
            result['PRECEDENT'] = final_precs

    # Cleanup: Party Name Cleaning and Jurisdictional Filtering
    jurisdiction = _detect_jurisdiction(result, " ".join(result.get('PETITIONER', []) + result.get('RESPONDENT', [])))
    
    if jurisdiction == 'CIVIL':
        # Remove criminal terms from civil cases
        result['LEGAL_TERM'] = [t for t in result.get('LEGAL_TERM', []) if t not in _CRIMINAL_ONLY_TERMS]
        result['STATUTE'] = [s for s in result.get('STATUTE', []) if not any(k in s.upper() for k in ["IPC", "CRPC", "NDPS"])]

    # Clean Petitioner/Respondent names
    for etype in ['PETITIONER', 'RESPONDENT']:
        if result.get(etype):
            cleaned_names = []
            for name in result[etype]:
                c_name = _clean_party_name(name)
                if c_name and len(c_name) > 2: # Ignore single initials or artifacts
                    cleaned_names.append(c_name)
            result[etype] = list(dict.fromkeys(cleaned_names)) # Maintain order, remove dups

    # Cleanup: Remove current case from PRECEDENT
    if result.get('PETITIONER') and result.get('RESPONDENT'):
        pet1 = result['PETITIONER'][0].lower()
        res1 = result['RESPONDENT'][0].lower()
        clean_precedents = []
        for prec in result.get('PRECEDENT', []):
            p_lower = prec.lower()
            if pet1 in p_lower and res1 in p_lower:
                continue # Skip, this is likely the current case header being misclassified
            clean_precedents.append(prec)
        result['PRECEDENT'] = clean_precedents

    return result
