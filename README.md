# VeriSum-Legal

**Indian Court Judgment Analyzer** — NER extraction, abstractive summarization, and hallucination verification.

---

## Quick Start

### 1. Activate the virtual environment
```powershell
cd "c:\Users\Sujay\New folder\verisum_legal"
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install NER model wheels (EkStep/OpenNyAI — ~500MB)
pip install https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.2.0/en_core_web_sm-3.2.0-py3-none-any.whl
```

### 3. Add your PDFs
Put your Indian court judgment PDFs in:
```
data/sample_judgments/
```

### 4. Run the pipeline
```powershell
# Analyze a single PDF
python run_pipeline.py --pdf data/sample_judgments/judgment1.pdf

# Analyze all PDFs in sample_judgments/
python run_pipeline.py --all

# NER only (no summarization — fast)
python run_pipeline.py --pdf data/sample_judgments/judgment1.pdf --ner-only

# Faster NER (doc-level instead of sentence-level)
python run_pipeline.py --pdf data/sample_judgments/judgment1.pdf --fast
```

Results are saved to `output/<filename>_result.json`.

---

## Project Structure
```
verisum_legal/
├── run_pipeline.py          ← CLI entry point
├── requirements.txt
├── README.md
│
├── pipeline/
│   ├── chunker.py           ← PDF extraction + sliding window
│   ├── ner_infer.py         ← Legal NER (en_legal_ner_trf + regex)
│   ├── summarizer_infer.py  ← BART summarization + entity injection
│   └── verifier.py          ← Hallucination detection
│
├── training/                ← Offline fine-tuning (not required for inference)
│   ├── train_ner.py
│   ├── train_summarizer.py
│   └── evaluate.py
│
├── data/
│   └── sample_judgments/    ← PUT YOUR PDFs HERE
│
└── output/                  ← JSON results saved here
```

---

## NER Entity Types

| Entity | Description |
|---|---|
| `PETITIONER` | Party filing the case |
| `RESPONDENT` | Opposing party |
| `JUDGE` | Presiding judge(s) |
| `COURT` | Name of the court |
| `STATUTE` | Legislation name |
| `PROVISION` | Section/Article/Rule |
| `CASE_NUMBER` | Case reference number |
| `DATE` | Legally significant dates |
| `PRECEDENT` | Cited prior cases |
| `CITATION` | Law report citations (regex) |
| `AMOUNT` | Monetary values (regex) |
| `LEGAL_TERM` | Domain-specific phrases (regex) |
| `ORGANIZATION` | Institutions/companies |
| `LOCATION` | Geographic references |

---

## GPU Notes (RTX 3050)
- BART-base loads in **4-bit quantized mode** by default (fastest, ~560MB VRAM)
- NER model (en_legal_ner_trf) runs on **CPU** — spaCy transformer inference
- Expected speed: NER ~30-60s, Summarization ~5-15s on RTX 3050

---

## Output Format
```json
{
  "file": "judgment1.pdf",
  "entities": {
    "PETITIONER": ["Ministry of Health & Family Welfare"],
    "RESPONDENT": ["Nagarjuna Construction Ltd."],
    "JUDGE": ["Justice Jasmeet Singh"],
    "COURT": ["High Court of Delhi"],
    "STATUTE": ["Arbitration and Conciliation Act, 1996"],
    "PROVISION": ["Section 34", "Section 37"],
    "AMOUNT": ["Rs. 147,89,73,233/-"],
    "DATE": ["09.04.2026", "08.05.2017"]
  },
  "summary": "The Delhi High Court dismissed...",
  "verification": {
    "overall_status": "VERIFIED",
    "hallucinated_provisions": [],
    "missing_entities": [],
    "verified_provisions": ["section 34"],
    "flags": []
  },
  "timing": {
    "extraction_sec": 1.2,
    "ner_sec": 45.3,
    "summarization_sec": 8.7,
    "total_sec": 56.4
  }
}
```
