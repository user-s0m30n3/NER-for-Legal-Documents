# Data Directory

## sample_judgments/
Put your Indian court judgment PDFs here.

Tested judgment set:
1. Ministry of Health vs Nagarjuna Construction (Delhi HC, 09-Apr-2026)
2. Roma Ahuja vs The State (Supreme Court, 09-Apr-2026)
3. Shivnanda Karadbhaje vs Satyawan Karadbhaje (Bombay HC, 10-Apr-2026)
4. Unknown vs State of Tripura (Tripura HC, 10-Apr-2026)
5. Veer Chand vs State of HP (HP HC, 09-Apr-2026)

## Training Data (for offline fine-tuning only)

### EkStep Legal NER Dataset
- Train: https://huggingface.co/datasets/opennyaiorg/InLegalNER/resolve/main/NER_TRAIN.zip
- Dev:   https://huggingface.co/datasets/opennyaiorg/InLegalNER/resolve/main/NER_DEV.zip
- Test:  https://huggingface.co/datasets/opennyaiorg/InLegalNER/resolve/main/NER_TEST.zip
- Or simply: `load_dataset("opennyaiorg/InLegalNER")` in Python

### IN-Abs Dataset (Indian SC Abstractive Summaries)
- https://github.com/Law-AI/summarization
- Download and place as data/IN_Abs_train.csv, data/IN_Abs_dev.csv
