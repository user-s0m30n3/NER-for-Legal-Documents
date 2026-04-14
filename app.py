import os
import shutil
from pathlib import Path
import streamlit as st

# Must be the very first Streamlit command
st.set_page_config(
    page_title="VeriSum-Legal | AI Judgment Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
# Ensure imports work regardless of execution directory
sys.path.insert(0, str(Path(__file__).parent))

# Caching for Models
@st.cache_resource(show_spinner="Loading NER models into memory (~500MB)...")
def get_ner_models():
    from pipeline.ner_infer import load_ner_models
    return load_ner_models()

@st.cache_resource(show_spinner="Loading Quantized BART Summarizer (~1.6GB)...")
def get_summarizer_models():
    from pipeline.summarizer_infer import load_summarizer
    return load_summarizer(quantize=True)

# ── Custom CSS for Premium Look ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
        font-family: 'Inter', sans-serif;
    }
    h1 {
        color: #1a202c;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .subtitle {
        color: #718096;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .entity-card {
        background-color: #ffffff;
        border: 1px solid #eef2f7;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        height: 100%;
    }
    .entity-title {
        color: #4a5568;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .entity-tag {
        background-color: #edf2f7;
        color: #2d3748;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid #e2e8f0;
        line-height: 1.4;
    }
    .summary-box {
        background-color: #ffffff;
        border-left: 6px solid #3182ce;
        padding: 30px;
        border-radius: 0 16px 16px 0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
        font-size: 1.2rem;
        line-height: 1.8;
        color: #1a202c;
        margin-bottom: 2.5rem;
    }
    .sidebar-help {
        font-size: 0.85rem;
        color: #4a5568;
        background-color: #f7fafc;
        padding: 12px;
        border-radius: 8px;
        margin-top: 6px;
        border: 1px solid #edf2f7;
    }
</style>
""", unsafe_allow_html=True)

# ── Metadata ───────────────────────────────────────────────────────────────
ICON_MAP = {
    'PETITIONER': '👤',
    'RESPONDENT': '👥',
    'JUDGE': '⚖️',
    'COURT': '🏛️',
    'CASE_NUMBER': '📑',
    'CITATION': '🔗',
    'DATE': '📅',
    'STATUTE': '📖',
    'PROVISION': '📍',
    'PRECEDENT': '📚',
    'LEGAL_TERM': '🛠️',
    'AMOUNT': '💰',
    'LOCATION': '📍',
    'ORGANIZATION': '🏢'
}

TITLE_MAP = {
    'PETITIONER': 'Petitioner / Appellant',
    'RESPONDENT': 'Respondent',
    'JUDGE': 'Presiding Judge',
    'COURT': 'Court / Jurisdiction',
    'CASE_NUMBER': 'Case Number',
    'CITATION': 'Legal Citation',
    'DATE': 'Key Dates',
    'STATUTE': 'Statutes Referenced',
    'PROVISION': 'Specific Provisions',
    'PRECEDENT': 'Judicial Precedents',
    'LEGAL_TERM': 'Legal Concepts',
    'AMOUNT': 'Monetary Amounts',
    'LOCATION': 'Locations',
    'ORGANIZATION': 'Organizations'
}

# ── File Handling ────────────────────────────────────────────────────────────
TEMP_DIR = Path("data/temp_uploads")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def clear_temp_dir():
    for item in TEMP_DIR.iterdir():
        if item.is_file(): item.unlink()

# ── Main UI App ───────────────────────────────────────────────────────────────
def main():
    st.markdown("<h1>⚖️ VeriSum-Legal</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Premium Legal Intelligence & Abstractive Summary Engine</p>', unsafe_allow_html=True)
    
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Document Ingestion")
        uploaded_file = st.file_uploader("Upload PDF judgment", type=["pdf"])
        
        st.write("---")
        st.markdown("### ⚙️ Analysis Parameters")
        
        fast_ner = st.checkbox("Fast Preview Mode", value=False)
        st.markdown('<div class="sidebar-help"><b>Fast Mode:</b> Optimized for rapid indexing. Skips intensive sentence-level cross-referencing.</div>', unsafe_allow_html=True)
        
        ner_only = st.checkbox("Extraction Only", value=False)
        st.markdown('<div class="sidebar-help"><b>Extraction Only:</b> Retrieves entities but skips LLM-based abstractive summarization.</div>', unsafe_allow_html=True)
        
        st.write("---")
        if st.button("Purge System Temp"):
            clear_temp_dir()
            st.rerun()

    # ── Execution ────────────────────────────────────────────────────────────
    if uploaded_file is not None:
        from run_pipeline import run_single
        
        # Load models
        legal_nlp, preamble_nlp = get_ner_models()
        tok, mod, dev = (None, None, None)
        if not ner_only:
             tok, mod, dev = get_summarizer_models()

        st.markdown("---")
        if st.button("🚀 Analyze Judgment", use_container_width=True, type="primary"):
            clear_temp_dir()
            temp_path = TEMP_DIR / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Executing Intelligent Legal Extraction..."):
                try:
                    result = run_single(
                        pdf_path=str(temp_path),
                        legal_nlp=legal_nlp,
                        preamble_nlp=preamble_nlp,
                        bart_tokenizer=tok,
                        bart_model=mod,
                        bart_device=dev,
                        ner_only=ner_only,
                        fast=fast_ner
                    )
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")
                    st.stop()
            
            # --- Results Rendering ---
            summary = result.get('summary')
            entities = result.get('entities', {})
            verif = result.get('verification', {})

            # 1. Verification Status
            if verif:
                status = verif.get('overall_status', 'UNKNOWN')
                if status == 'VERIFIED':
                    st.success("✅ **Intelligence Verified:** Extracted legal logic matches source document with 100% ground-truth consistency.")
                else:
                    st.warning(f"💡 **Analysis Note:** {', '.join(verif.get('flags', []))}")

            # 2. Summary Section
            if summary:
                st.subheader("Abstractive Legal Summary")
                st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
            
            # 3. Dynamic Entity Grid
            st.subheader("Judgment Intelligence Index")
            
            # Filter out empty categories
            active_entities = {k: v for k, v in entities.items() if v}
            
            # Order categories logically (Parties -> Court -> Law -> Others)
            preferred_order = [
                'PETITIONER', 'RESPONDENT', 'JUDGE', 
                'COURT', 'CASE_NUMBER', 'CITATION', 'DATE', 
                'STATUTE', 'PROVISION', 'PRECEDENT', 'LEGAL_TERM',
                'AMOUNT', 'LOCATION', 'ORGANIZATION'
            ]
            
            sorted_categories = [cat for cat in preferred_order if cat in active_entities]
            
            # Render in rows of 3
            for i in range(0, len(sorted_categories), 3):
                batch = sorted_categories[i:i+3]
                cols = st.columns(3)
                for j, category in enumerate(batch):
                    with cols[j]:
                        icon = ICON_MAP.get(category, '•')
                        title = TITLE_MAP.get(category, category)
                        items = active_entities[category]
                        
                        # Generate Tag Cloud HTML
                        tag_html = "".join([f'<div class="entity-tag">{item}</div>' for item in items])
                        
                        st.markdown(f"""
                        <div class="entity-card">
                            <div class="entity-title"><span>{icon}</span> {title}</div>
                            <div class="tag-container">
                                {tag_html}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            with st.expander("Technical Audit: View Raw Data Structure"):
                st.json(result)

    else:
        # Landing Page
        st.info("👈 **Awaiting Document:** Upload a judicial PDF from the sidebar to begin automated extraction and abstraction.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            ### System Capabilities
            - **Entity Normalization:** Automatically resolves acronyms like *IPC*, *CrPC*, and *CPC*.
            - **Case Linker:** Merges precedent names with their respective citations.
            - **Hallucination Guard:** Deterministically verifies generated summaries against source text.
            """)
        with c2:
            st.markdown("""
            ### Privacy & Security
            - **Local Inference:** All processing occurs within this environment.
            - **Memory Resident:** Large models are cached for speed after the initial load.
            - **No External Callouts:** No data is sent to external API providers.
            """)

if __name__ == "__main__":
    main()
