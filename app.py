"""
app.py — Adverse Keyword Search v5
Full pipeline explainability, themed UI, writeups per step
"""
import os, json, tempfile, re
import pandas as pd
import streamlit as st
from pathlib import Path

from pipeline import AnalysisPipeline, load_keywords
from config.settings import settings
from modules.audit_trail import load_audit_records
from modules.documents_db import load_all_documents, get_db_stats

st.set_page_config(
    page_title="Adverse Keyword Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme & CSS ───────────────────────────────────────────────────────────────
ACCENT    = "#2563eb"   # blue
ACCENT_L  = "#eff6ff"
SUCCESS   = "#16a34a"
SUCCESS_L = "#f0fdf4"
WARN      = "#d97706"
WARN_L    = "#fffbeb"
DANGER    = "#dc2626"
DANGER_L  = "#fef2f2"
NEUTRAL   = "#475569"
BORDER    = "#e2e8f0"

st.markdown(f"""
<style>
/* ── Global font ── */
html, body, [class*="css"] {{ font-family: 'Segoe UI', system-ui, sans-serif; }}

/* ── Sidebar background ── */
section[data-testid="stSidebar"] > div {{
    background: linear-gradient(180deg, #1e3a5f 0%, #1e2d4a 100%);
    padding-top: 1rem;
}}

/* ── All sidebar text → light ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] strong,
section[data-testid="stSidebar"] b,
section[data-testid="stSidebar"] em,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown *,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stRadio span,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSlider span,
section[data-testid="stSidebar"] .stSlider p,
section[data-testid="stSidebar"] .stToggle label,
section[data-testid="stSidebar"] .stToggle span,
section[data-testid="stSidebar"] .stFileUploader label,
section[data-testid="stSidebar"] .stFileUploader span,
section[data-testid="stSidebar"] .stFileUploader p,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stCaption * {{
    color: #e2e8f0 !important;
}}

/* ── Sidebar section bold headers brighter ── */
section[data-testid="stSidebar"] strong,
section[data-testid="stSidebar"] b,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: #f8fafc !important;
}}

/* ── Sidebar divider ── */
section[data-testid="stSidebar"] hr {{
    border-color: #334d6e !important;
}}

/* ── Sidebar success/info/warning boxes ── */
section[data-testid="stSidebar"] .stAlert {{
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.2) !important;
}}
section[data-testid="stSidebar"] .stAlert * {{
    color: #e2e8f0 !important;
}}

/* ── App header ── */
.app-header {{
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 20px;
    color: white;
}}
.app-header h1 {{ color: white !important; margin: 0; font-size: 1.9rem; }}
.app-header p  {{ color: #bfdbfe; margin: 4px 0 0 0; font-size: 0.95rem; }}

/* ── Pipeline step cards ── */
.step-explain {{
    background: {ACCENT_L};
    border-left: 4px solid {ACCENT};
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0 12px 0;
    font-size: 0.88rem;
    color: #1e40af;
    line-height: 1.6;
}}
.step-explain b {{ color: #1d4ed8; }}

.how-it-works {{
    background: #fafafa;
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.83rem;
    color: {NEUTRAL};
    line-height: 1.65;
}}

/* ── Match cards ── */
.kw-pill {{
    display: inline-block;
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-radius: 4px;
    padding: 2px 8px;
    font-weight: 700;
    color: #92400e;
    font-size: 0.88rem;
}}
.badge-neg {{
    display: inline-block;
    background: #d1fae5;
    border: 1px solid #10b981;
    color: #065f46;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.75rem;
    font-weight: 600;
}}
.badge-aff {{
    display: inline-block;
    background: #fee2e2;
    border: 1px solid #ef4444;
    color: #991b1b;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.75rem;
    font-weight: 600;
}}

/* ── Risk banners ── */
.risk-high   {{ background:{DANGER_L}; border:1px solid #fca5a5; border-radius:10px; padding:16px 20px; }}
.risk-medium {{ background:{WARN_L};   border:1px solid #fcd34d; border-radius:10px; padding:16px 20px; }}
.risk-low    {{ background:{SUCCESS_L};border:1px solid #86efac; border-radius:10px; padding:16px 20px; }}

/* ── Stat chips ── */
.chip {{
    display: inline-block;
    background: {ACCENT_L};
    color: {ACCENT};
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 2px;
    border: 1px solid #bfdbfe;
}}

/* ── Section headers ── */
.sec-hdr {{
    font-size: 1rem;
    font-weight: 700;
    color: #1e3a5f;
    border-bottom: 2px solid {ACCENT};
    padding-bottom: 4px;
    margin: 16px 0 10px 0;
    display: inline-block;
}}

/* ── Info callout ── */
.callout-info {{
    background: {ACCENT_L};
    border: 1px solid #93c5fd;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #1e40af;
    margin: 6px 0;
}}
.callout-warn {{
    background: {WARN_L};
    border: 1px solid #fcd34d;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #92400e;
    margin: 6px 0;
}}
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
CONF_ICON  = {"High":"🔴","Medium":"🟡","Low":"🟠","Informational":"🔵"}
STEP_ICON  = {"ok":"✅","warning":"⚠️","error":"❌"}
RISK_ICON  = {"High":"🔴","Medium":"🟡","Low":"🟢"}
RISK_CLS   = {"High":"risk-high","Medium":"risk-medium","Low":"risk-low"}
RISK_COLOR = {"High":"#991b1b","Medium":"#92400e","Low":"#166534"}

# Pipeline step explanations shown to the user
STEP_EXPLAIN = {
    1: {
        "title": "Document Ingestion",
        "what": "The system reads and parses the uploaded document into structured text chunks. Each chunk retains positional metadata — page number, section heading, and source type — so we know exactly where in the document every match was found.",
        "how": "Different parsers handle different formats: pdfplumber for PDFs (preserving page boundaries and table structures), python-docx for Word files (preserving heading hierarchy), openpyxl for Excel (sheet by sheet), and BeautifulSoup for HTML/ACORD forms. Email .eml/.msg files are decoded and the body extracted.",
        "why": "Chunking by paragraph rather than treating the whole document as one string is critical — it lets us attach precise location metadata to every match and detect which section a match belongs to."
    },
    2: {
        "title": "Exact Keyword Matching — Aho-Corasick",
        "what": "The system scans every document chunk against the entire keyword dictionary in a single linear pass. Every exact or near-exact match is identified with its position and the category it belongs to.",
        "how": "The Aho-Corasick algorithm builds a finite automaton from all keywords at startup. It then processes each chunk in O(n) time — meaning the speed doesn't slow down as the dictionary grows. Matching runs on normalized text (lowercased, abbreviations expanded) to handle minor variations without fuzzy logic.",
        "why": "Exact matching is the backbone because it's fully deterministic and auditable. An underwriter can always point to exactly which phrase triggered a flag. All exact matches are assigned High confidence — no threshold to tune."
    },
    3: {
        "title": "Semantic Similarity Matching — ChromaDB + Embeddings",
        "what": "Beyond exact keywords, the system finds document passages that are semantically similar to risk categories — catching synonyms, paraphrases, and domain-adjacent language that exact matching would miss.",
        "how": "Every keyword is converted into a numerical vector (embedding) using the all-MiniLM-L6-v2 sentence transformer model and stored in a ChromaDB vector database. Document chunks are embedded the same way, then cosine similarity is computed between each chunk and every keyword vector. Chunks exceeding the similarity threshold are flagged.",
        "why": "A phrase like 'history of environmental contamination claims' won't match the keyword 'prior litigation' exactly, but its embedding will be geometrically close. This catches the coverage gaps that fuzzy string matching was attempting to solve — but with meaningful similarity rather than character distance."
    },
    4: {
        "title": "Negation Detection",
        "what": "Each match is checked for negating language in its surrounding context. A match is tagged Affirmed (the adverse condition is stated) or Negated (the adverse condition is explicitly denied).",
        "how": "A configurable token window (default: 6 tokens) before each keyword match is scanned for negation indicators: 'no', 'not', 'without', 'never', 'no history of', 'no prior', 'absence of', 'denies any'. Additional phrase-level regex patterns catch constructions like 'no known history of X' even when the negation word is not immediately adjacent.",
        "why": "This is the single most impactful quality improvement over fuzzy matching. 'No prior litigation' and 'significant prior litigation' are nearly identical strings — but opposite in risk meaning. Without negation detection, both would be flagged identically."
    },
    5: {
        "title": "Section Weighting",
        "what": "Match confidence is adjusted based on which section of the document the match was found in. A keyword in the Risk Description carries much more weight than the same keyword in standard boilerplate.",
        "how": "Each document section is assigned a weight from 0.1 to 1.0. Risk Description, Loss History, and Property Details sections carry weight ≥ 0.9 and can promote a Medium match to High. General Conditions, Exclusions, and Signature sections carry weight ≤ 0.4 and demote any match to Informational.",
        "why": "Standard policy documents contain adverse terms in exclusion clauses by design — 'this policy does not cover asbestos remediation' appears in virtually every submission. Without section weighting, these create noise that buries the genuine risk signals."
    },
    6: {
        "title": "Co-occurrence Risk Scoring",
        "what": "The system looks beyond individual keywords to detect patterns — specifically, whether high-risk categories appear together within the same document section. Certain category combinations are historically correlated with declined submissions.",
        "how": "A co-occurrence matrix is built across all affirmed matches within a configurable proximity window (default: 3 paragraphs). Known high-risk category pairs — such as Environmental Liability + Prior Litigation, or Financial Distress + Structural Defects — carry elevated combined weights. These weights feed into a composite document-level risk score.",
        "why": "A single adverse keyword mention is a data point. The co-occurrence of multiple risk categories in the same section is a pattern — and patterns are what experienced underwriters actually act on. This moves the system from simple detection to risk assessment."
    },
    7: {
        "title": "LLM Context Interpretation",
        "what": "For Medium and Low confidence matches (from the semantic layer), a language model reads the surrounding paragraph and provides a yes/no determination with a one-sentence rationale explaining why the text does or does not indicate the adverse risk category.",
        "how": "The document chunk and matched category are sent to the configured LLM (Groq llama-3.3-70b or OpenAI gpt-4o-mini) with a structured prompt. The model returns JSON with three fields: confirmed (bool), rationale (string), confidence (tier). High confidence exact matches bypass this step entirely — they are already deterministic.",
        "why": "A semantic similarity score of 0.79 tells an underwriter nothing actionable. A sentence like 'Yes — the text describes a previous out-of-court settlement which constitutes prior litigation activity' is immediately actionable. The LLM is gated to ambiguous matches only to keep API costs proportional to the actual uncertainty in the document."
    },
    8: {
        "title": "Audit Trail & Document Storage",
        "what": "Every analysis run is permanently recorded — the document fingerprint, keyword dictionary version active at run time, all match results, LLM assessments, and the composite risk score.",
        "how": "Two append-only stores are written: the audit trail JSONL (for regulatory traceability) and the documents database JSONL (for the Documents DB tab). Both use content-addressable records keyed by run ID, and the dictionary version is hashed so you can always determine which keyword set was in use when a specific document was analyzed.",
        "why": "In insurance, underwriting decisions must be defensible. If a submission was analyzed and a risk category was not flagged, you need to be able to demonstrate whether that category was in scope at analysis time. Versioning the dictionary and recording it with every run makes this possible."
    }
}


def highlight_keyword_plain(text: str, keyword: str) -> str:
    if not keyword or not text:
        return text
    return re.sub(re.escape(keyword), f"**{keyword}**", text, flags=re.IGNORECASE)


def render_step_explainer(step_num: int):
    info = STEP_EXPLAIN.get(step_num, {})
    if not info:
        return

    # "What" block — always visible
    st.markdown(f"""
    <div class="step-explain">
        <b>What this step does:</b> {info['what']}
    </div>""", unsafe_allow_html=True)

    # "How & Why" — toggled via button (avoids nested expander error)
    toggle_key = f"step_detail_{step_num}"
    if toggle_key not in st.session_state:
        st.session_state[toggle_key] = False

    label = "▲ Hide details" if st.session_state[toggle_key] else "▼ How it works & why it matters"
    if st.button(label, key=f"btn_step_{step_num}",
                 use_container_width=False):
        st.session_state[toggle_key] = not st.session_state[toggle_key]

    if st.session_state[toggle_key]:
        st.markdown(f"""
        <div class="how-it-works">
            <b>⚙️ How:</b> {info['how']}<br><br>
            <b>💡 Why:</b> {info['why']}
        </div>""", unsafe_allow_html=True)


def render_match_card(m, idx):
    conf      = getattr(m, "confidence", "Low")
    icon      = CONF_ICON.get(conf, "")
    kw        = getattr(m, "keyword", getattr(m, "matched_keyword", ""))
    negated   = getattr(m, "negation_flag", False)
    mtype     = getattr(m, "match_type", "")
    page      = getattr(m, "page", None)
    section   = getattr(m, "section", "") or "—"
    score     = getattr(m, "similarity_score", None)
    llm_rat   = getattr(m, "llm_rationale", "")
    llm_conf  = getattr(m, "llm_confirmed", None)
    cat       = getattr(m, "category", "")
    ctx_raw   = getattr(m, "chunk_text", "")[:600]

    borders   = {"High":"#ef4444","Medium":"#f59e0b","Low":"#f97316","Informational":"#3b82f6"}
    bc        = borders.get(conf, "#94a3b8")
    page_str  = f"Page {page}" if page else "—"
    score_str = f" · sim {score}" if score else ""
    neg_label = "✓ Negated" if negated else "⚠ Affirmed"
    neg_cls   = "badge-neg" if negated else "badge-aff"

    st.markdown(f"""
    <div style="border:1px solid {BORDER}; border-left:5px solid {bc};
                border-radius:8px; padding:12px 16px 4px 16px; margin:8px 0 0 0;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px">
        <span style="font-size:0.98rem;font-weight:700;color:#0f172a">{icon} [{conf}] &nbsp; {cat}</span>
        <span style="font-size:0.76rem;color:#94a3b8">#{idx+1} &nbsp;·&nbsp; {page_str} &nbsp;·&nbsp; {section}{score_str}</span>
      </div>
      <div style="margin:6px 0 8px;font-size:0.85rem">
        Keyword: <span class="kw-pill">{kw}</span>
        &nbsp;<span class="{neg_cls}">{neg_label}</span>
        &nbsp;<code style="font-size:0.73rem;background:#f1f5f9;padding:1px 6px;border-radius:3px">{mtype}</code>
      </div>
    </div>""", unsafe_allow_html=True)

    ctx_md = highlight_keyword_plain(ctx_raw, kw)
    st.markdown(
        f"<div style='border:1px solid {BORDER}; border-top:none; border-radius:0 0 8px 8px;"
        f"padding:10px 16px 12px; margin-bottom:4px; font-size:0.83rem; line-height:1.75;"
        f"color:#334155'>{ctx_md}</div>",
        unsafe_allow_html=True
    )
    if llm_rat:
        llm_icon = "✅" if llm_conf else "❌"
        st.markdown(
            f"<div style='padding:6px 16px 10px; font-size:0.8rem; color:#64748b;"
            f"border-left:3px solid #94a3b8; margin:0 0 8px 0; background:#f8fafc; border-radius:0 0 6px 6px'>"
            f"🤖 <b>LLM ({('confirmed' if llm_conf else 'not confirmed')}):</b> {llm_icon} {llm_rat}</div>",
            unsafe_allow_html=True
        )


def get_pipeline(keywords_dict: dict, provider: str) -> "AnalysisPipeline":
    """
    Returns a pipeline instance, reinitializing when the dictionary or
    provider changes. Stored in session_state keyed by dict hash + provider
    so it rebuilds exactly when the inputs change — no stale cache issues.
    """
    import hashlib
    dict_hash = hashlib.md5(
        json.dumps(keywords_dict, sort_keys=True).encode()
    ).hexdigest()[:12]
    cache_key = f"pipeline_{dict_hash}_{provider}"

    if cache_key not in st.session_state:
        # Clear any old pipeline to free memory
        old_keys = [k for k in st.session_state if k.startswith("pipeline_")]
        for k in old_keys:
            del st.session_state[k]
        with st.spinner("Initializing pipeline — loading models..."):
            st.session_state[cache_key] = AnalysisPipeline(
                keywords_dict, llm_provider=provider
            )

    return st.session_state[cache_key]


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p style="color:#f8fafc;font-size:1.3rem;font-weight:700;margin:0">🔍 AKS Config</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;font-size:0.82rem;margin:0 0 8px 0">Adverse Keyword Search</p>', unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#334d6e;margin:8px 0"/>', unsafe_allow_html=True)

    st.markdown('<p style="color:#93c5fd;font-size:0.82rem;font-weight:600;letter-spacing:0.05em;margin:8px 0 6px 0">🤖 LLM PROVIDER</p>', unsafe_allow_html=True)
    llm_provider = st.radio("Provider", ["groq","openai"], horizontal=True,
                             label_visibility="collapsed")
    if llm_provider == "groq":
        k = st.text_input("Groq API Key", value=settings.groq_api_key, type="password")
        if k: settings.groq_api_key = k
        st.markdown(f'<p style="color:#7dd3fc;font-size:0.78rem;margin:2px 0 0 0">Model: <code style="background:#0f2744;color:#67e8f9;padding:1px 5px;border-radius:3px">{settings.groq_model}</code></p>', unsafe_allow_html=True)
    else:
        k = st.text_input("OpenAI API Key", value=settings.openai_api_key, type="password")
        if k: settings.openai_api_key = k
        st.markdown(f'<p style="color:#7dd3fc;font-size:0.78rem;margin:2px 0 0 0">Model: <code style="background:#0f2744;color:#67e8f9;padding:1px 5px;border-radius:3px">{settings.openai_model}</code></p>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#334d6e;margin:12px 0"/>', unsafe_allow_html=True)
    st.markdown('<p style="color:#93c5fd;font-size:0.82rem;font-weight:600;letter-spacing:0.05em;margin:0 0 6px 0">📖 KEYWORD DICTIONARY</p>', unsafe_allow_html=True)
    dict_upload = st.file_uploader("Upload JSON dictionary", type=["json"],
                                    label_visibility="collapsed")
    if dict_upload:
        # Store in session state so it persists across reruns
        loaded = json.loads(dict_upload.read())
        st.session_state["keywords_dict"] = loaded
        st.session_state["keywords_source"] = dict_upload.name

    if "keywords_dict" in st.session_state:
        keywords_dict = st.session_state["keywords_dict"]
        source = st.session_state.get("keywords_source", "uploaded")
        total_kw = sum(len(v) for v in keywords_dict.values())
        st.success(f"✅ **{source}** — {len(keywords_dict)} categories · {total_kw} keywords")
        if st.button("↺ Reset to default dictionary", use_container_width=True):
            st.session_state.pop("keywords_dict", None)
            st.session_state.pop("keywords_source", None)
            st.rerun()
    else:
        keywords_dict = load_keywords()
        total_kw = sum(len(v) for v in keywords_dict.values())
        st.info(f"📖 Default — {len(keywords_dict)} categories · {total_kw} keywords")

    with st.expander("Preview categories"):
        for cat, kws in keywords_dict.items():
            st.markdown(f"**{cat}** ({len(kws)})")
            st.caption(", ".join(kws[:4]) + ("..." if len(kws) > 4 else ""))

    st.markdown('<hr style="border-color:#334d6e;margin:12px 0"/>', unsafe_allow_html=True)
    st.markdown('<p style="color:#93c5fd;font-size:0.82rem;font-weight:600;letter-spacing:0.05em;margin:0 0 6px 0">⚙️ ANALYSIS SETTINGS</p>', unsafe_allow_html=True)
    run_llm = st.toggle("Enable LLM interpretation", value=True,
                         help="Sends medium/low confidence matches to Groq or OpenAI for context validation")
    sem_thresh = st.slider("Semantic similarity threshold", 0.50, 0.95,
                            settings.semantic_similarity_threshold, 0.01,
                            help="Minimum cosine similarity for semantic matches. Lower = more matches, more noise.")
    settings.semantic_similarity_threshold = sem_thresh
    thresh_label = "conservative — fewer, higher-quality hits" if sem_thresh >= 0.80 else "balanced" if sem_thresh >= 0.70 else "broad — more coverage, more review needed"
    st.markdown(f'<p style="color:#94a3b8;font-size:0.77rem;margin:2px 0 0 0">At {sem_thresh:.2f}: {thresh_label}</p>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#334d6e;margin:12px 0"/>', unsafe_allow_html=True)
    st.markdown('<p style="color:#93c5fd;font-size:0.82rem;font-weight:600;letter-spacing:0.05em;margin:0 0 6px 0">🗃️ SYSTEM STATUS</p>', unsafe_allow_html=True)

    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.vectordb_path)
        names  = [c.name for c in client.list_collections()]
        if "keyword_categories" in names:
            col = client.get_collection("keyword_categories")
            st.success(f"**ChromaDB** ✅\n\n{col.count()} keyword vectors stored")
        else:
            st.warning("ChromaDB empty")
    except Exception as e:
        st.error(f"ChromaDB: {e}")

    stats = get_db_stats()
    if stats["total"] > 0:
        st.success(f"**Documents DB** ✅\n\n{stats['total']} runs · 🔴{stats['high_risk']} 🟡{stats['medium_risk']} 🟢{stats['low_risk']}")
    else:
        st.info("**Documents DB** — empty")

    # Show which pipeline is currently loaded
    import hashlib
    active_hash = hashlib.md5(json.dumps(keywords_dict, sort_keys=True).encode()).hexdigest()[:12]
    pipeline_key = f"pipeline_{active_hash}_{llm_provider}"
    if pipeline_key in st.session_state:
        st.markdown(f'<p style="color:#4ade80;font-size:0.77rem;margin:4px 0 0 0">✅ Pipeline loaded · dict <code style="background:#0f2744;color:#67e8f9;padding:1px 4px;border-radius:2px">{active_hash}</code></p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color:#fbbf24;font-size:0.77rem;margin:4px 0 0 0">⏳ Pipeline not yet initialized · will load on first analysis</p>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
  <h1>🔍 Adverse Keyword Search</h1>
  <p>Insurance Submission Document Analysis · Multi-layer matching pipeline with explainability</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Analyze",
    "⚙️ Pipeline Steps",
    "📊 Results",
    "🗄️ Documents DB",
    "📋 Audit Trail",
])


# ════════════════════════════════════════════════════════════
# TAB 1 — Upload & Analyze
# ════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="sec-hdr">Upload Submission Document</div>', unsafe_allow_html=True)
        st.caption("Supported: PDF · DOCX · XLSX · EML · MSG · HTML (ACORD) · TXT")
        uploaded = st.file_uploader("Document", label_visibility="collapsed",
                                     type=["pdf","docx","xlsx","xls","msg","eml","html","htm","xml","txt"])

    with col_r:
        st.markdown('<div class="sec-hdr">What the pipeline checks</div>', unsafe_allow_html=True)
        for cat, kws in keywords_dict.items():
            st.markdown(f"**{cat}** — {len(kws)} keywords")

    # Pipeline overview explainer
    with st.expander("ℹ️ How the analysis pipeline works", expanded=False):
        st.markdown("""
        This system runs every uploaded document through **8 sequential steps**. Each step builds
        on the previous one, and together they produce results that are more accurate, explainable,
        and actionable than any single matching approach could achieve on its own.

        | Step | Name | Purpose |
        |------|------|---------|
        | 1 | Document Ingestion | Parse file into chunks with page/section metadata |
        | 2 | Exact Matching | Aho-Corasick keyword scan — deterministic, High confidence |
        | 3 | Semantic Matching | Embedding similarity — catches synonyms and paraphrases |
        | 4 | Negation Detection | Tag matches as affirmed vs denied ("no prior litigation") |
        | 5 | Section Weighting | Demote boilerplate matches, promote risk-section matches |
        | 6 | Co-occurrence Scoring | Detect high-risk category combinations |
        | 7 | LLM Interpretation | Natural-language explanation of ambiguous matches |
        | 8 | Audit & Storage | Permanent record of this run for traceability |

        Switch to the **Pipeline Steps** tab after analysis to see exactly what happened at each step.
        """)

    if uploaded:
        st.divider()
        c1, c2, c3 = st.columns([4, 1, 1])
        with c1: st.info(f"📄 **{uploaded.name}** — {round(uploaded.size/1024, 1)} KB")
        with c2: run_btn = st.button("▶ Analyze", type="primary", use_container_width=True)
        with c3:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.pop("result", None)
                st.rerun()

        if run_btn:
            file_bytes = uploaded.read()
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            # Live step progress
            pbar  = st.progress(0)
            ptext = st.empty()
            pstep = st.empty()

            STEP_SHORT = {
                1: "Parsing document into structured chunks...",
                2: "Running Aho-Corasick exact keyword scan...",
                3: "Computing semantic embeddings and querying ChromaDB...",
                4: "Scanning for negation patterns (no/not/never/without)...",
                5: "Adjusting confidence by document section...",
                6: "Computing cross-category co-occurrence risk score...",
                7: f"Sending ambiguous matches to {llm_provider.upper()} for context validation...",
                8: "Writing audit record and saving to documents database...",
            }

            def pcb(step, name):
                pbar.progress(step / 8)
                ptext.markdown(f"⏳ **Step {step}/8 — {name}**")
                pstep.markdown(f"<div class='callout-info'>{STEP_SHORT.get(step,'')}</div>",
                               unsafe_allow_html=True)

            with st.spinner(""):
                pipeline = get_pipeline(keywords_dict, llm_provider)
                result   = pipeline.run(tmp_path, run_llm=run_llm, progress_cb=pcb, file_bytes=file_bytes)

            pbar.progress(1.0)
            ptext.markdown("✅ **Analysis complete!**")
            pstep.empty()
            os.unlink(tmp_path)
            st.session_state["result"] = result

            # Risk banner
            cooc = result.cooccurrence
            risk = cooc.document_risk_score
            cls  = RISK_CLS.get(risk, "risk-low")
            col  = RISK_COLOR.get(risk, "#166534")
            ico  = RISK_ICON.get(risk, "")
            st.markdown(f"""
            <div class="{cls}">
              <div style="font-size:1.4rem;font-weight:800;color:{col}">{ico} {risk} Risk Document</div>
              <div style="color:{col}aa;margin-top:4px">{cooc.notes}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("")

            # Metrics
            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("Total Matches",   len(result.all_matches))
            m2.metric("🔴 High",         sum(1 for m in result.all_matches if m.confidence=="High"))
            m3.metric("🟡 Medium",       sum(1 for m in result.all_matches if m.confidence=="Medium"))
            m4.metric("🟠 Low",          sum(1 for m in result.all_matches if m.confidence=="Low"))
            m5.metric("✅ Negated",      sum(1 for m in result.all_matches if getattr(m,"negation_flag",False)))
            m6.metric("Run ID",          result.run_id)

            if cooc.triggered_categories:
                chips = " ".join([f'<span class="chip">{c}</span>' for c in cooc.triggered_categories])
                st.markdown(f"**Triggered categories:** {chips}", unsafe_allow_html=True)

            if cooc.high_risk_combos_found:
                for c1x,c2x,w in cooc.high_risk_combos_found:
                    st.warning(f"⚡ High-risk combination: **{c1x}** + **{c2x}** (combined weight {w})")

            if result.errors:
                with st.expander("⚠️ Warnings"):
                    for e in result.errors: st.warning(e)

            st.markdown("""<div class="callout-info">
            → Switch to <b>Pipeline Steps</b> to see what happened at each stage,
            or <b>Results</b> to explore the findings with full context paragraphs.
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — Pipeline Steps (with explainability)
# ════════════════════════════════════════════════════════════
with tab2:
    result = st.session_state.get("result")
    if not result:
        st.markdown("""<div class="callout-info">
        Run an analysis from the <b>Analyze</b> tab first. The pipeline steps will appear here
        with full explanations of what each step did and what it found in your document.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid {BORDER};border-radius:8px;
                    padding:12px 16px;margin-bottom:16px;font-size:0.87rem">
          🗂 <b>Run:</b> <code>{result.run_id}</code> &nbsp;·&nbsp;
          📄 <b>File:</b> {result.filename} &nbsp;·&nbsp;
          📁 <b>Type:</b> {result.source_type.upper()} &nbsp;·&nbsp;
          ⚖️ <b>Risk:</b> {RISK_ICON.get(result.cooccurrence.document_risk_score,"")} {result.cooccurrence.document_risk_score}
        </div>""", unsafe_allow_html=True)

        for sr in result.step_results:
            sico = STEP_ICON.get(sr.status, "ℹ️")
            info = STEP_EXPLAIN.get(sr.step, {})

            with st.expander(
                f"{sico} **Step {sr.step} — {sr.name}** &nbsp;·&nbsp; {sr.summary}",
                expanded=(sr.step <= 2 or sr.status in ("error","warning"))
            ):
                # Explainability block first
                render_step_explainer(sr.step)

                st.markdown("---")
                st.markdown("**Results from this document:**")

                d = sr.detail
                if not d:
                    st.caption("No detail available.")
                    continue

                if sr.step == 1:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Chunks extracted",  d.get("chunks",0))
                    c2.metric("Source format",     d.get("source_type","").upper())
                    c3.metric("Sections detected", len([s for s in d.get("sections",[]) if s]))
                    if d.get("sections"):
                        secs = [s for s in d["sections"] if s]
                        if secs:
                            st.markdown("**Sections found in document:** " +
                                " · ".join([f"`{s}`" for s in secs]))
                    st.markdown(f"""<div class="callout-info">
                    The document was split into <b>{d.get('chunks',0)} chunks</b>. Each chunk carries its
                    page number and section label so every downstream match can be precisely located.
                    </div>""", unsafe_allow_html=True)

                elif sr.step == 2:
                    c1,c2 = st.columns(2)
                    c1.metric("Exact matches found", d.get("match_count",0))
                    c2.metric("Categories triggered", len(d.get("categories_hit",[])))
                    if d.get("categories_hit"):
                        chips = " ".join([f'<span class="chip">{c}</span>'
                                          for c in d["categories_hit"]])
                        st.markdown(chips, unsafe_allow_html=True)
                    if d.get("match_count",0) == 0:
                        st.markdown('<div class="callout-warn">No exact keyword matches found. The semantic layer (Step 3) may still find relevant content.</div>',
                                    unsafe_allow_html=True)
                    if d.get("sample"):
                        st.markdown("**Sample exact matches:**")
                        st.dataframe(pd.DataFrame(d["sample"]), use_container_width=True, hide_index=True)

                elif sr.step == 3:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Semantic matches",    d.get("match_count",0))
                    c2.metric("Vectors in ChromaDB", d.get("vectordb_vectors",0))
                    c3.metric("Threshold used",      f"{settings.semantic_similarity_threshold:.2f}")
                    st.caption(f"Dictionary fingerprint: `{d.get('dict_hash','')}`  ·  DB path: `{d.get('vectordb_path','')}`")
                    if d.get("match_count",0) == 0:
                        st.markdown(f'<div class="callout-warn">No semantic matches above the {settings.semantic_similarity_threshold:.2f} threshold. Try lowering the threshold in the sidebar if you expect more coverage.</div>',
                                    unsafe_allow_html=True)
                    if d.get("sample"):
                        st.markdown("**Top semantic matches by similarity score:**")
                        st.dataframe(pd.DataFrame(d["sample"]), use_container_width=True, hide_index=True)

                elif sr.step == 4:
                    c1,c2,c3 = st.columns(3)
                    total = d.get("negated_count",0) + d.get("affirmed_count",0)
                    c1.metric("Total matches reviewed", total)
                    c2.metric("✅ Negated (denied)",    d.get("negated_count",0))
                    c3.metric("⚠️ Affirmed (adverse)",  d.get("affirmed_count",0))
                    if d.get("negated_count",0) > 0:
                        st.markdown(f"""<div class="callout-info">
                        <b>{d['negated_count']} match(es) were negated</b> — the document explicitly denies
                        these adverse conditions. They appear in results with a green "Negated" badge
                        and are excluded from the risk score.
                        </div>""", unsafe_allow_html=True)
                    for ex in d.get("negated_examples",[]):
                        st.markdown(f"- `{ex['kw']}` — *\"{ex['context'][:140]}...\"*")

                elif sr.step == 5:
                    c1,c2 = st.columns(2)
                    c1.metric("Demoted to Informational", d.get("demoted_to_informational",0))
                    c2.metric("Matches retained",
                              len(result.all_matches) - d.get("demoted_to_informational",0))
                    if d.get("demoted_to_informational",0) > 0:
                        st.markdown(f"""<div class="callout-info">
                        <b>{d['demoted_to_informational']} match(es)</b> were found in low-weight sections
                        (General Conditions, Exclusions, Signatures) and demoted to Informational.
                        These are standard policy language — not submission-specific risk language.
                        </div>""", unsafe_allow_html=True)
                    if d.get("section_breakdown"):
                        df = pd.DataFrame(list(d["section_breakdown"].items()),
                                          columns=["Section","Matches"])
                        df = df[df["Matches"] > 0].sort_values("Matches", ascending=False)
                        if not df.empty:
                            st.markdown("**Match distribution by section:**")
                            st.dataframe(df, use_container_width=True, hide_index=True)

                elif sr.step == 6:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Composite Risk Score", d.get("risk_score",""))
                    c2.metric("Numeric Value",        d.get("risk_value",""))
                    c3.metric("Categories triggered", len(d.get("triggered_categories",[])))
                    # Dict debug info
                    st.markdown(
                        f"<div class='callout-info'>📖 Dictionary used: "
                        f"<code>{d.get('dict_category_count','?')} categories</code> · "
                        f"hash <code>{d.get('dict_used_hash','?')}</code> · "
                        f"sample: {', '.join(d.get('dict_sample_categories',[]))}"
                        f"</div>", unsafe_allow_html=True
                    )
                    combos = d.get("high_risk_combos",[])
                    if combos:
                        st.markdown(f"""<div class="callout-warn">
                        <b>⚡ {len(combos)} high-risk category combination(s) detected.</b>
                        These combinations are historically correlated with declined submissions.
                        </div>""", unsafe_allow_html=True)
                        for combo in combos:
                            st.warning(f"⚡ {combo}")
                    else:
                        st.warning("No high-risk combos found — check that the correct keyword dictionary is loaded (see dict hash above vs sidebar hash)")
                    if d.get("category_frequency"):
                        df = pd.DataFrame(list(d["category_frequency"].items()),
                                          columns=["Category","Match Count"])
                        st.markdown("**Matches per risk category:**")
                        col_a, col_b = st.columns([2,3])
                        with col_a:
                            st.dataframe(df.sort_values("Match Count", ascending=False),
                                         use_container_width=True, hide_index=True)
                        with col_b:
                            st.bar_chart(df.set_index("Category"))

                elif sr.step == 7:
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Matches sent to LLM",  d.get("total_interpreted",0))
                    c2.metric("✅ Confirmed adverse", d.get("confirmed",0))
                    c3.metric("❌ Not confirmed",     d.get("denied",0))
                    c4.metric("Provider",             d.get("provider","").upper())
                    if d.get("total_interpreted",0) == 0:
                        st.markdown("""<div class="callout-info">
                        No medium/low confidence matches to interpret — either all matches were High
                        confidence (exact hits) or LLM interpretation was disabled.
                        </div>""", unsafe_allow_html=True)
                    if d.get("sample"):
                        st.markdown("**LLM assessments (sample):**")
                        for s in d["sample"]:
                            ic = "✅" if s["confirmed"] else "❌"
                            st.markdown(
                                f"<div style='border:1px solid {BORDER};border-radius:6px;"
                                f"padding:8px 12px;margin:4px 0;font-size:0.83rem'>"
                                f"{ic} <b>{s['cat']}</b> — {s['rationale']}</div>",
                                unsafe_allow_html=True
                            )

                elif sr.step == 8:
                    st.success(f"✅ Run `{d.get('run_id','')}` saved to audit trail and documents database.")
                    st.markdown(f"""<div class="callout-info">
                    The dictionary fingerprint recorded for this run is <code>{result.ingested_doc and 'recorded' or 'recorded'}</code>.
                    If the keyword dictionary is later updated, this run can still be compared against
                    the version that was active at analysis time.
                    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — Results
# ════════════════════════════════════════════════════════════
with tab3:
    result = st.session_state.get("result")
    if not result:
        st.markdown("""<div class="callout-info">
        Run an analysis from the <b>Analyze</b> tab first.
        </div>""", unsafe_allow_html=True)
    else:
        cooc = result.cooccurrence
        risk = cooc.document_risk_score
        cls  = RISK_CLS.get(risk, "risk-low")
        col  = RISK_COLOR.get(risk, "#166534")
        ico  = RISK_ICON.get(risk, "")

        st.markdown(f"""
        <div class="{cls}">
          <div style="font-size:1.3rem;font-weight:800;color:{col}">{ico} {risk} Risk — {result.filename}</div>
          <div style="color:{col}aa;margin-top:4px;font-size:0.9rem">{cooc.notes}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("")

        if cooc.triggered_categories:
            cols = st.columns(min(len(cooc.triggered_categories), 4))
            for i, cat in enumerate(cooc.triggered_categories):
                freq = cooc.category_frequency.get(cat, 0)
                cols[i%4].metric(cat, freq)

        if cooc.high_risk_combos_found:
            for c1x, c2x, w in cooc.high_risk_combos_found:
                st.warning(f"⚡ **{c1x}** + **{c2x}** — combined risk weight: {w}")

        st.divider()

        # Confidence guide
        with st.expander("📖 Understanding confidence levels and match types"):
            st.markdown("""
            | Level | Meaning | Source |
            |-------|---------|--------|
            | 🔴 **High** | Exact keyword match — fully deterministic | Aho-Corasick engine |
            | 🟡 **Medium** | Semantic similarity ≥ 0.80 — strong paraphrase match | ChromaDB embeddings |
            | 🟠 **Low** | Semantic similarity 0.72–0.79 — possible match, needs review | ChromaDB embeddings |
            | 🔵 **Informational** | Match found in boilerplate/exclusion section — low weight | Section weighter |

            **Match types:**
            - `exact` — the keyword appeared verbatim (or after normalization) in the document
            - `semantic` — the passage was semantically similar to a risk category

            **Affirmed vs Negated:**
            - ⚠️ **Affirmed** — the adverse condition is stated ("prior litigation was filed")
            - ✓ **Negated** — the adverse condition is explicitly denied ("no prior litigation")
            """)

        # Filters
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            fconf = st.multiselect("Confidence", ["High","Medium","Low","Informational"],
                                   default=["High","Medium","Low"])
        with f2:
            all_cats = sorted(set(getattr(m,"category","") for m in result.all_matches))
            fcat = st.multiselect("Category", all_cats, default=all_cats)
        with f3:
            ftype = st.multiselect("Match type", ["exact","semantic"],
                                   default=["exact","semantic"])
        with f4:
            show_neg = st.checkbox("Include negated", value=False)

        filtered = [
            m for m in result.all_matches
            if m.confidence in fconf
            and getattr(m,"category","") in fcat
            and getattr(m,"match_type","") in ftype
            and (show_neg or not getattr(m,"negation_flag",False))
        ]

        st.caption(f"**{len(filtered)}** of **{len(result.all_matches)}** total matches")
        view = st.radio("Display as", ["Cards — with paragraph context", "Flat table"],
                        horizontal=True, label_visibility="collapsed")

        if view == "Cards — with paragraph context":
            if not filtered:
                st.success("No matches with current filters.")
            for i, m in enumerate(filtered):
                render_match_card(m, i)
        else:
            rows = []
            for m in filtered:
                rows.append({
                    "Conf":     CONF_ICON.get(m.confidence,"")+" "+m.confidence,
                    "Category": getattr(m,"category",""),
                    "Keyword":  getattr(m,"keyword",getattr(m,"matched_keyword","")),
                    "Type":     getattr(m,"match_type",""),
                    "Page":     getattr(m,"page","-"),
                    "Section":  getattr(m,"section",""),
                    "Affirmed": "Negated" if getattr(m,"negation_flag",False) else "Yes",
                    "Score":    getattr(m,"similarity_score",""),
                    "LLM":      str(getattr(m,"llm_confirmed","-")),
                    "LLM Note": getattr(m,"llm_rationale",""),
                    "Context":  getattr(m,"chunk_text","")[:200],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        if result.all_matches:
            rows_exp = [{
                "Confidence":    m.confidence,
                "Category":      getattr(m,"category",""),
                "Keyword":       getattr(m,"keyword",getattr(m,"matched_keyword","")),
                "Match Type":    getattr(m,"match_type",""),
                "Page":          getattr(m,"page","-"),
                "Section":       getattr(m,"section",""),
                "Affirmed":      not getattr(m,"negation_flag",False),
                "Similarity":    getattr(m,"similarity_score",""),
                "LLM Confirmed": getattr(m,"llm_confirmed",""),
                "LLM Rationale": getattr(m,"llm_rationale",""),
                "Context":       getattr(m,"chunk_text","")[:500],
            } for m in result.all_matches]
            st.download_button("⬇️ Export full results CSV",
                pd.DataFrame(rows_exp).to_csv(index=False),
                f"results_{result.run_id}.csv", "text/csv")


# ════════════════════════════════════════════════════════════
# TAB 4 — Documents DB
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-hdr">All Analyzed Documents</div>', unsafe_allow_html=True)
    st.caption("Every document analyzed by this system is stored here with full match history.")

    stats = get_db_stats()
    if stats["total"] == 0:
        st.markdown("""<div class="callout-info">
        No documents analyzed yet. Upload and analyze a document from the <b>Analyze</b> tab
        to populate the database.
        </div>""", unsafe_allow_html=True)
    else:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Runs",      stats["total"])
        c2.metric("Unique Files",    stats["unique_files"])
        c3.metric("🔴 High Risk",    stats["high_risk"])
        c4.metric("🟡 Medium Risk",  stats["medium_risk"])
        c5.metric("🟢 Low Risk",     stats["low_risk"])

        st.divider()

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            risk_filter = st.multiselect("Risk filter", ["High","Medium","Low"],
                                          default=["High","Medium","Low"])
        with fc2:
            type_filter = st.multiselect("Type filter",
                ["pdf","docx","xlsx","eml","email","txt","acord","html"], default=[])
        with fc3:
            search_term = st.text_input("Search filename", placeholder="e.g. harborview")

        docs = load_all_documents(200)
        if risk_filter:
            docs = [d for d in docs if d.get("risk_score","") in risk_filter]
        if type_filter:
            docs = [d for d in docs if d.get("source_type","") in type_filter]
        if search_term:
            docs = [d for d in docs if search_term.lower() in d.get("filename","").lower()]

        st.caption(f"Showing {len(docs)} documents")

        for doc in docs:
            risk    = doc.get("risk_score","?")
            rico    = RISK_ICON.get(risk,"")
            fname   = doc.get("filename","")
            ts      = doc.get("timestamp","")[:19]
            run_id  = doc.get("run_id","")
            stype   = doc.get("source_type","").upper()
            summ    = doc.get("summary",{})
            cats    = doc.get("triggered_categories",[])
            combos  = doc.get("high_risk_combos",[])
            matches = doc.get("matches",[])

            with st.expander(
                f"{rico} **{fname}** · `{run_id}` · {ts} · {stype} · "
                f"Risk: **{risk}** · {summ.get('total_matches',0)} matches"
            ):
                s1,s2,s3,s4,s5,s6 = st.columns(6)
                s1.metric("Total",    summ.get("total_matches",0))
                s2.metric("🔴 High",  summ.get("high_confidence",0))
                s3.metric("🟡 Medium",summ.get("medium_confidence",0))
                s4.metric("Negated",  summ.get("negated",0))
                s5.metric("Exact",    summ.get("exact_matches",0))
                s6.metric("Semantic", summ.get("semantic_matches",0))

                if cats:
                    chips = " ".join([f'<span class="chip">{c}</span>' for c in cats])
                    st.markdown(f"**Triggered categories:** {chips}", unsafe_allow_html=True)

                if combos:
                    for combo in combos:
                        st.warning(f"⚡ **{combo.get('cat1','')}** + **{combo.get('cat2','')}** (weight {combo.get('weight','')})")

                freq = doc.get("category_frequency",{})
                if freq:
                    col_a, col_b = st.columns([1,2])
                    with col_a:
                        df_f = pd.DataFrame(list(freq.items()), columns=["Category","Matches"])
                        st.dataframe(df_f.sort_values("Matches",ascending=False),
                                     use_container_width=True, hide_index=True)
                    with col_b:
                        st.bar_chart(df_f.set_index("Category"))

                if matches:
                    affirmed = [m for m in matches if not m.get("negation_flag",False)]
                    negated  = [m for m in matches if m.get("negation_flag",False)]
                    view_m = st.radio("Show matches",
                                      ["Affirmed only","Negated only","All"],
                                      horizontal=True, key=f"dv_{run_id}")
                    show_list = {"Affirmed only":affirmed,"Negated only":negated,"All":matches}[view_m]
                    df_m = pd.DataFrame([{
                        "Conf":     CONF_ICON.get(m.get("confidence",""),"")+" "+m.get("confidence",""),
                        "Category": m.get("category",""),
                        "Keyword":  m.get("keyword",""),
                        "Type":     m.get("match_type",""),
                        "Page":     m.get("page","-"),
                        "Section":  m.get("section",""),
                        "Affirmed": "Negated" if m.get("negation_flag") else "Yes",
                        "LLM":      str(m.get("llm_confirmed","-")),
                        "Context":  m.get("chunk_text","")[:200],
                    } for m in show_list])
                    if not df_m.empty:
                        st.dataframe(df_m, use_container_width=True, hide_index=True)

                    st.download_button(
                        f"⬇️ Export results",
                        pd.DataFrame(matches).to_csv(index=False),
                        f"results_{run_id}.csv","text/csv",
                        key=f"exp_{run_id}"
                    )


# ════════════════════════════════════════════════════════════
# TAB 5 — Audit Trail
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-hdr">Audit Trail</div>', unsafe_allow_html=True)
    st.caption("Append-only record of every analysis run. Dictionary version is stored with each run for regulatory traceability.")

    records = [r for r in load_audit_records(50) if r.get("type") != "decision"]
    if not records:
        st.info("No audit records yet.")
    else:
        st.caption(f"{len(records)} runs recorded")
        for rec in reversed(records[:20]):
            s    = rec.get("summary",{})
            cooc = rec.get("cooccurrence",{})
            risk = cooc.get("risk_score","?")
            with st.expander(
                f"{RISK_ICON.get(risk,'')} Run `{rec.get('run_id','')}` · "
                f"{rec.get('document',{}).get('filename','')} · "
                f"{rec.get('timestamp','')[:19]} · {risk} Risk"
            ):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Matches",  s.get("total_matches",0))
                c2.metric("High",     s.get("high_confidence",0))
                c3.metric("Negated",  s.get("negated_matches",0))
                c4.metric("Provider", rec.get("llm_provider","").upper())
                st.caption(f"Dictionary version: `{rec.get('dictionary_version','')}`")
                cats = cooc.get("triggered_categories",[])
                if cats:
                    chips = " ".join([f'<span class="chip">{c}</span>' for c in cats])
                    st.markdown(f"**Categories:** {chips}", unsafe_allow_html=True)
                combos = cooc.get("high_risk_combos",[])
                for combo in combos:
                    st.warning(f"⚡ {combo.get('cat1','')} + {combo.get('cat2','')} (weight {combo.get('weight','')})")
