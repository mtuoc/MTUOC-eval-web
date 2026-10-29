"""
MTUOC-eval Web Edition (Streamlit - Exact Qt GUI Clone)
- Full GPU acceleration with PyTorch, CUDA cache cleanup, and RTX 5090 optimizations.
- Radar Charts: Un gràfic de radar PER CADA MÈTRICA comparant els sistemes (spokes = sistemes; requereix >= 3 sistemes).
  També gràfics de barres per mètrica comparant els sistemes.
- Suport per a mètriques amb valors negatius (p. ex. COMET-QE) tant a les barres com al radar.
- Renderització directa en Base64 Data URI per evitar errors de Starlette/Uvicorn GZipMiddleware.
- Botó de descàrrega client-side HTML5 que no recarrega la pàgina ni canvia de pestanya.
- Capçalera sense talls (padding-top: 4.5rem).
- 5 pestanyes principals idèntiques a PyQt6 amb exportacions a LaTeX, Excel i CSV.
"""

import os
import io
import gc
import difflib
import math
import base64
import warnings
import traceback
from typing import List, Dict, Any, Optional

import yaml
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*findfont:.*")

# Optimize Tensor Cores for modern GPUs (RTX 5090 / 4090)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Metric Imports & Fallbacks
# ---------------------------------------------------------------------------
try:
    import Levenshtein
    def fast_char_distance(s1: str, s2: str) -> int:
        return Levenshtein.distance(s1, s2)
    def fast_word_distance(w1: list, w2: list) -> int:
        return Levenshtein.distance(w1, w2)
except ImportError:
    def fast_char_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return fast_char_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

    def fast_word_distance(w1: list, w2: list) -> int:
        return fast_char_distance(w1, w2)

try:
    import sacrebleu
    from sacrebleu.metrics import BLEU, CHRF, TER
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False


# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_PATH = os.path.join(SCRIPT_DIR, "config.yaml")

DEFAULT_CONFIG = {
    "Hardware": {"device": "auto", "batch_size_gpu": 32, "batch_size_cpu": 8},
    "SacreBLEU": {
        "bleu_tokenize": "13a", "bleu_smooth_method": "exp", "bleu_lowercase": False,
        "chrf_char_order": 6, "chrf_word_order": 2, "chrf_whitespace": False,
        "ter_normalized": False, "ter_no_punct": False, "ter_case_sensitive": False
    },
    "Models": {
        "comet_default": "Unbabel/wmt22-comet-da",
        "comet_qe_default": "Unbabel/wmt20-comet-qe-da",
        "bertscore_default": "bert-base-multilingual-cased",
        "sbert_qe_default": "sentence-transformers/LaBSE"
    }
}

def load_config() -> dict:
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    return loaded
        except Exception:
            pass
    return DEFAULT_CONFIG

CONFIG = load_config()
MODELS_CFG = CONFIG.get("Models", DEFAULT_CONFIG["Models"])


# ---------------------------------------------------------------------------
# Calculation Helpers
# ---------------------------------------------------------------------------
def compute_diff(ref: str, hyp: str) -> str:
    matcher = difflib.SequenceMatcher(None, ref, hyp)
    parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            parts.append(ref[i1:i2])
        elif tag == 'delete':
            parts.append("{-" + ref[i1:i2] + "-}")
        elif tag == 'insert':
            parts.append("{+" + hyp[j1:j2] + "+}")
        elif tag == 'replace':
            parts.append("{-" + ref[i1:i2] + "-}{+" + hyp[j1:j2] + "+}")
    return "".join(parts)


def resolve_hf_model(model_name_or_path: str) -> str:
    import glob
    model_name_or_path = model_name_or_path.strip()
    if os.path.isfile(model_name_or_path):
        return model_name_or_path

    repo_id = model_name_or_path.split("huggingface.co/")[-1].strip("/")
    try:
        from comet import download_model
        path = download_model(repo_id)
        if path:
            if os.path.isfile(path):
                return path
            ckpts = glob.glob(os.path.join(path, "**", "*.ckpt"), recursive=True)
            if ckpts:
                return ckpts[0]
    except Exception:
        pass

    try:
        from huggingface_hub import snapshot_download
        folder = snapshot_download(repo_id=repo_id)
        ckpts = glob.glob(os.path.join(folder, "**", "*.ckpt"), recursive=True)
        return ckpts[0] if ckpts else folder
    except Exception as e:
        raise RuntimeError(f"Could not load model '{repo_id}': {e}")


def display_figure_safely(fig: plt.Figure, download_filename: str):
    """
    Renderitza la figura de Matplotlib com a PNG inline en Base64.
    Evita completament les peticions HTTP a Uvicorn/Starlette (GZipMiddleware),
    i utilitza un botó de descàrrega directe en client per no recarregar la pàgina.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.getvalue()
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    plt.close(fig)

    col_img, col_btn = st.columns([4, 1.2])
    with col_img:
        st.markdown(
            f'<div style="text-align: center; margin-bottom: 12px;">'
            f'<img src="data:image/png;base64,{b64_str}" style="max-width: 100%; height: auto; border-radius: 6px; border: 1px solid #d0d7de;" />'
            f'</div>',
            unsafe_allow_html=True
        )
    with col_btn:
        download_html = f"""
        <div style="margin-top: 10px;">
            <a href="data:image/png;base64,{b64_str}" download="{download_filename}" style="
                display: inline-block;
                width: 100%;
                padding: 0.55rem 1rem;
                background-color: #f6f8fa;
                color: #24292f;
                font-weight: 600;
                font-size: 14px;
                text-align: center;
                text-decoration: none;
                border: 1px solid #d0d7de;
                border-radius: 6px;
                box-shadow: 0 1px 0 rgba(27,31,36,0.04);
                cursor: pointer;
            " onmouseover="this.style.backgroundColor='#f3f4f6'" onmouseout="this.style.backgroundColor='#f6f8fa'">
                💾 Save as PNG...
            </a>
        </div>
        """
        st.markdown(download_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Streamlit Page Setup & Custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MTUOC-eval v6.5 — Machine Translation Evaluation Platform",
    page_icon="🖥️",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        font-weight: 600;
        border-radius: 4px 4px 0px 0px;
    }
    .group-title {
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 8px;
        color: #24292f;
    }
</style>
""", unsafe_allow_html=True)

if "ref_sys_count" not in st.session_state:
    st.session_state["ref_sys_count"] = 1
if "qe_sys_count" not in st.session_state:
    st.session_state["qe_sys_count"] = 1
if "results_data" not in st.session_state:
    st.session_state["results_data"] = None

hw_setting = CONFIG.get("Hardware", {}).get("device", "auto").lower()
if hw_setting == "cuda":
    use_cuda = torch.cuda.is_available()
elif hw_setting == "cpu":
    use_cuda = False
else:
    use_cuda = torch.cuda.is_available()

device_str = "cuda" if use_cuda else "cpu"
batch_size_gpu = CONFIG.get("Hardware", {}).get("batch_size_gpu", 32)
batch_size_cpu = CONFIG.get("Hardware", {}).get("batch_size_cpu", 8)
batch_size = batch_size_gpu if use_cuda else batch_size_cpu

st.markdown("### MTUOC-eval v6.5 — Machine Translation Evaluation Platform")

main_tab_ref, main_tab_qe, main_tab_segments, main_tab_vis, main_tab_sig = st.tabs([
    "Reference-Based Evaluation",
    "Referenceless (QE)",
    "Segment Details",
    "Visualizations",
    "Signatures"
])

# ===========================================================================
# PESTANYA 1: Reference-Based Evaluation
# ===========================================================================
with main_tab_ref:
    st.markdown('<div class="group-title">Input Files</div>', unsafe_allow_html=True)
    col_src, col_ref = st.columns(2)
    with col_src:
        ref_src_file = st.file_uploader("Source file (Optional, needed for COMET):", type=None, key="ref_src_uploader")
    with col_ref:
        ref_ref_file = st.file_uploader("Reference file (* Required):", type=None, key="ref_ref_uploader")

    st.markdown("---")
    st.markdown('<div class="group-title">MT Systems to Evaluate</div>', unsafe_allow_html=True)

    ref_systems = []
    for i in range(st.session_state["ref_sys_count"]):
        col_lbl, col_name, col_file = st.columns([1, 2.5, 6])
        with col_lbl:
            st.markdown(f"<p style='margin-top: 32px; font-weight: bold;'>System #{i+1}:</p>", unsafe_allow_html=True)
        with col_name:
            s_name = st.text_input(f"Name #{i+1}", value=f"System_{i+1}", key=f"ref_sys_name_{i}", label_visibility="collapsed")
        with col_file:
            s_file = st.file_uploader(f"File #{i+1}", type=None, key=f"ref_sys_file_{i}", label_visibility="collapsed")
        ref_systems.append({"name": s_name, "file": s_file})

    col_btn_add, col_btn_rem = st.columns([2, 8])
    with col_btn_add:
        if st.button("+ Add Another System", key="btn_add_ref_sys"):
            st.session_state["ref_sys_count"] += 1
            st.rerun()
    with col_btn_rem:
        if st.session_state["ref_sys_count"] > 1:
            if st.button("Remove Last System", key="btn_rem_ref_sys"):
                st.session_state["ref_sys_count"] -= 1
                st.rerun()

    st.markdown("---")
    st.markdown('<div class="group-title">Metrics & Models</div>', unsafe_allow_html=True)
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        cb_bleu = st.checkbox("BLEU (SacreBLEU)", value=True, key="ref_cb_bleu")
    with c_m2:
        cb_chrf = st.checkbox("chrF++ (Character & Word n-grams)", value=True, key="ref_cb_chrf")
    with c_m3:
        cb_ter = st.checkbox("TER (Translation Edit Rate)", value=True, key="ref_cb_ter")
    with c_m4:
        cb_ed = st.checkbox("Edit Distance (%EdDist & WER)", value=True, key="ref_cb_ed")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        c_sub1, c_sub2 = st.columns([1, 3])
        with c_sub1:
            cb_comet = st.checkbox("COMET:", value=False, key="ref_cb_comet")
        with c_sub2:
            txt_comet_model = st.text_input("COMET Model", value=MODELS_CFG.get("comet_default", "Unbabel/wmt22-comet-da"), key="ref_txt_comet", label_visibility="collapsed")
    with col_c2:
        b_sub1, b_sub2 = st.columns([1, 3])
        with b_sub1:
            cb_bert = st.checkbox("BERTScore:", value=False, key="ref_cb_bert")
        with b_sub2:
            txt_bert_model = st.text_input("BERTScore Model", value=MODELS_CFG.get("bertscore_default", "bert-base-multilingual-cased"), key="ref_txt_bert", label_visibility="collapsed")

    btn_run_ref = st.button("Run Reference-Based Evaluation", type="primary", width='stretch')

# ===========================================================================
# PESTANYA 2: Referenceless (QE)
# ===========================================================================
with main_tab_qe:
    st.markdown('<div class="group-title">Input File (No reference needed)</div>', unsafe_allow_html=True)
    qe_src_file = st.file_uploader("Source file (* Required):", type=None, key="qe_src_uploader")

    st.markdown("---")
    st.markdown('<div class="group-title">MT Systems to Evaluate</div>', unsafe_allow_html=True)

    qe_systems = []
    for i in range(st.session_state["qe_sys_count"]):
        col_lbl, col_name, col_file = st.columns([1, 2.5, 6])
        with col_lbl:
            st.markdown(f"<p style='margin-top: 32px; font-weight: bold;'>System #{i+1}:</p>", unsafe_allow_html=True)
        with col_name:
            s_name = st.text_input(f"QE Name #{i+1}", value=f"System_{i+1}", key=f"qe_sys_name_{i}", label_visibility="collapsed")
        with col_file:
            s_file = st.file_uploader(f"QE File #{i+1}", type=None, key=f"qe_sys_file_{i}", label_visibility="collapsed")
        qe_systems.append({"name": s_name, "file": s_file})

    col_btn_qe_add, col_btn_qe_rem = st.columns([2, 8])
    with col_btn_qe_add:
        if st.button("+ Add Another System", key="btn_add_qe_sys"):
            st.session_state["qe_sys_count"] += 1
            st.rerun()
    with col_btn_qe_rem:
        if st.session_state["qe_sys_count"] > 1:
            if st.button("Remove Last System", key="btn_rem_qe_sys"):
                st.session_state["qe_sys_count"] -= 1
                st.rerun()

    st.markdown("---")
    st.markdown('<div class="group-title">Quality Estimation Metrics</div>', unsafe_allow_html=True)
    col_qe_c, col_qe_s = st.columns(2)
    with col_qe_c:
        sub_c1, sub_c2 = st.columns([1.2, 3])
        with sub_c1:
            cb_comet_qe = st.checkbox("COMET-QE:", value=True, key="qe_cb_comet")
        with sub_c2:
            txt_comet_qe_model = st.text_input("COMET-QE Model", value=MODELS_CFG.get("comet_qe_default", "Unbabel/wmt20-comet-qe-da"), key="qe_txt_comet", label_visibility="collapsed")
    with col_qe_s:
        sub_s1, sub_s2 = st.columns([1.5, 3])
        with sub_s1:
            cb_sbert_qe = st.checkbox("SBERT (Cosine Sim):", value=False, key="qe_cb_sbert")
        with sub_s2:
            txt_sbert_qe_model = st.text_input("SBERT Model", value=MODELS_CFG.get("sbert_qe_default", "sentence-transformers/LaBSE"), key="qe_txt_sbert", label_visibility="collapsed")

    btn_run_qe = st.button("Run Referenceless (QE) Evaluation", type="primary", width='stretch')


# ===========================================================================
# MOTOR D'EXECUCIÓ
# ===========================================================================
active_mode = None
if btn_run_ref:
    active_mode = "reference"
elif btn_run_qe:
    active_mode = "referenceless"

if active_mode:
    if active_mode == "reference":
        if not ref_ref_file:
            st.error("Reference file is required for reference-based evaluation.")
            st.stop()
        configured_systems = [s for s in ref_systems if s["file"] is not None]
        if not configured_systems:
            st.error("Please configure at least one system with a valid file.")
            st.stop()

        ref_ref_file.seek(0)
        ref_lines = [l.decode("utf-8").rstrip("\r\n") for l in ref_ref_file]
        
        if ref_src_file:
            ref_src_file.seek(0)
            src_lines = [l.decode("utf-8").rstrip("\r\n") for l in ref_src_file]
        else:
            src_lines = []
        total_lines = len(ref_lines)

        sys_dict = {}
        for s in configured_systems:
            s["file"].seek(0)
            hyps = [l.decode("utf-8").rstrip("\r\n") for l in s["file"]]
            if len(hyps) != total_lines:
                st.error(f"Line mismatch: expected {total_lines} lines, but '{s['name']}' has {len(hyps)} lines.")
                st.stop()
            sys_dict[s["name"]] = hyps

        selected_m = {
            "bleu": cb_bleu, "chrf": cb_chrf, "ter": cb_ter, "ed_dist": cb_ed,
            "comet": cb_comet, "bertscore": cb_bert
        }
    else:
        if not qe_src_file:
            st.error("Source file is required for referenceless (QE) evaluation.")
            st.stop()
        configured_systems = [s for s in qe_systems if s["file"] is not None]
        if not configured_systems:
            st.error("Please configure at least one system with a valid file.")
            st.stop()

        qe_src_file.seek(0)
        src_lines = [l.decode("utf-8").rstrip("\r\n") for l in qe_src_file]
        ref_lines = []
        total_lines = len(src_lines)

        sys_dict = {}
        for s in configured_systems:
            s["file"].seek(0)
            hyps = [l.decode("utf-8").rstrip("\r\n") for l in s["file"]]
            if len(hyps) != total_lines:
                st.error(f"Line mismatch: expected {total_lines} lines, but '{s['name']}' has {len(hyps)} lines.")
                st.stop()
            sys_dict[s["name"]] = hyps

        selected_m = {
            "comet_qe": cb_comet_qe, "sbert_qe": cb_sbert_qe
        }

    prog_slot = st.empty()
    status_slot = st.empty()
    prog_bar = prog_slot.progress(5)
    status_slot.text("Initializing environment...")

    corpus_summary = {}
    signatures = {}
    seg_scores = {s: {} for s in sys_dict}

    try:
        # FASE 1: Standard metrics
        if active_mode == "reference":
            sb_cfg = CONFIG.get("SacreBLEU", {})
            status_slot.text("Evaluating standard metrics (BLEU, chrF++, TER, Edit Distance)...")
            prog_bar.progress(15)

            if selected_m.get("bleu") and HAS_SACREBLEU:
                b_obj = BLEU(tokenize=sb_cfg.get("bleu_tokenize", "13a"), smooth_method=sb_cfg.get("bleu_smooth_method", "exp"), effective_order=True)
                corpus_summary["BLEU"] = {s: round(b_obj.corpus_score(hyps, [ref_lines]).score, 2) for s, hyps in sys_dict.items()}
                for s, hyps in sys_dict.items():
                    seg_scores[s]["BLEU"] = [round(b_obj.sentence_score(h, [r]).score, 2) for h, r in zip(hyps, ref_lines)]
                signatures["BLEU"] = str(b_obj.get_signature())

            if selected_m.get("chrf") and HAS_SACREBLEU:
                c_obj = CHRF(char_order=sb_cfg.get("chrf_char_order", 6), word_order=sb_cfg.get("chrf_word_order", 2))
                corpus_summary["chrF++"] = {s: round(c_obj.corpus_score(hyps, [ref_lines]).score, 2) for s, hyps in sys_dict.items()}
                for s, hyps in sys_dict.items():
                    seg_scores[s]["chrF++"] = [round(c_obj.sentence_score(h, [r]).score, 2) for h, r in zip(hyps, ref_lines)]
                signatures["chrF++"] = str(c_obj.get_signature())

            if selected_m.get("ter") and HAS_SACREBLEU:
                t_obj = TER(normalized=sb_cfg.get("ter_normalized", False))
                corpus_summary["TER"] = {s: round(t_obj.corpus_score(hyps, [ref_lines]).score, 2) for s, hyps in sys_dict.items()}
                for s, hyps in sys_dict.items():
                    seg_scores[s]["TER"] = [round(t_obj.sentence_score(h, [r]).score, 2) for h, r in zip(hyps, ref_lines)]
                signatures["TER"] = str(t_obj.get_signature())

            if selected_m.get("ed_dist"):
                for s, hyps in sys_dict.items():
                    tot_c = sum(fast_char_distance(h, r) for h, r in zip(hyps, ref_lines))
                    tot_cl = sum(max(len(r), 1) for r in ref_lines)
                    tot_w = sum(fast_word_distance(h.split(), r.split()) for h, r in zip(hyps, ref_lines))
                    tot_wl = sum(max(len(r.split()), 1) for r in ref_lines)
                    corpus_summary.setdefault("%EdDist (Char)", {})[s] = round((tot_c / tot_cl) * 100.0, 2)
                    corpus_summary.setdefault("WER (Word)", {})[s] = round((tot_w / tot_wl) * 100.0, 2)

                    seg_scores[s]["%EdDist"] = [round((fast_char_distance(h, r) / max(len(r), 1)) * 100.0, 2) for h, r in zip(hyps, ref_lines)]
                    seg_scores[s]["WER"] = [round((fast_word_distance(h.split(), r.split()) / max(len(r.split()), 1)) * 100.0, 2) for h, r in zip(hyps, ref_lines)]

                signatures["%EdDist (Char)"] = "Levenshtein Character Edit Distance normalized by Reference length (%)"
                signatures["WER (Word)"] = "Levenshtein Word Error Rate normalized by Reference word count (%)"

            # FASE 2: COMET
            if selected_m.get("comet"):
                if not src_lines:
                    st.warning("COMET skipped: source file was not provided.")
                else:
                    status_slot.text(f"Evaluating COMET ({txt_comet_model})...")
                    prog_bar.progress(40)
                    try:
                        from comet import load_from_checkpoint
                        ckpt = resolve_hf_model(txt_comet_model)
                        c_inst = load_from_checkpoint(ckpt)
                        
                        for s, hyps in sys_dict.items():
                            data = [{"src": s_l, "mt": h, "ref": r} for s_l, h, r in zip(src_lines, hyps, ref_lines)]
                            try:
                                out = c_inst.predict(data, batch_size=batch_size, gpus=1 if use_cuda else 0, num_workers=0, progress_bar=False)
                            except Exception:
                                out = c_inst.predict(data, batch_size=batch_size, gpus=0, num_workers=0, progress_bar=False)
                            
                            corpus_summary.setdefault("COMET", {})[s] = round(float(out.system_score), 4)
                            seg_scores[s]["COMET"] = [round(float(sc), 4) for sc in out.scores]
                            
                        signatures["COMET"] = f"model:{txt_comet_model}|device:{device_str}"
                        del c_inst
                        if use_cuda: torch.cuda.empty_cache()
                    except Exception as comet_err:
                        st.warning(f"COMET evaluation skipped due to environment incompatibility: {comet_err}")

            # FASE 3: BERTScore
            if selected_m.get("bertscore"):
                status_slot.text(f"Evaluating BERTScore ({txt_bert_model})...")
                prog_bar.progress(70)
                try:
                    import bert_score
                    for s, hyps in sys_dict.items():
                        _, _, F1 = bert_score.score(hyps, ref_lines, model_type=txt_bert_model.strip(), device=device_str, batch_size=batch_size, verbose=False)
                        f1_vals = F1.tolist()
                        corpus_summary.setdefault("BERTScore (F1)", {})[s] = round(float(sum(f1_vals) / len(f1_vals)), 4)
                        seg_scores[s]["BERTScore"] = [round(float(v), 4) for v in f1_vals]
                    signatures["BERTScore"] = f"model:{txt_bert_model}|device:{device_str}"
                    if use_cuda: torch.cuda.empty_cache()
                except Exception as bert_err:
                    st.warning(f"BERTScore evaluation skipped: {bert_err}")

        # Referenceless (QE) Mode
        else:
            if selected_m.get("comet_qe"):
                status_slot.text(f"Evaluating COMET-QE ({txt_comet_qe_model})...")
                prog_bar.progress(40)
                try:
                    from comet import load_from_checkpoint
                    ckpt = resolve_hf_model(txt_comet_qe_model)
                    cqe_inst = load_from_checkpoint(ckpt)
                    
                    for s, hyps in sys_dict.items():
                        data = [{"src": s_l, "mt": h} for s_l, h in zip(src_lines, hyps)]
                        try:
                            out = cqe_inst.predict(data, batch_size=batch_size, gpus=1 if use_cuda else 0, num_workers=0, progress_bar=False)
                        except Exception:
                            out = cqe_inst.predict(data, batch_size=batch_size, gpus=0, num_workers=0, progress_bar=False)
                        
                        corpus_summary.setdefault("COMET-QE", {})[s] = round(float(out.system_score), 4)
                        seg_scores[s]["COMET-QE"] = [round(float(sc), 4) for sc in out.scores]
                        
                    signatures["COMET-QE"] = f"model:{txt_comet_qe_model}|device:{device_str}"
                    del cqe_inst
                    if use_cuda: torch.cuda.empty_cache()
                except Exception as cqe_err:
                    st.warning(f"COMET-QE evaluation skipped: {cqe_err}")

            if selected_m.get("sbert_qe"):
                status_slot.text(f"Evaluating SBERT Cosine Similarity ({txt_sbert_qe_model})...")
                prog_bar.progress(70)
                try:
                    from sentence_transformers import SentenceTransformer, util
                    sbert_model = SentenceTransformer(txt_sbert_qe_model, device=device_str)
                    src_emb = sbert_model.encode(src_lines, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
                    for s, hyps in sys_dict.items():
                        hyp_emb = sbert_model.encode(hyps, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
                        cos_sims = util.cos_sim(src_emb, hyp_emb).diagonal().tolist()
                        scores_list = [round(float(v), 4) for v in cos_sims]
                        corpus_summary.setdefault("SBERT-QE (Cosine)", {})[s] = round(float(sum(scores_list) / len(scores_list)), 4)
                        seg_scores[s]["SBERT-QE"] = scores_list
                    signatures["SBERT-QE"] = f"model:{txt_sbert_qe_model}|metric:cosine_similarity|device:{device_str}"
                    del sbert_model
                    if use_cuda: torch.cuda.empty_cache()
                except Exception as sbert_err:
                    st.warning(f"SBERT evaluation skipped: {sbert_err}")

        status_slot.text("Building detailed segment records...")
        prog_bar.progress(95)

        all_segment_records = []
        system_names = list(sys_dict.keys())
        sample_metrics = list(seg_scores[system_names[0]].keys()) if system_names else []

        for s in system_names:
            hyps = sys_dict[s]
            for i in range(total_lines):
                rec = {"ID": i + 1, "System": s}
                for m in sample_metrics:
                    rec[m] = seg_scores[s][m][i]
                rec["Hypothesis"] = hyps[i]
                if ref_lines:
                    rec["Reference"] = ref_lines[i]
                if src_lines:
                    rec["Original"] = src_lines[i]
                if ref_lines:
                    rec["DIFF"] = compute_diff(ref_lines[i], hyps[i])
                all_segment_records.append(rec)

        prog_bar.progress(100)
        status_slot.text("Evaluation completed.")

        # Desem els resultats a la sessió
        st.session_state["results_data"] = {
            "corpus": corpus_summary,
            "segments_df": pd.DataFrame(all_segment_records),
            "systems": system_names,
            "signatures": signatures,
            "has_ref": bool(ref_lines),
            "has_src": bool(src_lines),
            "mode": active_mode
        }
        st.success("Evaluation finished successfully! Check the other tabs for results.")

    except Exception as e:
        if use_cuda: torch.cuda.empty_cache()
        st.error(f"Evaluation Error: {e}")
        st.code(traceback.format_exc(), language="python")


# ===========================================================================
# PESTANYES DE RESULTATS: Segment Details, Visualizations, Signatures
# ===========================================================================
results = st.session_state["results_data"]

with main_tab_segments:
    if results is None:
        st.info("Run an evaluation first to see segment-level details.")
    else:
        st.markdown("<b>Detailed Segments Table:</b>", unsafe_allow_html=True)
        st.dataframe(results["segments_df"], width='stretch', height=520)

with main_tab_vis:
    if results is None:
        st.info("Run an evaluation first to view charts.")
    else:
        corpus = results["corpus"]
        systems = results["systems"]

        numeric_metrics = [m for m, s_dict in corpus.items() if any(isinstance(v, (int, float)) for v in s_dict.values())]
        if not numeric_metrics:
            st.info("No numeric metrics available to plot.")
        elif len(systems) <= 1:
            st.info("Comparative charts require at least 2 systems to be evaluated.")
        else:
            # Per cada mètrica oferim Bar Chart i (si len(systems) >= 3) Radar Chart comparant els sistemes
            subtab_labels = []
            for m in numeric_metrics:
                subtab_labels.append(f"Bar: {m}")
                if len(systems) >= 3:
                    subtab_labels.append(f"Radar: {m}")

            vis_subtabs = st.tabs(subtab_labels)
            palette = plt.cm.tab10(np.linspace(0, 1, max(len(systems), 10)))

            tab_cursor = 0
            for metric in numeric_metrics:
                # -----------------------------------------------------------
                # A. Gràfic de Barres comparant els sistemes per a aquesta mètrica
                # -----------------------------------------------------------
                with vis_subtabs[tab_cursor]:
                    fig, ax = plt.subplots(figsize=(8.5, 4.5))
                    scores = [float(corpus[metric].get(s, 0.0)) for s in systems]
                    bar_w = 0.45 if len(systems) > 1 else 0.25
                    bars = ax.bar(systems, scores, color=[palette[i % len(palette)] for i in range(len(systems))], width=bar_w, edgecolor='black', linewidth=0.8)
                    ax.set_title(f"{metric} — Comparison Across Systems", fontsize=12, fontweight="bold", pad=12)
                    ax.set_ylabel("Score", fontsize=10)
                    ax.grid(axis='y', linestyle='--', alpha=0.5)

                    for bar_item in bars:
                        h = bar_item.get_height()
                        fmt = f"{h:.2f}" if abs(h) >= 1 else f"{h:.4f}"
                        ax.annotate(fmt, xy=(bar_item.get_x() + bar_item.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

                    y_vals = [v for v in scores if isinstance(v, (int, float))]
                    if y_vals:
                        min_y, max_y = min(y_vals), max(y_vals)
                        span = max_y - min_y if max_y != min_y else abs(max_y) * 0.2
                        margin = span * 0.2 if span != 0 else 1.0
                        lower_limit = min_y - margin / 2 if min_y < 0 else max(0.0, min_y - margin / 2)
                        ax.set_ylim([lower_limit, max_y + margin])

                    fig.tight_layout()
                    clean_name = metric.replace(" ", "_").replace("%", "pct").replace("/", "_")
                    display_figure_safely(fig, f"{clean_name}_bars.png")
                tab_cursor += 1

                # -----------------------------------------------------------
                # B. Gràfic de Radar comparant els sistemes per a aquesta mètrica
                # (Cada eix o spoke és un SISTEMA, avaluant la mateixa mètrica)
                # -----------------------------------------------------------
                if len(systems) >= 3:
                    with vis_subtabs[tab_cursor]:
                        fig_rad = plt.figure(figsize=(8.5, 5.2))
                        ax_rad = fig_rad.add_subplot(111, polar=True)

                        raw_vals = [float(corpus[metric].get(s, 0.0)) for s in systems]
                        # Tanquem el cercle del polígon
                        plot_vals = raw_vals + raw_vals[:1]

                        num_sys = len(systems)
                        angles = np.linspace(0, 2 * np.pi, num_sys, endpoint=False).tolist()
                        angles += angles[:1]

                        # Etiquetes a cada vèrtex: Nom del sistema + Puntuació
                        spoke_labels = []
                        for s_name, val in zip(systems, raw_vals):
                            fmt = f"{val:.2f}" if abs(val) >= 1 else f"{val:.4f}"
                            spoke_labels.append(f"{s_name}\n({fmt})")

                        # Dibuixem el polígon radar comparant els sistemes
                        ax_rad.plot(angles, plot_vals, linewidth=2.5, color="#1f77b4", marker='o', markersize=6)
                        ax_rad.fill(angles, plot_vals, color="#1f77b4", alpha=0.20)

                        ax_rad.set_xticks(angles[:-1])
                        ax_rad.set_xticklabels(spoke_labels, fontsize=9.5, fontweight='bold')

                        # Escala radial (suporta valors negatius com ara COMET-QE)
                        min_val, max_val = min(raw_vals), max(raw_vals)
                        if min_val == max_val:
                            bottom = min_val - abs(min_val) * 0.2 if min_val < 0 else 0.0
                            top = max_val + abs(max_val) * 0.2 if max_val != 0 else 1.0
                            ax_rad.set_ylim(bottom, top)
                        else:
                            diff = max_val - min_val
                            bottom = min_val - diff * 0.25 if min_val < 0 else max(0.0, min_val - diff * 0.25)
                            top = max_val + diff * 0.15
                            ax_rad.set_ylim(bottom, top)

                        ax_rad.set_title(f"{metric} — Radar Comparison Across Systems", fontsize=11.5, fontweight="bold", pad=16)
                        ax_rad.grid(True, linestyle='--', alpha=0.6)
                        fig_rad.tight_layout()
                        display_figure_safely(fig_rad, f"{clean_name}_radar.png")
                    tab_cursor += 1

with main_tab_sig:
    st.markdown("<b>Evaluation Signatures for Papers:</b><br><small>Copy and paste these exact signatures to guarantee reproducibility.</small>", unsafe_allow_html=True)
    if results is None:
        st.text_area("Signatures Output", value="# Signatures will appear here after evaluation.", height=260, label_visibility="collapsed")
    else:
        sig_lines = ["# Evaluation Signatures & Reproducibility Metadata\n"]
        for m, sig in results["signatures"].items():
            sig_lines.append(f"{m}:\n  {sig}\n")
        st.text_area("Signatures Output", value="\n".join(sig_lines), height=260, label_visibility="collapsed")


# ===========================================================================
# BOTTOM GLOBAL AREA
# ===========================================================================
st.markdown("---")
col_stat, col_exp_latex, col_exp_excel, col_exp_csv = st.columns([3, 2, 2, 2])

with col_stat:
    status_msg = "Evaluation completed." if results is not None else "Ready."
    st.markdown(f"**Status:** `{status_msg}`")

if results is not None:
    corpus_dict = results["corpus"]
    systems = results["systems"]
    metrics_list = list(corpus_dict.keys())

    rows = []
    for m in metrics_list:
        row = {"Metric": m}
        for s in systems:
            row[s] = corpus_dict[m].get(s, "")
        rows.append(row)
    corpus_df = pd.DataFrame(rows).set_index("Metric")

    with col_exp_latex:
        col_def = "l" + "c" * len(systems)
        latex_lines = [
            "% Add \\usepackage{booktabs} to your LaTeX preamble",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Machine Translation evaluation results across systems.}",
            "\\label{tab:mt_evaluation_results}",
            f"\\begin{{tabular}}{{{col_def}}}",
            "\\toprule",
            "\\textbf{Metric} & " + " & ".join(f"\\textbf{{{s}}}" for s in systems) + " \\\\",
            "\\midrule"
        ]
        for m in metrics_list:
            vals = [str(corpus_dict[m].get(s, "")) for s in systems]
            escaped_m = m.replace("%", "\\%")
            latex_lines.append(f"{escaped_m} & " + " & ".join(vals) + " \\\\")
        latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        st.download_button("Export to LaTeX...", "\n".join(latex_lines), "mt_evaluation_table.tex", "text/plain", width='stretch')

    with col_exp_excel:
        xl_buf = io.BytesIO()
        with pd.ExcelWriter(xl_buf, engine='openpyxl') as writer:
            corpus_df.to_excel(writer, sheet_name="Corpus Summary")
            results["segments_df"].to_excel(writer, sheet_name="Segment Details", index=False)
        st.download_button("Export to Excel (.xlsx)", xl_buf.getvalue(), "mt_evaluation_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width='stretch')

    with col_exp_csv:
        csv_buf = io.StringIO()
        results["segments_df"].to_csv(csv_buf, index=False)
        st.download_button("Export to CSV (.csv)", csv_buf.getvalue(), "mt_evaluation_segments.csv", "text/csv", width='stretch')

    st.dataframe(corpus_df, width='stretch')
else:
    with col_exp_latex:
        st.button("Export to LaTeX...", disabled=True, width='stretch')
    with col_exp_excel:
        st.button("Export to Excel (.xlsx)", disabled=True, width='stretch')
    with col_exp_csv:
        st.button("Export to CSV (.csv)", disabled=True, width='stretch')
