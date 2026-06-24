"""Streamlit demo for audio-rag — entry point.

Prerequisites:
    pip install streamlit
    # Ollama must be running locally with the required model pulled

Run from the project root:
    streamlit run demo/app.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Ensure the project root (/app) is on sys.path so src.audio_rag.* is importable
# when Streamlit is launched from the repo root or via Docker.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

import session
import ui_components


def _cited_references(summary: str, references: list) -> list:
    """Return only the references whose filename is explicitly mentioned in the answer.

    The LLM is instructed to cite sources by name; we match both the full
    filename (e.g. 'podcast.wav') and the stem ('podcast') to handle cases
    where the model drops the extension.
    Falls back to the full reference list if no filename is found in the text.
    """
    return [
        ref for ref in references
        if ref.file_name in summary or Path(ref.file_name).stem in summary
    ]

# ── Page config (must be the first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="audio-rag demo",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session bootstrap ─────────────────────────────────────────────────────────
session.init()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎙️ audio-rag")
st.sidebar.caption("Local, privacy-first RAG over audio files")
st.sidebar.divider()

# Knowledge base status
ui_components.render_kb_status(
    st.session_state.kb_ready,
    len(st.session_state.indexed_files),
)

# File uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload audio files",
    type=["mp3", "wav", "m4a", "ogg", "flac"],
    accept_multiple_files=True,
    help="Supported formats: mp3, wav, m4a, ogg, flac",
)

ingest_clicked = st.sidebar.button(
    "▶ Transcribe & Index", use_container_width=True, type="primary"
)

if ingest_clicked:
    if not uploaded_files:
        st.sidebar.warning("Select at least one audio file first.")
    else:
        pipeline = session.get_pipeline()  # ensures pipeline is ready
        progress = st.sidebar.progress(0, text="Starting…")
        new_count = 0

        for i, uf in enumerate(uploaded_files, 1):
            progress.progress(i / len(uploaded_files), text=f"Processing {uf.name}…")

            if session.already_indexed(uf.name):
                st.sidebar.caption(f"⏭ `{uf.name}` already indexed — skipping")
                continue

            try:
                meta = session.ingest_uploaded_file(uf)
                st.session_state.indexed_files.append(meta)
                new_count += 1
            except Exception as exc:
                st.sidebar.error(f"❌ Failed: `{uf.name}`")
                with st.sidebar.expander("Error details"):
                    st.code(traceback.format_exc())

        progress.empty()
        if new_count:
            st.sidebar.success(f"✅ {new_count} new file(s) indexed.")
        st.rerun()

# Indexed files list
ui_components.render_indexed_files_sidebar(st.session_state.indexed_files)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("audio-rag demo")

tab_chat, tab_transcriptions = st.tabs(["💬 Chat", "📄 Transcriptions"])

# ── Chat tab ──────────────────────────────────────────────────────────────────
with tab_chat:
    # Render existing history
    for msg in st.session_state.chat_history:
        ui_components.render_chat_message(msg)

    question = st.chat_input("Ask a question about your audio files…")

    if question:
        if not st.session_state.kb_ready:
            st.warning(
                "⚠️ Upload and index at least one audio file before asking questions."
            )
        else:
            # Append and immediately render the user turn
            user_msg: dict = {"role": "user", "content": question, "sources": []}
            st.session_state.chat_history.append(user_msg)
            ui_components.render_chat_message(user_msg)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        pipeline = session.get_pipeline()
                        answer = pipeline.query(question)

                        cited = _cited_references(answer.summary, answer.references)

                        st.markdown(answer.summary)
                        ui_components.render_sources(cited)

                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": answer.summary,
                                "sources": cited,
                            }
                        )
                    except Exception as exc:
                        cfg = session.get_config()
                        err_str = str(exc).lower()
                        if "connection" in err_str or "connect" in err_str or "refused" in err_str:
                            ui_components.render_ollama_error(cfg.OLLAMA_BASE_URL)
                        else:
                            st.error(f"Error: {exc}")
                            with st.expander("Stack trace"):
                                st.code(traceback.format_exc())

# ── Transcriptions tab ────────────────────────────────────────────────────────
with tab_transcriptions:
    if not st.session_state.indexed_files:
        st.info(
            "No files indexed in this session.  \n"
            "Upload audio files using the sidebar, then click **Transcribe & Index**."
        )
    else:
        for f in st.session_state.indexed_files:
            header = (
                f"📄 {f['name']}  —  language: `{f['language']}`"
                f"  —  {f['chunks']} chunk(s)"
            )
            with st.expander(header, expanded=False):
                st.text_area(
                    "Full transcription",
                    value=f["text"],
                    height=220,
                    disabled=True,
                    key=f"transcript_{f['name']}",
                )
