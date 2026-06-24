"""Reusable Streamlit UI components for the audio-rag demo."""
from __future__ import annotations

import streamlit as st


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ── Sidebar components ────────────────────────────────────────────────────────

def render_kb_status(kb_ready: bool, session_file_count: int) -> None:
    """Show a colour-coded knowledge base status indicator in the sidebar."""
    if kb_ready:
        label = (
            f"🟢 {session_file_count} file{'s' if session_file_count != 1 else ''} indexed"
            if session_file_count > 0
            else "🟢 Data available (from previous session)"
        )
        st.sidebar.success(label)
    else:
        st.sidebar.error("🔴 Knowledge base empty")


def render_indexed_files_sidebar(indexed_files: list[dict]) -> None:
    """Show a collapsible list of files indexed in the current session."""
    if not indexed_files:
        return
    with st.sidebar.expander(
        f"Indexed this session ({len(indexed_files)})", expanded=False
    ):
        for f in indexed_files:
            st.caption(
                f"📄 **{f['name']}**  \n"
                f"Language: `{f['language']}` · {f['chunks']} chunks · {f['indexed_at']}"
            )


# ── Chat components ───────────────────────────────────────────────────────────

def render_sources(sources: list) -> None:
    """Render a list of Reference objects as a collapsible expander."""
    if not sources:
        return
    with st.expander("📎 Sources used"):
        for ref in sources:
            indices = ", ".join(str(i) for i in ref.chunk_indices)
            start = _fmt_time(ref.start_time)
            end = _fmt_time(ref.end_time)
            st.markdown(
                f"└─ `{ref.file_name}` — chunk {indices} ({start} – {end})"
            )


def render_chat_message(msg: dict) -> None:
    """Render a single chat message; assistant messages include their sources."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])


def render_ollama_error(base_url: str) -> None:
    st.error(
        f"🔌 Ollama not reachable at `{base_url}`.  \n"
        "Make sure Ollama is running on the host with the required model pulled."
    )
