from __future__ import annotations

import html
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 120

COLLECTION_OPTIONS = [
    "trustdoc_os_paragraph",
    "trustdoc_os_fixed",
]

CHUNKS_PATH_OPTIONS = [
    "data/processed/chunks_paragraph_all.json",
    "data/processed/chunks_fixed_all.json",
]

EXAMPLE_QUESTIONS = [
    "What is demand paging?",
    "What are first-fit and best-fit in contiguous allocation?",
    "What are the four necessary conditions for deadlock?",
    "What is TLB reach and why does it matter?",
]

SOURCE_TABLE_COLUMNS = [
    "rank",
    "source_file",
    "page_number",
    "chunk_id",
    "distance",
    "dense_rank",
    "bm25_rank",
    "hybrid_score",
    "rerank_score",
]


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --trustdoc-bg: #f7f8fb;
            --trustdoc-card: #ffffff;
            --trustdoc-border: #e4e7ee;
            --trustdoc-muted: #667085;
            --trustdoc-text: #182230;
            --trustdoc-accent: #2563eb;
        }

        .stApp {
            background: var(--trustdoc-bg);
            color: var(--trustdoc-text);
        }

        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--trustdoc-border);
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        .trustdoc-kicker {
            color: var(--trustdoc-accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .trustdoc-subtitle {
            color: var(--trustdoc-muted);
            font-size: 1.05rem;
            margin-top: -0.4rem;
            margin-bottom: 1.35rem;
        }

        .trustdoc-card {
            background: var(--trustdoc-card);
            border: 1px solid var(--trustdoc-border);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 0.75rem 0 1.2rem;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }

        .trustdoc-card-title {
            color: var(--trustdoc-text);
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .trustdoc-card-body {
            color: var(--trustdoc-muted);
            font-size: 0.95rem;
            line-height: 1.55;
        }

        .trustdoc-answer {
            color: var(--trustdoc-text);
            font-size: 1rem;
            line-height: 1.65;
            white-space: pre-wrap;
        }

        .trustdoc-meta {
            color: var(--trustdoc-muted);
            font-size: 0.86rem;
            line-height: 1.5;
        }

        .trustdoc-footer {
            border-top: 1px solid var(--trustdoc-border);
            color: var(--trustdoc-muted);
            font-size: 0.84rem;
            margin-top: 2rem;
            padding-top: 1rem;
            text-align: center;
        }

        div[data-testid="stButton"] > button {
            border: 1px solid var(--trustdoc-border);
            border-radius: 8px;
            background: #ffffff;
            color: var(--trustdoc-text);
            font-weight: 600;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: var(--trustdoc-accent);
            color: var(--trustdoc-accent);
        }

        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 8px;
            background: var(--trustdoc-accent);
            border-color: var(--trustdoc-accent);
            color: #ffffff;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_payload(
    query: str,
    collection: str,
    chunks_path: str,
    retrieval_mode: str,
    top_k: int,
    initial_top_k: int,
    dense_top_k: int,
    bm25_top_k: int,
    use_rerank: bool,
) -> Dict[str, Any]:
    return {
        "query": query.strip(),
        "collection": collection,
        "chunks_path": chunks_path,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "initial_top_k": initial_top_k,
        "dense_top_k": dense_top_k,
        "bm25_top_k": bm25_top_k,
        "use_rerank": use_rerank,
    }


def call_api(
    api_base_url: str,
    endpoint: str,
    payload: Dict[str, Any],
    timeout_seconds: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    url = f"{api_base_url.rstrip('/')}{endpoint}"

    try:
        response = requests.post(url, json=payload, timeout=timeout_seconds)
    except requests.exceptions.ConnectionError:
        return (
            None,
            "FastAPI backend is not reachable. Start it with: "
            "PYTHONPATH=. python -m uvicorn app.api.main:app --reload",
        )
    except requests.exceptions.Timeout:
        return None, f"Request timed out after {timeout_seconds} seconds."
    except requests.exceptions.RequestException as exc:
        return None, f"Request failed: {exc}"

    if not response.ok:
        response_text = response.text.strip()
        if response_text:
            return None, f"Backend returned HTTP {response.status_code}: {response_text}"
        return None, f"Backend returned HTTP {response.status_code}."

    try:
        return response.json(), None
    except ValueError:
        return None, "Backend returned a non-JSON response."


def render_answer(answer: str) -> None:
    safe_answer = html.escape(answer or "No answer returned.")
    st.markdown(
        f"""
        <div class="trustdoc-card">
            <div class="trustdoc-card-title">Answer</div>
            <div class="trustdoc-answer">{safe_answer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_score(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 5)
    return value if value is not None else ""


def source_rows(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for source in sources:
        rows.append(
            {
                column: normalize_score(source.get(column))
                for column in SOURCE_TABLE_COLUMNS
            }
        )
    return rows


def render_sources_table(sources: List[Dict[str, Any]]) -> None:
    st.subheader("Retrieved Sources")

    if not sources:
        st.info("No retrieved sources returned.")
        return

    st.dataframe(
        source_rows(sources),
        use_container_width=True,
        hide_index=True,
    )


def render_source_cards(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return

    for index, source in enumerate(sources, start=1):
        rank = source.get("rank", index)
        source_file = source.get("source_file") or "Unknown file"
        page_number = source.get("page_number")
        chunk_id = source.get("chunk_id") or "Unknown chunk"
        label = f"Source {rank} - {source_file}"
        if page_number is not None:
            label = f"{label}, page {page_number}"

        with st.expander(label):
            optional_metadata = {
                "distance": source.get("distance"),
                "dense_rank": source.get("dense_rank"),
                "bm25_rank": source.get("bm25_rank"),
                "hybrid_score": source.get("hybrid_score"),
                "rerank_score": source.get("rerank_score"),
            }
            metadata_text = " | ".join(
                f"{key}: {normalize_score(value)}"
                for key, value in optional_metadata.items()
                if value is not None
            )
            st.markdown(
                f"""
                <div class="trustdoc-meta">
                    <strong>Rank:</strong> {html.escape(str(rank))}<br>
                    <strong>File:</strong> {html.escape(str(source_file))}<br>
                    <strong>Page:</strong> {html.escape(str(page_number or ""))}<br>
                    <strong>Chunk ID:</strong> {html.escape(str(chunk_id))}<br>
                    <strong>Scores:</strong> {html.escape(metadata_text or "No optional scores returned.")}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("**Preview**")
            st.write(source.get("text_preview") or "No preview returned.")


def render_footer() -> None:
    st.markdown(
        """
        <div class="trustdoc-footer">
            TrustDoc RAG — Dense Retrieval · BM25 Hybrid · CrossEncoder Reranking · FastAPI Serving
        </div>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    if "question" not in st.session_state:
        st.session_state.question = ""


def main() -> None:
    st.set_page_config(
        page_title="TrustDoc RAG",
        page_icon="TD",
        layout="wide",
    )
    apply_styles()
    initialize_state()

    with st.sidebar:
        st.header("Settings")
        api_base_url = st.text_input("API Base URL", value=DEFAULT_API_BASE_URL)
        endpoint_mode = st.radio(
            "Endpoint mode",
            options=["RAG Answer", "Retrieve Only"],
            index=0,
        )
        retrieval_mode = st.selectbox(
            "Retrieval mode",
            options=["dense", "hybrid"],
            index=1,
        )
        use_rerank = st.checkbox("Use rerank", value=True)
        collection = st.selectbox(
            "Collection",
            options=COLLECTION_OPTIONS,
            index=0,
        )
        chunks_path = st.selectbox(
            "Chunks path",
            options=CHUNKS_PATH_OPTIONS,
            index=0,
        )

        st.divider()
        top_k = st.slider("top_k", min_value=1, max_value=10, value=5)
        initial_top_k = st.slider(
            "initial_top_k",
            min_value=1,
            max_value=30,
            value=10,
        )
        dense_top_k = st.slider(
            "dense_top_k",
            min_value=1,
            max_value=30,
            value=10,
        )
        bm25_top_k = st.slider(
            "bm25_top_k",
            min_value=1,
            max_value=30,
            value=10,
        )
        timeout_seconds = st.number_input(
            "Request timeout seconds",
            min_value=5,
            max_value=300,
            value=DEFAULT_TIMEOUT_SECONDS,
            step=5,
        )

    st.markdown('<div class="trustdoc-kicker">Developer RAG Demo</div>', unsafe_allow_html=True)
    st.title("TrustDoc RAG")
    st.markdown(
        '<div class="trustdoc-subtitle">PDF-based RAG demo with dense retrieval, hybrid retrieval, and reranking.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="trustdoc-card">
            <div class="trustdoc-card-title">System overview</div>
            <div class="trustdoc-card-body">
                Ask questions over indexed PDF lecture documents and inspect the retrieved evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Examples**")
    example_columns = st.columns(2)
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        with example_columns[index % 2]:
            if st.button(question, use_container_width=True):
                st.session_state.question = question

    with st.form("rag_query_form"):
        query = st.text_area(
            "Question",
            key="question",
            height=120,
            placeholder="Ask a question about the indexed operating systems lecture PDFs.",
        )
        submitted = st.form_submit_button("Submit", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("Enter a question before submitting.")
        else:
            endpoint = "/rag/query" if endpoint_mode == "RAG Answer" else "/rag/retrieve"
            payload = build_payload(
                query=query,
                collection=collection,
                chunks_path=chunks_path,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                initial_top_k=initial_top_k,
                dense_top_k=dense_top_k,
                bm25_top_k=bm25_top_k,
                use_rerank=use_rerank,
            )

            with st.spinner("Calling FastAPI backend..."):
                data, error = call_api(
                    api_base_url=api_base_url,
                    endpoint=endpoint,
                    payload=payload,
                    timeout_seconds=int(timeout_seconds),
                )

            if error:
                st.error(error)
            elif data:
                sources = data.get("retrieved_sources", [])
                if endpoint_mode == "RAG Answer":
                    render_answer(data.get("answer", ""))
                render_sources_table(sources)
                render_source_cards(sources)

                with st.expander("Raw API response"):
                    st.json(data)

    render_footer()


if __name__ == "__main__":
    main()
