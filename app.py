"""
app.py
Run with: streamlit run app.py

Semantic search over user research interviews.
Retrieves relevant chunks via ChromaDB + synthesizes insight via Groq (Llama 3).
"""

import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

COLLECTION_NAME = "interviews"
TOP_K = 4  # number of chunks to retrieve per query

st.set_page_config(
    page_title="Research Search",
    layout="centered",
)

st.markdown("""
<style>
    .main { max-width: 760px; margin: auto; }
    .chunk-card {
        background: #f8f9fa;
        border-left: 3px solid #4f8ef7;
        padding: 14px 18px;
        border-radius: 4px;
        margin-bottom: 12px;
        font-size: 0.92rem;
        line-height: 1.6;
        color: #1a1a1a;
    }
    .source-label {
        font-size: 0.75rem;
        color: #888;
        margin-bottom: 6px;
        font-family: monospace;
    }
    .insight-box {
        background: #eef4ff;
        border: 1px solid #c3d8ff;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 24px;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached resource loading ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection(COLLECTION_NAME)


@st.cache_resource
def load_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to your .env file.")
        st.stop()
    return Groq(api_key=api_key)


def search(query: str, top_k: int = TOP_K):
    """Embed query and retrieve top_k most relevant chunks from ChromaDB."""
    model = load_model()
    collection = load_collection()
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    scores = [round(1 - d, 3) for d in results["distances"][0]]  # cosine similarity
    return list(zip(chunks, sources, scores))


def synthesize(query: str, chunks: list[tuple]) -> str:
    """Send retrieved chunks to Groq/Llama and ask for a synthesized insight."""
    groq = load_groq_client()

    context = "\n\n---\n\n".join(
        f"[Source: {src}]\n{chunk}" for chunk, src, _ in chunks
    )

    prompt = f"""You are analyzing user research interview excerpts.

Question: {query}

Relevant interview excerpts:
{context}

Synthesize the key insight in 3-4 sentences. Be specific — reference what multiple users said, note any patterns or contradictions. Do not pad. Do not say "based on the excerpts provided."
"""

    response = groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-left: 3px solid #4f8ef7; padding: 18px 24px; margin-bottom: 32px; background: #f8faff; border-radius: 0 8px 8px 0;">

  <p style="font-size: 0.72rem; font-family: monospace; color: #888; margin: 0 0 8px 0; letter-spacing: 0.08em; text-transform: uppercase;">
    Application Demo &nbsp;·&nbsp; Great Question AI Engineering Intern
  </p>

  <h1 style="font-size: 1.5rem; font-weight: 650; color: #0f1117; margin: 0 0 10px 0; line-height: 1.3;">
    Interview Research Search Engine
  </h1>

  <p style="font-size: 0.93rem; color: #444; line-height: 1.7; margin: 0 0 16px 0;">
    Semantic search over user research interview transcripts — type a concept or question,
    not just a keyword, and the system retrieves the most relevant moments across all interviews
    and synthesizes a key insight. Built in response to Great Question's stated engineering challenge:
    <em>semantic search across tens of thousands of interview hours.</em>
  </p>

  <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;">
    <span style="background: #e8f0fe; color: #1a56db; font-size: 0.75rem; padding: 3px 10px; border-radius: 20px; font-family: monospace;">ChromaDB</span>
    <span style="background: #e8f0fe; color: #1a56db; font-size: 0.75rem; padding: 3px 10px; border-radius: 20px; font-family: monospace;">sentence-transformers</span>
    <span style="background: #e8f0fe; color: #1a56db; font-size: 0.75rem; padding: 3px 10px; border-radius: 20px; font-family: monospace;">all-MiniLM-L6-v2</span>
    <span style="background: #e8f0fe; color: #1a56db; font-size: 0.75rem; padding: 3px 10px; border-radius: 20px; font-family: monospace;">Groq · Llama 3.1</span>
    <span style="background: #e8f0fe; color: #1a56db; font-size: 0.75rem; padding: 3px 10px; border-radius: 20px; font-family: monospace;">Streamlit</span>
    <span style="background: #e8f0fe; color: #1a56db; font-size: 0.75rem; padding: 3px 10px; border-radius: 20px; font-family: monospace;">Python</span>
  </div>

  <a href="https://github.com/garvps/gq_research_demo" target="_blank"
     style="font-size: 0.82rem; color: #4f8ef7; text-decoration: none; font-family: monospace;">
    &#8599; github.com/garvps/gq_research_demo
  </a>

</div>

<p style="font-size: 0.88rem; color: #666; margin-bottom: 6px;">
  Try: <em>"what did users say about onboarding?"</em> &nbsp;·&nbsp; <em>"pricing concerns"</em> &nbsp;·&nbsp; <em>"AI trust issues"</em>
</p>
""", unsafe_allow_html=True)

query = st.text_input(
    label="Your question",
    placeholder="e.g. What did users say about onboarding?",
    label_visibility="collapsed",
)

if query:
    with st.spinner("Searching interviews..."):
        try:
            results = search(query)
        except Exception as e:
            st.error(f"ChromaDB error: {e}. Have you run `python ingest.py` yet?")
            st.stop()

    with st.spinner("Synthesizing insight..."):
        insight = synthesize(query, results)

    # Synthesized insight
    st.markdown("### Key Insight")
    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    # Source clips
    st.markdown("### Relevant Clips")
    for chunk, source, score in results:
        source_display = source.replace("_", " ").replace(".txt", "").title()
        st.markdown(
            f'<div class="chunk-card">'
            f'<div class="source-label">{source_display} &nbsp;·&nbsp; relevance: {score}</div>'
            f'{chunk}'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption("Built with ChromaDB · sentence-transformers · Groq Llama 3 · Streamlit")