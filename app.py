import streamlit as st
import os
import pandas as pd
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from src.ingestion import extract_text_from_pdf
from src.embedder_faiss import Chunking
from src.retrieval import Retriever
from src.generation import GenerationClient
from src.benchmark import run_benchmark
from src.config import FILE_PATH, GROQ_API_KEY

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="RAG Comparison: Vector vs Hybrid", layout="wide")


# Initialize components
@st.cache_resource
def get_retriever():
    return Retriever()


@st.cache_resource
def get_generator():
    return GenerationClient()


retriever = get_retriever()
generator = get_generator()
chunker = Chunking()

# Sidebar for API Status
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_status = "✅ Configured" if GROQ_API_KEY else "❌ Missing"

    
    st.write(f"**Groq API:** {groq_status}")

    if not GROQ_API_KEY:
        st.error("Please add missing API keys to your .env file.")

st.title("🚀 RAG Strategy Comparison")
st.markdown(
    """
This application compares two RAG retrieval techniques:
1. **Vector-Only**: semantic search using Google Gemini Embeddings.
2. **Hybrid**: BM25 keyword search followed by vector reranking.
"""
)

tabs = st.tabs(["📥 Ingestion", "🔍 Batch Query", "📊 Benchmark"])

# --- TAB 1: INGESTION ---
with tabs[0]:
    st.header("Document Ingestion")
    st.write("Upload PDF documents to populate the FAISS vector database.")

    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if not uploaded_files:
            st.error("Please upload at least one PDF file.")
        else:
            all_chunks = []
            with st.spinner("Extracting text and chunking..."):
                progress_bar = st.progress(0)
                for i, uploaded_file in enumerate(uploaded_files):
                    # Save temporary file
                    temp_path = os.path.join(FILE_PATH, uploaded_file.name)
                    os.makedirs(FILE_PATH, exist_ok=True)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Extract text
                    text = extract_text_from_pdf(temp_path)

                    # Semantic Chunking
                    chunks = chunker.semantic_chunking(text)
                    all_chunks.extend(chunks)

                    progress_bar.progress((i + 1) / len(uploaded_files))

            st.info(
                f"Extracted {len(all_chunks)} semantic chunks. Generating embeddings..."
            )

            with st.spinner("Generating embeddings and updating FAISS index..."):
                retriever.update_index(all_chunks)

            st.success("FAISS index updated successfully!")

# --- TAB 2: QUERY ---
with tabs[1]:
    st.header("Comparative Querying")
    query = st.text_input(
        "Enter your question:", placeholder="What happens at the event horizon?"
    )
    k = st.slider("Number of retrieved chunks (k):", 1, 10, 3)

    if st.button("Run Comparison"):
        if not query:
            st.warning("Please enter a query.")
        elif not retriever.chunks:
            st.error("Vector database is empty. Please ingest documents first.")
        else:
            col1, col2 = st.columns(2)

            # Retrieval functions
            def run_vector_rag():
                results = retriever.vector_search(query, k=k)
                context = []
                for res in results:
                    context.append(res["chunk"])
                response = generator.generate_response(query, context)
                return response, results

            def run_hybrid_rag():
                results = retriever.hybrid_search(query, k=k)
                context = []
                for res in results:
                    context.append(res["chunk"])
                response = generator.generate_response(query, context)
                return response, results

            # Execute in parallel
            with st.spinner("Generating responses in parallel..."):
                with ThreadPoolExecutor() as executor:
                    vec_future = executor.submit(run_vector_rag)
                    hyb_future = executor.submit(run_hybrid_rag)

                    vec_res, vec_chunks = vec_future.result()
                    hyb_res, hyb_chunks = hyb_future.result()

            with col1:
                st.subheader("🔵 Vector-Only RAG")
                st.markdown(vec_res)
                with st.expander("Show Retrieved Chunks"):
                    for i, res in enumerate(vec_chunks):
                        st.markdown(f"**Chunk {i+1} (Score: {res['score']:.4f})**")
                        st.text(res["chunk"])
                        st.divider()

            with col2:
                st.subheader("🟣 Hybrid RAG")
                st.markdown(hyb_res)
                with st.expander("Show Retrieved Chunks"):
                    for i, res in enumerate(hyb_chunks):
                        st.markdown(f"**Chunk {i+1} (Score: {res['score']:.4f})**")
                        st.text(res["chunk"])
                        st.divider()

# --- TAB 3: BENCHMARK ---
with tabs[2]:
    st.header("Performance Benchmarking")
    st.write(
        "Evaluate retrieval performance using 8 compliance-related queries based on policy and operations manuals. Metrics include Precision@k, Recall@k, and Latency."
    )

    if st.button("Run Benchmark Suites"):
        if not retriever.chunks:
            st.error("Vector database is empty. Please ingest documents first.")
        else:
            with st.spinner("Running 16 retrieval tests..."):
                benchmark_results = run_benchmark(retriever, k=3)

            # Prepare data for visualization
            flat_data = []
            for res in benchmark_results:
                flat_data.append(
                    {
                        "Query": res["query"],
                        "Method": "Vector",
                        "Precision": res["vector"]["precision"],
                        "Recall": res["vector"]["recall"],
                        "Latency (s)": res["vector"]["latency"],
                    }
                )
                flat_data.append(
                    {
                        "Query": res["query"],
                        "Method": "Hybrid",
                        "Precision": res["hybrid"]["precision"],
                        "Recall": res["hybrid"]["recall"],
                        "Latency (s)": res["hybrid"]["latency"],
                    }
                )

            df = pd.DataFrame(flat_data)

            # Summary Metrics
            st.subheader("Aggregate Performance")
            avg_df = (
                df.groupby("Method")[["Precision", "Recall", "Latency (s)"]]
                .mean()
                .reset_index()
            )
            st.table(avg_df)

            # Visualizations
            c1, c2 = st.columns(2)
            with c1:
                fig_prec = px.bar(
                    df,
                    x="Query",
                    y="Precision",
                    color="Method",
                    barmode="group",
                    title="Precision@k Comparison",
                )
                st.plotly_chart(fig_prec, width="stretch")

            with c2:
                fig_lat = px.line(
                    df,
                    x="Query",
                    y="Latency (s)",
                    color="Method",
                    markers=True,
                    title="Retrieval Latency (Query by Query)",
                )
                st.plotly_chart(fig_lat, width="stretch")

            st.subheader("Detailed Breakdown")
            st.dataframe(df)
