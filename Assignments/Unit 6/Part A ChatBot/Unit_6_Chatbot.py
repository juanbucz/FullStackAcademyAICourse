""" Essentials and Applications of Generative AI_ Unit End Projects - A - Chatbot

This Unit 6 End Project is STRONGLY based on the following demo from George
Most of the inplace comments still apply.
Any changes/derivations from the base demo are commented

RAG Knowledge System demo

This demo shows how to build a Retrieval-Augmented Generation (RAG) pipeline:
1. **Ingest** - load documents from a source, embed them, store in ChromaDB
2. **Query** - retrieve relevant chunks and pass them as context to an LLM

Architecture:
    Source → Ingestor → Embeddings → ChromaDB
                                        ↓
                          Question → Retriever → Context → Prompt → LLM → Answer

Usage:
    python demos/rag_system/rag_demo.py

Environment variables:
    None required with switch to using ChromaDB
"""

import os
import sys
import logging
from pathlib import Path

# Suppress noisy key warnings from sentence-transformers checkpoints
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import gradio as gr
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# Add src directory to path so relative ingestor imports work
sys.path.insert(0, str(Path(__file__).parent))

from ingestors import PDFIngestor

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

temperature = 0.1
collection_name = "nestles_hr_policy_documents"

# ---------------------------------------------------------------------------
# Embeddings (local, no API key required)
# ---------------------------------------------------------------------------

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Using ChromaDB instead of PostgreSQL - simpler/cleaner implementation, no standard DB variables required
# ChromaDB runs in process, storing data in local file on disk.
# ---------------------------------------------------------------------------

vector_store = Chroma(
                        collection_name=collection_name,
                        embedding_function=embeddings,
                        persist_directory="./chroma_db"
                    )

# ---------------------------------------------------------------------------
# LLM backends - For simplicity sake, use local ChatOllama and instantiate client
#                
# ---------------------------------------------------------------------------

ollama_model = "qwen2.5:3b"
ollama_client = ChatOllama(model=ollama_model, temperature=temperature)

huggingface_model = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
model = AutoModelForCausalLM.from_pretrained(huggingface_model)

hf_pipeline = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=512,
                        temperature=temperature,
                    )

# ---------------------------------------------------------------------------
# Ingestor registry
# ---------------------------------------------------------------------------

INGESTORS = {
    "PDF": PDFIngestor(),
}

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful human resources assistant. Answer the question using only the "
        "provided context. If the context does not contain enough information "
        "to answer, say so honestly.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])

QUESTION_PLACEHOLDER='e.g. What is the employee training policy?'

DEFAULT_ANSWER_PLACEHOLDER = 'Please enter a question. Make sure to ingest the Nestles HR Policy PDF first.'


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _format_sources(docs) -> str:

    sources = []

    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "Unknown")
        source = doc.metadata.get("source", "")
        preview = doc.page_content[:200].replace("\n", " ")
        sources.append(f"[{i}] {title}\n    {source}\n    \"{preview}...\"")

    return "\n\n".join(sources)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def ingest_documents(file_path: str) -> str:
    """Load documents from the selected source and store them in ChromaDB."""

    if not file_path.strip():
        return "Please enter a valid Nestles HR PDF Document"

    ingestor = INGESTORS['PDF']

    try:
        doc, chunks = ingestor.load(file_path.strip())

    except Exception as e:
        return f"Error loading document: {e}"

    if not doc:
        return "Document was not found or could not successfully loaded and chunked."

    try:
        vector_store.add_documents(chunks)

    except Exception as e:
        return f"Error storing documents: {e}"

    # PDF Ingestor/Loader returns different metrics than Wikipedia Ingestor/Loader
    # In single document mode, the metadata is accessible
    metadata = doc[0].metadata

    parts = ['']

    if metadata['producer']:
        parts.append(f"Producer: {metadata['producer']}")
    else:
        parts.append('Producer: (not extracted)')

    if metadata['creator']:
        parts.append(f"Creator: {metadata['creator']}")
    else:
        parts.append('Creator: (not extracted)')        

    if metadata['creationdate']:
        parts.append(f"Creation Date: {metadata['creationdate']}")
    else:
        parts.append('Creation Date: (not extracted)')  

    if metadata['moddate']:
        parts.append(f"Modification Date: {metadata['moddate']}")
    else:
        parts.append('Modification Date: (not extracted')  

    if metadata['source']:
        parts.append(f"Source: {metadata['source']}")
    else:
        parts.append('Source: (not extracted)')          

    if metadata['total_pages']:
        parts.append(f"Total Pages: {metadata['total_pages']}")
    else:
        parts.append('Total Pages: (not extracted)')          

    total_chars = len(doc[0].page_content)
    parts.append(f'Total Characters: {total_chars}')

    total_words = len(doc[0].page_content.split())
    parts.append(f'Total Words: {total_words}')

    total_lines = doc[0].page_content.count("\n")
    parts.append(f'Total Lines: {total_lines}')

    total_chunks = len(chunks)
    parts.append(f'Total Chunks: {total_chunks}')

    avg_chunk_chars = total_chars // total_chunks
    parts.append(f'Average Chunks Characters: {avg_chunk_chars}')

    extraction_summary = "\n".join(parts) 

    return (
        f"Source: Nestles HR PDF Document: | Path: {file_path}\n\n"
        f"Extracted metadata:\n{extraction_summary}"
    )


# Query Rag - backend_selector will give us either ollama or huggingface transformers
#
def query_rag(question: str, backend: str, k: int) -> tuple[str, str]:
    """Retrieve relevant chunks and generate a grounded answer."""

    if not question.strip():
        return DEFAULT_ANSWER_PLACEHOLDER, ""
    
    llm = ollama_client if backend == "Ollama" else HuggingFacePipeline(pipeline=hf_pipeline)

    retriever = vector_store.as_retriever(search_kwargs={"k": int(k)})

    # Fetch docs separately so we can display them as sources
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return (
            "No relevant documents found. Try ingesting some content first.",
            "",
        )

    chain = (
        {"context": lambda _: _format_docs(retrieved_docs), "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    try:
        answer = chain.invoke(question)
    
    except Exception as e:
        return f"Error generating answer: {e}", ""

    sources_text = _format_sources(retrieved_docs)

    return answer, sources_text


def clear_collection() -> tuple[str, any]:
    """Delete all documents from the vector store collection."""

    global vector_store

    try:
        vector_store.delete_collection()

        # Re-initialise so the store is ready for new ingestions
        vector_store = Chroma(
                                collection_name=collection_name,
                                embedding_function=embeddings,
                                persist_directory="./chroma_db"
                            )


        return "Collection cleared. Ready for new ingestion.", gr.update(value=None)

    except Exception as e:
        return f"Error clearing collection: {e}", gr.update(value=None)
    
def clear_question_and_results() -> tuple[str, str, str]:
    """Delete/Clear question, results and listed sources from form - Reset the form."""

    question_input.placeholder=QUESTION_PLACEHOLDER
    answer_output.placeholder=DEFAULT_ANSWER_PLACEHOLDER

    return '', '',  gr.update(value=None)
    

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Nestles HR Document RAG System") as demo:

    gr.Markdown("""
    # Nestles HR Document RAG Knowledge System

    Build a searchable knowledge base from PDF articles, then ask questions
    grounded in the ingested content.

    **How it works:**
    1. **Ingest** - "chunks" the document, embed and stores it on disk in ChromaDB
    2. **Query** - your question retrieves the most relevant chunks, which the LLM uses to answer
    """)

    with gr.Tabs():

        # ------------------------------------------------------------------
        # Tab 1: Ingest
        # ------------------------------------------------------------------
        with gr.Tab("1. Ingest documents"):
            ingest_instructions = gr.Markdown("""
            Upload the Nestles HR Policy PDF to build the knowledge base.
            The document will be split into chunks, embedded, and stored in ChromaDB for querying.
            """)

            with gr.Row():
                with gr.Column():

                    gr.Markdown("Select a Nestles HR Policy PDF document to ingest.")
                    document_source = gr.File(
                        label="Upload Nestles HR Policy PDF",
                        #value="./the_nestle_hr_policy_pdf_2012.pdf",   -- Preloading with value changes this "from open" to "file save"; go figure
                        file_count="single",
                        file_types=[".pdf"],
                        type="filepath"
                    )

                    with gr.Row():
                        ingest_btn = gr.Button("Ingest", variant="primary")
                        clear_btn = gr.Button("Clear collection", variant="stop")

                with gr.Column():
                    ingest_status = gr.Textbox(label="Status", lines=4, interactive=False)

            ingest_btn.click(
                fn=ingest_documents,
                inputs=[document_source],
                outputs=[ingest_status],
            )
            clear_btn.click(
                fn=clear_collection,
                inputs=[],
                outputs=[ingest_status, document_source],
            )

        # ------------------------------------------------------------------
        # Tab 2: Query
        # ------------------------------------------------------------------
        with gr.Tab("2. Query knowledge base"):
            gr.Markdown("""
            Ask a question. The system retrieves the most relevant chunks from the
            knowledge base and uses them as context for the answer.
            """)

            with gr.Row():
                # Choose between Ollama and HuggingFace Transformer
                backend_selector = gr.Radio(
                    choices=["Ollama", "HuggingFace Transformers"],
                    value="Ollama",
                    label="Model backend",
                    info=f"Ollama: {ollama_model} | HuggingFace Transformers: {huggingface_model}",
                )
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Chunks to retrieve (k)",
                    info="More chunks = more context, but slower and more expensive.",
                )
                
            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder=QUESTION_PLACEHOLDER,
                        lines=3,
                    )
                    ask_btn = gr.Button("Ask", variant="primary")
                    clear_question_btn = gr.Button("Clear Question and Results", variant="stop")

                with gr.Column():
                    answer_output = gr.Textbox(
                        label="Answer",
                        placeholder=DEFAULT_ANSWER_PLACEHOLDER,
                        interactive=False)

            with gr.Accordion("Sources", open=False):
                sources_output = gr.Textbox(label="Retrieved chunks", lines=12, interactive=False)

            ask_btn.click(
                fn=query_rag,
                inputs=[question_input, backend_selector, k_slider],
                outputs=[answer_output, sources_output],
            )

            clear_question_btn.click(
                fn=clear_question_and_results,
                inputs=[],
                outputs=[question_input, answer_output, sources_output],
            )            

    gr.Markdown("""
    ---

    ## Key concepts

    | Component | Role |
    |-----------|------|
    | `HuggingFaceEmbeddings` | Turns text into vectors (numbers) that capture meaning |
    | `ChromaDB` | Stores vectors locally on disk; finds nearest neighbours fast |
    | `PDFIngestor` | Loads + chunks a PDF document |
    | Retriever | Finds the *k* most similar chunks to the question |
    | RAG chain | Injects retrieved chunks as context into the LLM prompt |

    """)


if __name__ == "__main__":
    demo.launch()
