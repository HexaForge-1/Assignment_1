!pip -q install "numpy<2" openai==1.52.2 httpx==0.27.2 sentence-transformers==3.0.1 faiss-cpu==1.8.0 pypdf2==3.0.1 gradio==4.44.0

import os, json, re, time
from typing import List, Dict
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import gradio as gr
from google.colab import files
from openai import OpenAI

# --- Configuration ---
PROVIDER = "openai"
OPENAI_MODEL = "gpt-4o-mini"

try:
    from google.colab import userdata
    os.environ["OpenAi_Key"] = userdata.get('OpenAi_Key')
except:
    os.environ.setdefault("OpenAi_Key", "Error-...")

def get_llm_client_and_model():
    api_key = os.environ.get("OpenAi_Key", "")
    if not api_key or "Error-" not in api_key:
        raise RuntimeError("Missing or invalid OpenAi_Key. Please set it in Colab Secrets.")
    return OpenAI(api_key=api_key), OPENAI_MODEL

def call_llm(messages: List[Dict]) -> str:
    client, model_name = get_llm_client_and_model()
    resp = client.chat.completions.create(model=model_name, messages=messages, temperature=0.2)
    return resp.choices[0].message.content.strip()

# --- PDF Processing ---
PDF_NAME = "info.pdf"
if not Path(PDF_NAME).exists():
    print("Please upload 'info.pdf' to proceed.")
    uploaded = files.upload()
    if uploaded: PDF_NAME = list(uploaded.keys())[0]

# Extract text
reader = PdfReader(PDF_NAME)
pages_text = [p.extract_text() or "" for p in reader.pages]

# Simple Word-based Chunking
all_chunks = []
for i, text in enumerate(pages_text):
    words = text.split()
    for j in range(0, len(words), 250):
        chunk = " ".join(words[j : j + 300]) # 300 words with 50 word overlap
        if chunk.strip():
            all_chunks.append({"page": i + 1, "text": chunk})

# --- RAG Setup (Embeddings & Indexing) ---
print("Initializing RAG index...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts = [c["text"] for c in all_chunks]
embs = embedder.encode(texts, normalize_embeddings=True)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs.astype(np.float32))

def answer_question(question: str):
    # Retrieval step
    q_emb = embedder.encode([question], normalize_embeddings=True)
    D, I = index.search(q_emb.astype(np.float32), 5)
    retrieved_chunks = [all_chunks[idx] for idx in I[0] if idx != -1]
    
    # Generation step
    context = "\n\n".join([f"[Page {c['page']}] {c['text']}" for c in retrieved_chunks])
    prompt = f"Using the context below, answer the question.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"
    return call_llm([{"role": "user", "content": prompt}])

# --- Interface ---
with gr.Blocks() as demo:
    gr.Markdown("#@ Policy Copilot  @#")
    q_input = gr.Textbox(label="Ask a question about your policy document")
    answer_output = gr.Markdown(label="AI Response")
    btn = gr.Button("Ask AI")
    btn.click(answer_question, inputs=[q_input], outputs=[answer_output])

# Launching with stable Colab settings
demo.launch(share=True, inline=True)
