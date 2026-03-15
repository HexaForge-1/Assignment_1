!pip -q install "numpy<2" openai==1.52.2 httpx==0.27.2 sentence-transformers==3.0.1 faiss-cpu==1.8.0 pypdf2==3.0.1 gradio==4.44.0

import os, json, re, textwrap
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

import gradio as gr
from google.colab import files
from openai import OpenAI

# 1) Configuration (I used OpenAI only)
PROVIDER = "openai"
OPENAI_MODEL = "gpt-4o-mini"

# I Prefered Colab > Secrets > OpenAi_Key; else allow pasting via UI below.
try:
    from google.colab import userdata
    os.environ["OpenAi_Key"] = userdata.get('OpenAi_Key')
except Exception:
    os.environ.setdefault("OpenAi_Key", "sk-...")  # placeholder

def get_llm_client_and_model():
    api_key = os.environ.get("OpenAi_Key", "")
    if not api_key or not api_key.strip() or (api_key.startswith("sk-") is False):
        raise RuntimeError("Missing or invalid OpenAi_Key. Set it in Colab Secrets or paste it in 'LLM Settings'.")
    return OpenAI(api_key=api_key), OPENAI_MODEL

def call_llm(messages: List[Dict]) -> str:
    client, model_name = get_llm_client_and_model()
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=900
    )
    return resp.choices[0].message.content.strip()

# 2) PDF Processing
PDF_NAME = "info.pdf"
if not Path(PDF_NAME).exists():
    print("Please upload 'info.pdf' to proceed.")
    uploaded = files.upload()
    if uploaded: PDF_NAME = list(uploaded.keys())[0]
assert Path(PDF_NAME).exists(), f"{PDF_NAME} not found."

def light_cleanup(text: str) -> str:
    t = (text or "").replace("\x00", " ")
    t = re.sub(r"[=_]{3,}", " ", t)       # strip header/line artifacts
    t = re.sub(r"[ \t]{2,}", " ", t)      # compress spaces
    t = t.replace("", "•").strip()       # normalize bullets
    return t

reader = PdfReader(PDF_NAME)
pages_text = []
for p in reader.pages:
    raw = p.extract_text() or ""
    cleaned = light_cleanup(raw)
    if cleaned:
        pages_text.append(cleaned)

if not pages_text:
    raise RuntimeError("No extractable text found.")

# 3) Chunking
all_chunks: List[Dict] = []
for i, text in enumerate(pages_text):
    words = text.split()
    for j in range(0, len(words), 250):
        chunk = " ".join(words[j : j + 300])
        if chunk.strip():
            all_chunks.append({"page": i + 1, "text": chunk})

if not all_chunks:
    raise RuntimeError("No chunks created - check PDF extraction.")

# 4) RAG Setup (Embeddings & Index)
print("Initializing RAG index...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts = [c["text"] for c in all_chunks]
embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs.astype(np.float32))

def _retrieve(query: str, k: int = 6) -> List[Dict]:
    q_emb = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb.astype(np.float32), k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1: 
            continue
        hits.append({**all_chunks[idx], "score": float(score)})
    return hits

# 5) Prompt Builders (adds guardrails, sources, and a Pre-check flow)
SYSTEM_BASE = """You are **Policy & Claims Copilot** for health insurance.
Only answer using the supplied CONTEXT. If the info is not in CONTEXT, say you don't have enough information.
Be precise. Where applicable, list: coverage/not covered, limits & sub-limits, waiting periods, co-pay/deductible,
required documents, and claim submission steps & timelines. Always mention page numbers. Use clear, simple English."""

def _make_context(snips: List[Dict]) -> str:
    blocks = []
    for s in snips:
        blocks.append(f"[page {s['page']}] {s['text'][:1500]}")  # trim for token safety
    return "\n\n".join(blocks)

def _build_qa_messages(user_q: str, snips: List[Dict]) -> List[Dict]:
    context = _make_context(snips)
    user_prompt = f"""CONTEXT:
{context}

QUESTION:
{user_q}

INSTRUCTIONS:
- Answer strictly from CONTEXT.
- If relevant, include: coverage/not covered, limits & sub-limits, waiting periods, co-pay/deductible,
  required documents, and claim submission steps & timelines.
- Add a short 'Why' section with brief quotes and page numbers.
- End with 'Sources: page X, page Y'.
"""
    return [{"role": "system", "content": SYSTEM_BASE},
            {"role": "user", "content": user_prompt}]

def _build_precheck_messages(scenario: str, snips: List[Dict]) -> List[Dict]:
    context = _make_context(snips)
    user_prompt = f"""You will assess a CLAIM SCENARIO against the policy in CONTEXT.

CONTEXT:
{context}

CLAIM SCENARIO:
{scenario}

TASK:
Return JSON with keys:
["coverage_likelihood","why","coverage_scope","limits_and_sublimits","waiting_periods",
 "co_pay_or_deductibles","exclusions_that_might_apply","required_documents","steps_and_timelines","sources"]

Rules:
1) coverage_likelihood: one of ["Likely Covered","Unclear/Need More Info","Likely Not Covered"].
2) In 'why', quote exact phrases with page numbers.
3) For each field, fill details from CONTEXT; if not present, say "Not found in provided policy context".
4) Keep it concise and factual.
"""
    return [{"role": "system", "content": SYSTEM_BASE},
            {"role": "user", "content": user_prompt}]

# 6) Q&A and Pre‑check
def answer_question(question: str):
    if not question or not question.strip():
        return "Please enter a question."
    retrieved_chunks = _retrieve(question, k=6)
    if not retrieved_chunks:
        return "I couldn't find information about that in the policy."
    messages = _build_qa_messages(question, retrieved_chunks)
    return call_llm(messages)

def precheck_claim(scenario: str) -> Tuple[str, str]:
    if not scenario or not scenario.strip():
        return "Please describe the claim scenario.", ""
    snips = _retrieve(scenario, k=8)
    if not snips:
        return "I couldn't find matching clauses for this scenario in the policy.", ""
    raw = call_llm(_build_precheck_messages(scenario, snips))
    # Robust JSON parsing
    try:
        jstart, jend = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[jstart:jend+1])
    except Exception:
        data = {"coverage_likelihood":"Unclear/Need More Info",
                "why": f"Could not parse JSON. Raw model output:\n{raw}",
                "coverage_scope":"", "limits_and_sublimits":"", "waiting_periods":"",
                "co_pay_or_deductibles":"", "exclusions_that_might_apply":"",
                "required_documents":"", "steps_and_timelines":"", "sources":[]}

    # Pretty print + sources
    pretty = []
    pretty.append(f"**Coverage likelihood:** {data.get('coverage_likelihood','')}")
    pretty.append(f"\n**Why (quotes & pages):**\n{data.get('why','')}")
    pretty.append(f"\n**Coverage scope:**\n{data.get('coverage_scope','')}")
    pretty.append(f"\n**Limits & Sub-limits:**\n{data.get('limits_and_sublimits','')}")
    pretty.append(f"\n**Waiting periods:**\n{data.get('waiting_periods','')}")
    pretty.append(f"\n**Co-pay/Deductibles:**\n{data.get('co_pay_or_deductibles','')}")
    pretty.append(f"\n**Exclusions that might apply:**\n{data.get('exclusions_that_might_apply','')}")
    pretty.append(f"\n**Documents needed:**\n{data.get('required_documents','')}")
    pretty.append(f"\n**Steps & Timelines:**\n{data.get('steps_and_timelines','')}")
    sources = data.get("sources", [])
    src_md = "• " + "\n• ".join(sources) if sources else ""
    return "\n".join(pretty), (f"**Sources used:**\n{src_md}" if src_md else "")

# 7) Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("#@ Policy Copilot (OpenAI) #@")

    with gr.Accordion("LLM Settings (OpenAI)", open=True):
        key_box = gr.Textbox(label="OpenAi_Key (not stored)", value="", type="password",
                             placeholder="Paste your OpenAI API key here")
        status = gr.Markdown()
        def _apply_key(k):
            if k: os.environ["OpenAi_Key"] = k.strip()
            return " Key applied." if os.environ.get("OpenAi_Key","").startswith("sk-") else " Please paste a valid key."
        gr.Button("Apply Key").click(_apply_key, [key_box], [status])

    with gr.Tab("Ask a Question"):
        q_input = gr.Textbox(label="Ask a question about your policy document",
                             placeholder="e.g., Is cataract covered? Any sub-limits or waiting period?")
        answer_output = gr.Markdown(label="AI Response")
        gr.Button("Ask AI").click(answer_question, inputs=[q_input], outputs=[answer_output])

    with gr.Tab("Claim Pre-check"):
        scenario_input = gr.Textbox(label="Describe the claim scenario", lines=6,
                                    placeholder="Example: My father (62) had cataract surgery in a non-network hospital. What is covered, sub-limits, co-pay, docs and timelines?")
        pre_out = gr.Markdown()
        src_out = gr.Markdown()
        gr.Button("Run Pre-check").click(precheck_claim, inputs=[scenario_input], outputs=[pre_out, src_out])

# Launching with stable Colab settings
demo.launch(share=True, inline=True)
