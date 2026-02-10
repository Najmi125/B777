import streamlit as st
import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------- 1. CONFIGURATION ----------------
st.set_page_config(page_title="B777 DDG Assistant", layout="centered", page_icon="✈️")

# File Safety Check
required_files = ["ddg_faiss.index", "ddg_metadata.json"]
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"❌ Missing required files: {', '.join(missing_files)}")
    st.info("Please ensure these files are uploaded to your GitHub repository.")
    st.stop()

# Initialize Session State
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

# ---------------- 2. PROMPTS ----------------
SYSTEM_PROMPT = """You are a Boeing 777 Dispatch Deviation Guide (DDG) assistant.
Rules:
- Answer ONLY using the provided DDG excerpts.
- Do NOT invent procedures.
- Always include references (DDG item number, ATA, page numbers).
- Use clear, operational aviation language."""

REQUIRED_OUTPUT_FORMAT = """
**REFERENCES:**
* DDG Item: <item number>
* Pages: <page range>

**DESCRIPTION:**
<Plain-English dispatch explanation>
"""

# ---------------- 3. BACKEND ----------------
@st.cache_resource
def load_backend():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ Groq API key not found. Check Streamlit Secrets.")
        st.stop()
    
    try:
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        index = faiss.read_index("ddg_faiss.index")
        with open("ddg_metadata.json", "r", encoding="utf-8") as f:
            metadatas = json.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load backend: {e}")
        st.stop()
    
    client = Groq(api_key=api_key)
    return embed_model, index, metadatas, client

# --- ⚠️ THIS IS THE FIXED FUNCTION ---
def build_context(results):
    blocks = []
    for i, item in enumerate(results):
        # Fallback logic: Try to find the text field
        # It checks for 'text', then 'content', then 'description', etc.
        text_content = item.get('text') or item.get('content') or item.get('page_content') or item.get('description')
        
        # If still None, print the keys to the screen so we can debug!
        if text_content is None:
            st.error(f"❌ Error in Item {i}: Could not find text field.")
            st.write("Available keys in your data:", item.keys())
            text_content = "DATA ERROR - NO TEXT FOUND"

        block = f"DDG ITEM: {item.get('item_full', 'N/A')}\nATA: {item.get('ata', 'N/A')}\nText: {text_content}"
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)
# -------------------------------------

# ---------------- 4. UI LOGIC ----------------
st.title("✈️ B777 DDG Assistant")
st.warning("⚠️ **SIMULATION ONLY.** DO NOT USE FOR REAL FLIGHT OPERATIONS.")

# Load backend
embed_model, index, metadatas, client = load_backend()

# Input section
query = st.text_input("Enter Pilot Discrepancy:", 
                      placeholder="E.g. Forward cargo air conditioning exhaust fan inoperative",
                      key="query_input")

col1, col2 = st.columns([1, 1])
with col1:
    search_clicked = st.button("Search Manuals", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

# Clear Logic
if clear_clicked:
    st.session_state.query_input = "" 
    st.session_state.processed = False
    st.rerun()

# Search Logic
if search_clicked and query:
    with st.spinner("Searching DDG..."):
        # Search
        q_emb = embed_model.encode([query], normalize_embeddings=True)
        scores, indices = index.search(q_emb, k=3)
        results = [metadatas[idx] for idx in indices[0]]
        context_text = build_context(results)
        
        # Prompt
        USER_PROMPT = f"""Pilot discrepancy: "{query}"
        
Context:
{context_text}

Answer using the REQUIRED OUTPUT FORMAT:
{REQUIRED_OUTPUT_FORMAT}"""
        
        try:
            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            st.markdown("### ✅ Dispatch Guidance")
            st.markdown(completion.choices[0].message.content)
            
            with st.expander("Show Retrieved Context"):
                st.text(context_text)
                
            st.session_state.processed = True
            
        except Exception as e:
            st.error(f"API Error: {e}")

elif search_clicked and not query:
    st.warning("Please enter a discrepancy.")
