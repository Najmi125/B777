import streamlit as st
import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------- 1. CONFIGURATION ----------------
st.set_page_config(page_title="B777 DDG Assistant", layout="centered", page_icon="✈️")

# Hide Profile Photo & Footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

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
- You are strictly a formatting engine. 
- You must output the metadata and the EXACT text provided in the context.
- DERIVE the MEL Item from the DDG Item number by removing the prefix (e.g., "2.") and suffix. 
  (Example: If DDG Item is "2.46-11-02.2", the MEL Item is "46-11-02").
- Do NOT summarize the "Page Text". Output it exactly as provided in the context.
"""

REQUIRED_OUTPUT_FORMAT = """
**REFERENCES:**
* DDG Item: <item number>
* ATA: <ata chapter>
* MEL Item: <item number (e.g. 46-11-02)>

**Page text**
<Insert the exact text content from the context here. Do not summarize.>
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

def build_context(results):
    blocks = []
    for item in results:
        # Tries to find the text. If missing, shows a placeholder.
        raw_text = item.get('text') or item.get('content') or "[[TEXT MISSING - UPDATE JSON]]"
        
        block = f"""
DDG ITEM: {item.get('item_full', 'N/A')}
ATA: {item.get('ata', 'N/A')}
Title: {item.get('title', 'N/A')}
TEXT CONTENT:
{raw_text}
"""
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)

# ---------------- 4. UI LOGIC ----------------
st.title("✈️ B777 DDG Assistant")

# --- UPDATED TEXT BLOCK ---
st.markdown("""
Find quick reference to DDG item/page.

**CAUTION**: For more accurate results, input text should be as close to DDG language as possible.  
*e.g. FCAC Flow Regulating Valve | Autothrottle Servo Motors*

**NOTE**: This is a prototype app for limited use.  
If you find it slow or see a 'Rate limit' error,  
please WhatsApp +92 337 1244809.
""")
# --------------------------

# Load backend
embed_model, index, metadatas, client = load_backend()

# Input section
query = st.text_input("Enter Pilot Discrepancy:", 
                      placeholder="E.g. Forward cargo air conditioning exhaust fan inoperative",
                      key="query_input")

col1, col2 = st.columns([1, 1])
with col1:
    search_clicked = st.button("click to search DDG", type="primary", use_container_width=True)
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
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                temperature=0.0,
                max_tokens=800
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
