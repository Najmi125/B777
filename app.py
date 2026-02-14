import streamlit as st
import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------- 1. CONFIGURATION ----------------
st.set_page_config(page_title="B777 DDG Assistant", layout="centered", page_icon="✈️")

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
    st.stop()

if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

# ---------------- 2. PROMPTS ----------------
SYSTEM_PROMPT = """You are a Boeing 777 Dispatch Deviation Guide (DDG) assistant.
Rules:
- You are strictly a formatting engine for metadata.
- DERIVE the MEL Item from the DDG Item number by removing the prefix (e.g., "2.") and suffix.
- Do NOT output the page text. Only output the References.
"""

# ⚠️ UPDATED FORMAT
REQUIRED_OUTPUT_FORMAT = """
**REFERENCES:**
* DDG Item: <Top Match Item Number>
* ATA: <ata chapter>
* MEL Item: <Top Match MEL number (e.g. 46-11-02)>

If above DDG item isn't correct, check <Second Match Item Number>
"""

# ---------------- 3. BACKEND ----------------
@st.cache_resource
def load_backend():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ Groq API key not found.")
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

st.markdown("""
Find quick reference to DDG item/page.

**CAUTION**: For more accurate results, input text should be as close to DDG language as possible.  
*e.g. FCAC Flow Regulating Valve | Autothrottle Servo Motors*

**NOTE**: This is a prototype app for limited use.  
If slow or error msgs such as 'Rate limit',  
please WhatsApp +92 337 1244809.
""")

embed_model, index, metadatas, client = load_backend()

query = st.text_input("Enter Discrepancy:", 
                      placeholder="e.g. Forward cargo air conditioning exhaust fan",
                      key="query_input")

col1, col2 = st.columns([1, 1])
with col1:
    search_clicked = st.button("click to search DDG", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state.query_input = "" 
    st.session_state.processed = False
    st.rerun()

if search_clicked and query:
    with st.spinner("Searching DDG..."):
        q_emb = embed_model.encode([query], normalize_embeddings=True)
        scores, indices = index.search(q_emb, k=3)
        results = [metadatas[idx] for idx in indices[0]]
        context_text = build_context(results)
        
        # --- DATA EXTRACTION FOR PROMPT ---
        # 1. Get Top Match Data
        top_item_num = results[0].get('item_full', 'N/A')
        top_match_text = results[0].get('text') or results[0].get('content') or "[[TEXT MISSING]]"
        
        # 2. Get Second Match Data (Safety check in case only 1 result found)
        second_item_num = results[1].get('item_full', 'N/A') if len(results) > 1 else "None"

        # --- UPDATED USER PROMPT ---
        # We explicitly tell the AI which item is #1 and which is #2
        USER_PROMPT = f"""Pilot discrepancy: "{query}"

Top Match Item Number: {top_item_num}
Second Match Item Number: {second_item_num}

Context Data:
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
                max_tokens=300
            )
            
            st.markdown("### ✅ Dispatch Guidance")
            
            # 1. Print AI Output (References + Second Item check)
            st.markdown(completion.choices[0].message.content)
            
            # 2. Print RAW Text of TOP MATCH (Manually)
            st.markdown("**Page text**")
            st.text(top_match_text)
            
            with st.expander("See other potential matches"):
                st.text(context_text)
                
            st.session_state.processed = True
            
        except Exception as e:
            st.error(f"API Error: {e}")

elif search_clicked and not query:
    st.warning("Please enter a discrepancy.")
