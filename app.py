import streamlit as st
import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------- 1. CONFIGURATION ----------------
st.set_page_config(page_title="B777 DDG Assistant", layout="centered", page_icon="✈️")

# --- CSS STYLES (UI FIXES) ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            [data-testid="stToolbar"] {
                display: none !important;
            }
            [data-testid="stStatusWidget"] {
                visibility: hidden;
            }
            .stTextInput > div > div > span {
                display: none;
            }
            div[data-testid="InputInstructions"] {
                display: none;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# File Safety Check
required_files = ["ddg_faiss.index", "ddg_metadata.json"]
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"❌ Missing required files: {', '.join(missing_files)}")
    st.stop()

# Initialize Session State
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

# --- CALLBACK TO CLEAR TEXT ---
def clear_text():
    st.session_state.query_input = ""
    st.session_state.processed = False

# ---------------- 2. PROMPTS ----------------
SYSTEM_PROMPT = """You are a Boeing 777 technical analyzer.
Your ONLY job is to select the best match from the provided candidates.
Output valid JSON only. Format: {"best_index": <integer_index>}
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

def build_candidates_string(results):
    blocks = []
    for i, item in enumerate(results):
        block = f"""
[INDEX {i}]
Item: {item.get('item_full', 'N/A')}
Title: {item.get('title', 'N/A')}
"""
        blocks.append(block)
    return "\n".join(blocks)

def get_mel_string(ddg_item):
    try:
        parts = ddg_item.split('.')
        if len(parts) >= 3:
            return parts[1] 
        return ddg_item
    except:
        return ddg_item

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

# Input Box
query = st.text_input("Enter Discrepancy:", 
                      placeholder="e.g. Forward cargo air conditioning exhaust fan inoperative",
                      key="query_input")

col1, col2 = st.columns([1, 1])
with col1:
    search_clicked = st.button("click to search DDG", type="primary", use_container_width=True)
with col2:
    st.button("Clear", use_container_width=True, on_click=clear_text)

# --- SEARCH LOGIC ---
if search_clicked and query:
    with st.spinner("Searching DDG..."):
        # 1. Search Logic
        q_emb = embed_model.encode([query], normalize_embeddings=True)
        scores, indices = index.search(q_emb, k=6) 
        results = [metadatas[idx] for idx in indices[0]]
        
        # 2. Build string for AI analysis
        candidates_text = build_candidates_string(results)

        # 3. Ask AI to pick the winner
        USER_PROMPT = f"""Pilot discrepancy: "{query}"

Analyze these 6 candidates. Which one matches the technical details (e.g. Pressure vs Temp) best?
Return ONLY the JSON with the best_index.

{candidates_text}
"""
        
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                temperature=0.0,
                max_tokens=50,
                response_format={"type": "json_object"}
            )
            
            # 4. Parse AI Response
            response_data = json.loads(completion.choices[0].message.content)
            best_index = int(response_data.get("best_index", 0))
            if best_index < 0 or best_index >= len(results):
                best_index = 0
            
            # 5. Extract Best Match Data
            selected_item = results[best_index]
            ddg_num = selected_item.get('item_full', 'N/A')
            ata_num = selected_item.get('ata', 'N/A')
            mel_num = get_mel_string(ddg_num)
            
            # Get Raw Text
            raw_text = selected_item.get('text') or selected_item.get('content') or "[[TEXT MISSING IN JSON]]"

            # 6. Display Output
            st.markdown("### ✅ Dispatch Guidance")
            
            # Simple Output as requested
            st.write(f"**DDG Item:** {ddg_num}")
            st.write(f"**MEL Item:** {mel_num}")
            
            st.markdown("### DDG page text")
            st.text(raw_text)
            
            st.session_state.processed = True
            
        except Exception as e:
            st.error(f"API Error: {e}")

elif search_clicked and not query:
    st.warning("Please enter a discrepancy.")
