import streamlit as st
import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------- 1. CONFIGURATION ----------------
st.set_page_config(page_title="B777 DDG Assistant", layout="centered", page_icon="✈️")

# --- CSS STYLES ---
# 1. Hide Profile/Footer
# 2. Hide "Press Enter to apply" instruction
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Hide the 'Press Enter to apply' hint */
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

if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

# ---------------- 2. PROMPTS ----------------
# We ask the AI to output JSON so we can parse it reliably in Python
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
    # This string is for the AI to read and judge
    blocks = []
    for i, item in enumerate(results):
        raw_text = item.get('text') or item.get('content') or ""
        # We give the AI the index number [0], [1], etc.
        block = f"""
[INDEX {i}]
Item: {item.get('item_full', 'N/A')}
Title: {item.get('title', 'N/A')}
Text: {raw_text[:200]}... (snippet)
"""
        blocks.append(block)
    return "\n".join(blocks)

def get_mel_string(ddg_item):
    # Python Logic to strip prefix/suffix: 2.36-21-01.1 -> 36-21-01
    try:
        # Split by dots, take the middle parts
        parts = ddg_item.split('.')
        if len(parts) >= 3:
            return parts[1] # Returns the middle section
        return ddg_item # Fallback
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

query = st.text_input("Enter Discrepancy:", 
                      placeholder="e.g. Forward cargo air conditioning exhaust fan inoperative",
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
        # 1. Search Logic
        q_emb = embed_model.encode([query], normalize_embeddings=True)
        # Get top 6 to allow AI to choose between similar items (Pressure vs Temp)
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
            # We use JSON mode to ensure clean integer output
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
            
            # Safety check: Ensure index is valid
            if best_index < 0 or best_index >= len(results):
                best_index = 0
            
            # 5. Extract Data using Python (Not AI)
            selected_item = results[best_index]
            ddg_num = selected_item.get('item_full', 'N/A')
            ata_num = selected_item.get('ata', 'N/A')
            mel_num = get_mel_string(ddg_num)
            
            # Get the EXACT Raw text from your JSON
            raw_text_display = selected_item.get('text') or selected_item.get('content') or "[[TEXT MISSING IN JSON]]"
            
            # 6. Display Output
            st.markdown("### ✅ Dispatch Guidance")
            
            st.markdown(f"""
**REFERENCES:**
* DDG Item: {ddg_num}
* ATA: {ata_num}
* MEL Item: {mel_num}
""")
            
            st.markdown("## DDG page text")
            st.text(raw_text_display) # <--- This prints exact raw text
            
            st.session_state.processed = True
            
        except Exception as e:
            st.error(f"API Error: {e}")

elif search_clicked and not query:
    st.warning("Please enter a discrepancy.")
