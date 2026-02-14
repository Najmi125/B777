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

YOUR TASK:
1. Analyze the User's technical query (e.g., "Pressure" vs "Temperature", "Left" vs "Right").
2. Look at the provided CANDIDATE ITEMS from the database.
3. Select the ONE item that best matches the specific technical details of the query.
   - If user asks for "Pressure", DO NOT select "Temperature" even if it is the first option.
   - If user asks for "Valve", DO NOT select "Sensor".
4. Output the Metadata and the FULL Page Text of that specific selected item.

DERIVE the MEL Item from the DDG Item number by removing the prefix (e.g., "2.") and suffix.
"""

REQUIRED_OUTPUT_FORMAT = """
**REFERENCES:**
* DDG Item: <Selected Item Number>
* ATA: <ata chapter>
* MEL Item: <Derived MEL number (e.g. 46-11-02)>

If above DDG item isn't correct, check <Next Best Item Number>

**Page text**
<Insert the EXACT FULL text of the selected item here. Do not summarize.>
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
    for i, item in enumerate(results):
        raw_text = item.get('text') or item.get('content') or "[[TEXT MISSING]]"
        block = f"""
[OPTION {i+1}]
DDG ITEM: {item.get('item_full', 'N/A')}
Title: {item.get('title', 'N/A')}
ATA: {item.get('ata', 'N/A')}
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
If you find it slow or see a 'Rate limit' error,  
please WhatsApp +92 337 1244809.
""")

embed_model, index, metadatas, client = load_backend()

query = st.text_input("Enter Pilot Discrepancy:", 
                      placeholder="E.g. Forward cargo air conditioning exhaust fan inoperative",
                      key="query_input")

# Search Button
search_clicked = st.button("click to search DDG", type="primary", use_container_width=True)

if search_clicked and query:
    with st.spinner("Searching DDG..."):
        # 1. Search
        q_emb = embed_model.encode([query], normalize_embeddings=True)
        
        # ⚠️ CHANGED: Increased k to 6 to ensure we capture the right item if it's not #1
        scores, indices = index.search(q_emb, k=6)
        results = [metadatas[idx] for idx in indices[0]]
        
        # 2. Build Context (Feeding all 6 options to the AI)
        context_text = build_context(results)

        # 3. Prompt (Asking AI to be the judge)
        USER_PROMPT = f"""Pilot discrepancy: "{query}"

Here are 6 Candidate Items from the manual. 
Analyze the Discrepancy carefully. 
Pick the Candidate that matches the technical keywords (e.g. Pressure vs Temp) best.

CANDIDATES:
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
            
            # Display Result
            st.markdown("### ✅ Dispatch Guidance")
            st.markdown(completion.choices[0].message.content)
            
            with st.expander("See raw search candidates"):
                st.text(context_text)
                
            st.session_state.processed = True
            
        except Exception as e:
            st.error(f"API Error: {e}")

elif search_clicked and not query:
    st.warning("Please enter a discrepancy.")
