import streamlit as st
import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="B777 DDG Assistant", layout="centered", page_icon="✈️")

# THE NEW UI TEXT YOU ASKED FOR
st.title("✈️ B777 DDG Assistant")
st.info("""
Find quick reference to DDG item. Click on reference to open relevant page.  
**CAUTION:** For accurate results, input text should be as close to DDG language as possible.  
*e.g. FCAC Flow Regulating Valve | Autothrottle Servo Motors* **NOTE:** This is a prototype app for limited use. If slow or error msgs such as 'Rate limit', please WhatsApp +92 337 1244809.
""")

# LOADING DATA
@st.cache_resource
def load_backend():
    api_key = st.secrets.get("GROQ_API_KEY")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index = faiss.read_index("ddg_faiss.index")
    with open("ddg_metadata.json", "r", encoding="utf-8") as f:
        metadatas = json.load(f)
    client = Groq(api_key=api_key)
    return embed_model, index, metadatas, client

embed_model, index, metadatas, client = load_backend()

query = st.text_input("Enter Discrepancy:", placeholder="Type here...")

if st.button("Search Manuals", type="primary"):
    if query:
        with st.spinner("Searching..."):
            q_emb = embed_model.encode([query], normalize_embeddings=True)
            scores, indices = index.search(q_emb, k=3)
            st.markdown("### ✅ Dispatch Guidance")
            for idx in indices[0]:
                item = metadatas[idx]
                ref = item.get('page_ref', 'N/A')
                page_num = item.get('page_index', 1)
                pdf_link = f"https://b777ddg.streamlit.app/static/B777DDG.pdf#page={page_num}"
                st.markdown(f"**DDG Item:** [{ref}]({pdf_link})")
                st.write("---")
