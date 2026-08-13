import os
import zipfile
import urllib.request
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# Oprește scanarea Streamlit pe modulele PyTorch (previne WinError 4551)
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# =========================================================
# ⚙️ 1. CONFIGURARE PAGINĂ STREAMLIT
# =========================================================
st.set_page_config(page_title="Asistent Juridic Moldova", page_icon="⚖️", layout="wide")

# =========================================================
# 📥 2. DESCĂRCARE AUTOMATĂ BAZĂ DE DATE DIN HUGGING FACE
# =========================================================
# ⚠️ Pune linkul tău direct din Hugging Face între ghilimele:
DB_URL = "https://huggingface.co/datasets/ChirilOfficial/asistent-juridic-db/resolve/main/chroma_db_FINAL_LOCAL.zip" 
ZIP_PATH = "chroma_db_FINAL_LOCAL.zip"
DB_DIR = "chroma_db_FINAL_LOCAL"

if not os.path.exists(DB_DIR):
    with st.spinner("🔍 Se descarcă baza de date juridică (se execută doar la prima pornire)..."):
        try:
            # Descărcare directă prin urllib (rapidă și fără blocaje)
            urllib.request.urlretrieve(DB_URL, ZIP_PATH)
            
            # Dezarhivare
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Ștergere arhivă zip
            if os.path.exists(ZIP_PATH):
                os.remove(ZIP_PATH)
        except Exception as e:
            st.error(f"Eroare la descărcarea bazei de date: {e}")
            st.stop()

# =========================================================
# 🔑 3. PRELUARE CHEIE API GROQ
# =========================================================
if "GROQ_API_KEY" not in st.secrets:
    st.error("🔑 Cheia GROQ_API_KEY nu a fost găsită în Streamlit Secrets!")
    st.stop()

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# =========================================================
# 🧠 4. ÎNCĂRCARE MODEL ȘI BAZĂ DE DATE (CACHED)
# =========================================================
@st.cache_resource
def load_rag_pipeline():
    chroma_client = chromadb.PersistentClient(path=f"./{DB_DIR}")
    
    collections = chroma_client.list_collections()
    if not collections:
        raise ValueError(f"Nu s-a găsit nicio colecție în folderul {DB_DIR}!")
        
    collection = collections[0]
    embedder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    return collection, embedder

collection, embedder = load_rag_pipeline()
client = Groq(api_key=GROQ_API_KEY)

# =========================================================
# 🎨 5. INTERFAȚĂ UTILIZATOR (UI)
# =========================================================
st.title("⚖️ Asistent Juridic AI - Republica Moldova")
st.caption("Căutare semantică și analiză inteligentă în baza de date unificată legis.md.")

with st.sidebar:
    st.header("⚖️ Despre Proiect")
    st.markdown("Acest asistent folosește Inteligența Artificială pentru a analiza baza de date legislativă oficială **legis.md**.")
    st.markdown("---")
    st.info("💡 **Sfat:** Adresează orice întrebare juridică privind Republica Moldova.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================================================
# 💬 6. PROCESARE ÎNTREBARE
# =========================================================
if user_query := st.chat_input("Adresează o întrebare despre legislația RM..."):

    history_context = ""
    if st.session_state.messages:
        recent_messages = st.session_state.messages[-4:]
        for msg in recent_messages:
            role_name = "Utilizator" if msg["role"] == "user" else "Asistent"
            clean_content = msg["content"].split("---")[0].strip()
            history_context += f"{role_name}: {clean_content}\n"

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Se analizează legislația Republicii Moldova..."):
            
            formatted_query = f"query: {user_query} Codul Civil Codul Contraventional Lege Republica Moldova"
            query_vector = embedder.encode([formatted_query], normalize_embeddings=True).tolist()

            results = collection.query(
                query_embeddings=query_vector,
                n_results=10
            )

            context_blocks = []
            sources = []
            
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                title = meta.get('title', 'Act Normativ fara titlu')
                url = meta.get('source_url', '#')
                
                context_blocks.append(f"--- ACT JURIDIC #{i+1} ---\nTitlu: {title}\nConținut: {doc}")
                sources.append(f"- [{title}]({url})")

            context_text = "\n\n".join(context_blocks)

            prompt = f"""Ești un asistent juridic de elită, expert în cadrul legislativ și codurile oficiale ale Republicii Moldova (Codul Civil, Codul Contravențional, Codul Muncii, Codul Penal etc.).

ISTORICUL CONVERSAȚIEI ANTERIOARE:
{history_context if history_context else "Nicio conversație anterioară."}

CONTEXT DIN BAZA DE DATE LEGISLATIVĂ (legis.md):
{context_text}

ÎNTREBARE UTILIZATOR:
{user_query}

REGULI MANDATORII PENTRU RĂSPUNS:
1. Răspunde ÎNTOTDEAUNA în limba în care a fost adresată întrebarea (Română sau Rusă).
2. Dacă utilizatorul adresează mai multe întrebări numerotate (ex: 1, 2, 3), RĂSPUNDE STRICT STRUCTURAT pe aceleași numere (1, 2, 3), oferind la fiecare punctul juridic direct + articolul din lege.
3. ARGUMENTARE JURIDICĂ STRICTĂ:
   - Specifică dacă este vorba de vechiul sau noul Cod Civil al RM (reforma a fost la 1 martie 2019).
   - Pentru termenii depășiți (cum ar fi moștenitori care vin după mulți ani), precizează că Notarul va respinge cererea și că este necesară o Hotărâre Judecătorească de repunere în termen (instanța de judecată).
   - Menționează că moștenitorul răspunde pentru datorii strict în limita valorii activelor moștenite.
4. Citează articole concrete unde este posibil.
5. Structurează răspunsul clar cu text îngroșat (**bold**) și puncte pentru lizibilitate."""

        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                top_p=0.2,
            )
            response_text = completion.choices[0].message.content
            
            full_response = f"{response_text}\n\n---\n### 📄 Surse Consultate din Baza de Date:\n" + "\n".join(sources)
            
            st.markdown(full_response)
            
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Eroare la procesarea cererii: {e}")
