import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------------------------------------------------
# 1. Configurare Pagină Web
# ---------------------------------------------------------
st.set_page_config(
    page_title="Asistent Juridic Moldova", 
    page_icon="⚖️", 
    layout="centered"
)

st.title("⚖️ Asistent Juridic AI - Republica Moldova")
st.markdown("Adresează o întrebare juridică. Sistemul caută în peste **1.16 milioane de articole de lege** și formulează un răspuns argumentat.")

# ---------------------------------------------------------
# 2. Configurare Credențiale (Streamlit Secrets)
# ---------------------------------------------------------
QDRANT_URL = "https://5ff2f6d0-eba5-423b-b98f-945782950dcc.us-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NGIyMWQ0ZTgtYmQ1OC00ZWVkLTlhNWItZmE5MTYxNjVhNmIxIn0.XXltHq_43TZZcTuR57V-M_egsOPI_a3OwSre6oDCeuc"

# Verificare cheie Groq din Streamlit Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("⚠️ Lipsesc setările API! Adaugă `GROQ_API_KEY` în panoul **Settings > Secrets** din Streamlit Cloud.")
    st.stop()

# ---------------------------------------------------------
# 3. Inițializare Servicii
# ---------------------------------------------------------
@st.cache_resource
def init_services():
    embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
    groq_client = Groq(api_key=GROQ_API_KEY)
    return embed_model, qdrant, groq_client

embed_model, qdrant, groq_client = init_services()

# ---------------------------------------------------------
# 4. Istoric Chat
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------
# 5. Flux Principal (Căutare Qdrant + Generare Groq)
# ---------------------------------------------------------
if prompt := st.chat_input("Exemplu: Care sunt drepturile angajatului la concediere?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Caut în baza de date juridică și formulez răspunsul..."):
            
            # Pasul A: Căutare semantică în Qdrant
            search_text = f"query: {prompt}"
            query_vector = embed_model.encode(search_text).tolist()
            
            rezultate = qdrant.query_points(
                collection_name="legis_md",
                query=query_vector,
                limit=10
            )

            # Construire context normativ
            context_text = ""
            surse = []
            for idx, point in enumerate(rezultate.points, 1):
                titlu = point.payload.get("title", "Act Normativ")
                doc = point.payload.get("document", "")
                context_text += f"\n--- EXTRACT {idx} [{titlu}] ---\n{doc}\n"
                if titlu not in surse:
                    surse.append(titlu)

            # Pasul B: System Prompt
            system_prompt = """Ești un Expert Consultativ Suprem în Dreptul Republicii Moldova. Misiunea ta este de a oferi consultanță juridică bazată exclusiv pe normele furnizate în context.

CAPITOLUL I. PRINCIPIUL SUPREM AL ANCOREI ÎN CONTEXT
1. Ești limitat la datele din [CONTEXT JURIDIC].
2. Este interzisă inventarea de articole sau sancțiuni neexistente în context.

CAPITOLUL II. STRUCTURA RĂSPUNSULUI
Răspunsul tău trebuie să folosească structura:
### 📌 1. CADRUL NORMATIV ȘI ÎNCADRAREA JURIDICĂ
### ⚖️ 2. ANALIZA DOGMATICĂ ȘI APLICATĂ A SPEȚEI
### ⏱️ 3. RIGORI PROCEDURALE ȘI TERMENE
### 💡 4. CONCLUZIA CONSULTATIVĂ ȘI PLANUL DE ACȚIUNE"""

            user_prompt = f"CONTEXT JURIDIC:\n{context_text}\n\nÎNTREBARE: {prompt}"

            # Pasul C: Apel API Groq cu modelul activ
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1536
                )
                raspuns_final = response.choices[0].message.content
            except Exception as e:
                raspuns_final = f"⚠️ Eroare API Groq: {e}\n\nVerifică dacă cheia API din Streamlit Secrets este validă."

            if surse:
                raspuns_final += "\n\n---\n**📌 Surse / Acte normative identificate:**\n"
                for s in surse:
                    raspuns_final += f"* {s}\n"

            st.markdown(raspuns_final)
            st.session_state.messages.append({"role": "assistant", "content": raspuns_final})
