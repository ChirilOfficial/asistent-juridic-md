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
# 2. Preluare și Verificare Cheie API
# ---------------------------------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "").strip()

if not GROQ_API_KEY:
    st.error("⚠️ Lipsesc setările API! Adaugă `GROQ_API_KEY` în Settings > Secrets în Streamlit Cloud.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

# Extragere dinamică modele din contul Groq
available_models = []
try:
    models_data = groq_client.models.list().data
    available_models = [m.id for m in models_data]
except Exception as e:
    st.sidebar.error(f"Eroare la citirea modelelor: {e}")

# Meniul din stânga pentru diagnostic exact
with st.sidebar:
    st.header("⚙️ Status Conexiune")
    st.success(f"Cheie detectată: `{GROQ_API_KEY[:7]}...`")
    st.subheader("Modele Groq disponibile în cont:")
    if available_models:
        for m in available_models:
            st.code(m, language="text")
    else:
        st.warning("Nu s-a putut încărca lista de modele.")

# ---------------------------------------------------------
# 3. Inițializare Qdrant & Embeddings
# ---------------------------------------------------------
QDRANT_URL = "https://5ff2f6d0-eba5-423b-b98f-945782950dcc.us-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NGIyMWQ0ZTgtYmQ1OC00ZWVkLTlhNWItZmE5MTYxNjVhNmIxIn0.XXltHq_43TZZcTuR57V-M_egsOPI_a3OwSre6oDCeuc"

@st.cache_resource
def load_qdrant_and_embed():
    embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)
    return embed_model, qdrant

embed_model, qdrant = load_qdrant_and_embed()

# ---------------------------------------------------------
# 4. Istoric Chat
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------
# 5. Flux Principal
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

            # Pasul C: Utilizare automată a primului model disponibil în cont
            if not available_models:
                raspuns_final = "⚠️ Nu s-a găsit niciun model accesibil pe acest cont Groq."
            else:
                model_de_folosit = available_models[0]
                try:
                    response = groq_client.chat.completions.create(
                        model=model_de_folosit,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1536
                    )
                    raspuns_final = response.choices[0].message.content
                except Exception as err:
                    raspuns_final = f"⚠️ Eroare la generare cu modelul `{model_de_folosit}`: {err}"

            if surse:
                raspuns_final += "\n\n---\n**📌 Surse / Acte normative identificate:**\n"
                for s in surse:
                    raspuns_final += f"* {s}\n"

            st.markdown(raspuns_final)
            st.session_state.messages.append({"role": "assistant", "content": raspuns_final})
