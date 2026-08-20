import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------------------------------------------------
# 1. Configurare Pagină Web (Afișaj curat pentru utilizatori)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Asistent Juridic Moldova", 
    page_icon="⚖️", 
    layout="centered"
)

st.title("⚖️ Asistent Juridic AI - Republica Moldova")
st.markdown("Adresează o întrebare juridică. Sistemul caută în peste **1.16 milioane de articole de lege** și formulează un răspuns argumentat.")

# ---------------------------------------------------------
# 2. Configurare Credențiale și Client Groq
# ---------------------------------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "").strip()

if not GROQ_API_KEY:
    st.error("⚠️ Serviciul este temporar indisponibil. Reîncercați mai târziu.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------
# 3. Selectare automată și silențioasă a celui mai bun model
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_best_available_model():
    # Ordinea de prioritate pentru performanță maximă pe text juridic
    preferred_priority = [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-128b",
        "qwen/qwen3.5-27b",
        "llama-3.1-8b-instant"
    ]
    try:
        models_data = groq_client.models.list().data
        available_ids = [m.id for m in models_data]
        
        # 1. Alege primul model din lista de preferințe care există în cont
        for pref in preferred_priority:
            if pref in available_ids:
                return pref
                
        # 2. Fallback: primul model de chat valid ce nu e guard/whisper/compound
        valid_chat_models = [
            m for m in available_ids 
            if not any(x in m.lower() for x in ["whisper", "guard", "safeguard", "compound"])
        ]
        if valid_chat_models:
            return valid_chat_models[0]
            
    except Exception:
        pass
        
    return "llama-3.3-70b-versatile"

BEST_MODEL = get_best_available_model()

# ---------------------------------------------------------
# 4. Inițializare Qdrant & Embeddings
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
# 5. Istoric Chat
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------
# 6. Flux Principal
# ---------------------------------------------------------
if prompt := st.chat_input("Exemplu: Care sunt drepturile angajatului la concediere?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Caut în baza de date juridică și formulez răspunsul..."):
            
            # Căutare semantică în Qdrant
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

            # System Prompt profesional
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

            try:
                response = groq_client.chat.completions.create(
                    model=BEST_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1200
                )
                raspuns_final = response.choices[0].message.content
            except Exception as err:
                raspuns_final = "⚠️ A apărut o eroare la generarea răspunsului. Vă rugăm să reîncercați."

            if surse:
                raspuns_final += "\n\n---\n**📌 Surse / Acte normative identificate:**\n"
                for s in surse:
                    raspuns_final += f"* {s}\n"

            st.markdown(raspuns_final)
            st.session_state.messages.append({"role": "assistant", "content": raspuns_final})
