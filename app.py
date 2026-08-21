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
# 2. Configurare Credențiale
# ---------------------------------------------------------
QDRANT_URL = st.secrets.get("QDRANT_URL", "https://5ff2f6d0-eba5-423b-b98f-945782950dcc.us-west-2-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NGIyMWQ0ZTgtYmQ1OC00ZWVkLTlhNWItZmE5MTYxNjVhNmIxIn0.XXltHq_43TZZcTuR57V-M_egsOPI_a3OwSre6oDCeuc")
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ---------------------------------------------------------
# 3. Inițializare Modele
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

# Afișează istoricul conversației
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------
# 5. Fluxul Principal (Întrebare -> Căutare -> Generare)
# ---------------------------------------------------------
if prompt := st.chat_input("Exemplu: Care sunt drepturile angajatului la concediere?"):
    # Afișăm întrebarea utilizatorului
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generăm răspunsul
    with st.chat_message("assistant"):
        with st.spinner("🔍 Caut în baza de date juridică și formulez răspunsul..."):
            
            # Pasul A: Căutare semantică pură în Qdrant Cloud
            search_text = f"query: {prompt}"
            query_vector = embed_model.encode(search_text).tolist()
            
            rezultate = qdrant.query_points(
                collection_name="legis_md",
                query=query_vector,
                limit=10  # Căutăm 10 fragmente relevante
            )

            # Construire context din legile găsite
            context_text = ""
            surse = []
            for idx, point in enumerate(rezultate.points, 1):
                titlu = point.payload.get("title", "Act Normativ")
                doc = point.payload.get("document", "")
                context_text += f"\n--- EXTRACT {idx} [{titlu}] ---\n{doc}\n"
                if titlu not in surse:
                    surse.append(titlu)

            # Pasul B: Generare răspuns cu model activ în Groq
            system_prompt = """Ești un Expert Consultativ Suprem în Dreptul Republicii Moldova (cu nivel de Partener de Casă de Avocatură și Magistrat).
Misiunea ta este să oferi o ANALIZĂ JURIDICĂ IMPECABILĂ, de o rigoare, acuratețe și profunzime absolute.

RIGORI ȘI PRINCIPII MANDATORII:
1. RIGORILE ȘI DETALIUL PROCEDURAL: Analizează cu precizie chirurgicală termenele legale (zile, luni), competențele organelor (ex: judecător de drepturi și libertăți vs. procuror), excepțiile, sancțiunile și nulitățile procedurale.
2. IERARHIA ACTELOR NORMATIVE (Specialia generalibus derogant): Prioritizează întotdeauna LEGILE SPECIALE și Codurile de profil în raport cu norma generală (ex: Legea SRL sau Codul Muncii au prioritate față de Codul Civil general).
3. ATENȚIE LA FORMA JURIDICĂ: Nu confunda tipurile de persoane juridice (ex: Societatea în Nume Colectiv - SNC vs. Societatea cu Răspundere Limitată - SRL). Verifică atent la ce tip de societate se referă articolul extras și precizează explicit forma juridică pe tot parcursul analizei!
4. STRICT BAZAT PE CONTEXT: Răspunde EXCLUSIV în baza textelor de lege furnizate în CONTEXT. Nu fabula și nu presupune.
5. CITARE EXACTĂ: Precizează numărul articolului, alineatul, litera și denumirea exactă a actului normativ.
6. SINTETIZARE STRUCTURATĂ: Prezintă analiza sub formă de concluzii juridice clare:
   - Cadru Legal & Norme Aplicabile
   - Condiții Procedurale & Termene Stricte
   - Excepții & Riscuri/Nulități
   - Concluzie / Recomandare Legală"""

            user_prompt = f"CONTEXT JURIDIC:\n{context_text}\n\nÎNTREBARE: {prompt}"

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
            
            # Adăugăm sursele la finalul răspunsului
            if surse:
                raspuns_final += "\n\n---\n**📌 Surse / Acte normative identificate:**\n"
                for s in surse:
                    raspuns_final += f"* {s}\n"

            st.markdown(raspuns_final)
            
            # Salvăm răspunsul în istoric
            st.session_state.messages.append({"role": "assistant", "content": raspuns_final})
