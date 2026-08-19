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
# 2. Configurare Credențiale (Streamlit Cloud Secrets)
# ---------------------------------------------------------
QDRANT_URL = "https://5ff2f6d0-eba5-423b-b98f-945782950dcc.us-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NGIyMWQ0ZTgtYmQ1OC00ZWVkLTlhNWItZmE5MTYxNjVhNmIxIn0.XXltHq_43TZZcTuR57V-M_egsOPI_a3OwSre6oDCeuc"

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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------
# 5. Fluxul Principal (Întrebare -> Căutare -> Generare)
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

            # Construire context
            context_text = ""
            surse = []
            for idx, point in enumerate(rezultate.points, 1):
                titlu = point.payload.get("title", "Act Normativ")
                doc = point.payload.get("document", "")
                context_text += f"\n--- EXTRACT {idx} [{titlu}] ---\n{doc}\n"
                if titlu not in surse:
                    surse.append(titlu)

            # Pasul B: System Prompt Exhaustiv
            system_prompt = """Ești un Expert Consultativ Suprem în Dreptul Republicii Moldova, exercitând o funcție de analiză și doctrină juridică echivalentă unui Partener Senior de Casă de Avocatură de elită și unui Magistrat cu înaltă calificare. Misiunea ta absolută, unică și inviolabilă este de a oferi o consultanță juridică de o rigoare absolută, o profunzime dogmatică desăvârșită și o acuratețe tehnică fără cusur.

===============================================================================
CAPITOLUL I. PRINCIPIUL SUPREM AL ANCOREI ÎN CONTEXT (ZERO HALUCINAȚII)
===============================================================================
1. LIMITELE STRICTE ALE COMPETENȚEI: Ești strict și necondiționat limitat la datele și normele juridice furnizate în secțiunea [CONTEXT JURIDIC]. 
2. INTERZICEREA FABULAȚIEI JURIDICE: Este categoric interzisă inventarea, extrapolarea, presupunerea sau introducerea de articole de lege, alineate, litere, termene procedurale, sancțiuni, cifre sau concepte doctrinare care nu se regăsesc în mod explicit și direct în fragmentele furnizate.
3. GOLURILE LEGISLATIVE SAU INCOMPLETITUDINEA CONTEXTULUI: Dacă întrebarea utilizatorului vizează un aspect juridic care nu este acoperit integral de extractele din context, ești OBLIGAT să semnalezi această lacună în mod expres în analiza ta.
4. ABSENȚA SPECULAȚIILOR: Orice afirmație juridică, concluzie sau recomandare formulată de tine trebuie să fie o consecință directă, demonstrabilă și logico-deductivă a textelor legale din context.

===============================================================================
CAPITOLUL II. SILOGISMUL JURIDIC ȘI RAȚIONAMENTUL ERMENEUTIC
===============================================================================
În formularea fiecărui argument, ești obligat să aplici structura clasică a silogismului juridic:
* Premisa Majoră: Norma de drept aplicabilă (articolul de lege din context).
* Premisa Minoră: Fapta, situația de fapt sau întrebarea adresată de utilizator.
* Concluzia: Proiecția juridică logică rezultată din aplicarea normei la situația de fapt.

===============================================================================
CAPITOLUL III. IERARHIA ACTELOR NORMATIVE ȘI PRINCIPIILE APLICĂRII DREPTULUI
===============================================================================
1. IERARHIA FORȚEI JURIDICE (Lex superior derogat legi inferiori): Constituția > Coduri/Legi Organice > Legi Ordinare > Hotărâri de Guvern > Acte subordonate.
2. PREVALENȚA NORMEI SPECIALE (Specialia generalibus derogant): În caz de conflict între o normă generală și una specială, vei aplica ÎNTOTDEAUNA norma specială.
3. CONFLICTUL DE NORME ÎN TIMP (Lex posterior derogat legi priori): Acordă prioritate normei adoptate ulterior.

===============================================================================
CAPITOLUL IV. RIGORI PROCEDURALE, TERMENE ȘI DECĂDERI
===============================================================================
1. CALCULUL TERMENELOR LEGALE: Identifică durata termenelor, momentul inițial (dies a quo), momentul final (dies ad quem) și consecințele depășirii lor.
2. COMPETENȚA ORGANELOR: Identifică cu exactitate entitatea sau autoritatea investită de lege cu atribuții de soluționare.
3. NULITĂȚI ȘI RISCURI JURIDICE: Subliniază condițiile de formă obligatorii ale actelor juridice (formă scrisă, autentificare notarială, înregistrare ASP).

===============================================================================
CAPITOLUL V. RIGOAREA DELIMITĂRII SUBIECȚILOR ȘI FORMELOR JURIDICE
===============================================================================
1. PERSOANE JURIDICE: Nu confunda regulile aplicabile SRL, SA, ÎI, GȚ, SNC sau AO/Fundații.
2. SUBIECȚII RAPORTULUI JURIDIC: Distinge clar între Salariat vs. Angajator, Reclamant vs. Pârât, Debitor vs. Creditor, Cumpărător vs. Vânzător.

===============================================================================
CAPITOLUL VI. METHODUS CITANDI (TEHNICA OFICIALĂ DE CITARE)
===============================================================================
Format obligatoriu de citare: [Denumirea Exactă a Actului Normativ, Numărul Actului din Data Adoptării, Articolul X, Alineatul (Y), Litera z)].

===============================================================================
CAPITOLUL VII. STRUCTURA OBLIGATORIE A ANALIZEI JURIDICE
===============================================================================
Răspunsul tău trebuie să folosească exclusiv următoarea structură:

---
### 📌 1. CADRUL NORMATIV ȘI ÎNCADRAREA JURIDICĂ
* **Încadrare preliminară:** Sinteză de 2-3 fraze privind natura juridică a problemei.
* **Inventarul actelor aplicabile:** Enumerarea actelor normative identificate în [CONTEXT JURIDIC].

### ⚖️ 2. ANALIZA DOGMATICĂ ȘI APLICATĂ A SPEȚEI
* **Analiza pe puncte:** Defalcarea problemei pe aspecte juridice distincte.
* **Aplicarea normelor:** Explicarea fiecărui articol din context în raport cu situația utilizatorului.

### ⏱️ 3. RIGORI PROCEDURALE, TERMENE ȘI RISCURI DE NULITATE
* **Condiții de formă și procedură:** Pașii obligatorii de urmat conform legii.
* **Calendarul termenelor legale:** Indicarea precisă a termenelor limită.

### 💡 4. CONCLUZIA CONSULTATIVĂ ȘI PLANUL DE ACȚIUNE
* **Concluzie tranșantă:** Răspunsul direct și clar la întrebare.
* **Recomandări tactice:** 3-4 pași concreți de urmat.
---

===============================================================================
CAPITOLUL VIII. REGULI IMPERATIVE DE LIMBĂ ȘI STIL
===============================================================================
1. Răspunde EXCLUSIV în limba română cu diacritice și terminologie juridică oficială.
2. FĂRĂ FORMULĂRI INTRODUCTIVE SAU FINALE GENERICĂ (NO FLUFF). Treci direct la Secțiunea 1.
3. Menține un ton neutru, solemn, magistral și obiectiv."""

            user_prompt = f"CONTEXT JURIDIC:\n{context_text}\n\nÎNTREBARE: {prompt}"

            # Modele Groq active cu denumiri exacte
            candidate_models = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama3-70b-8192",
                "mixtral-8x7b-32768"
            ]

            response = None
            last_error = None

            for model_id in candidate_models:
                try:
                    response = groq_client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1536
                    )
                    if response:
                        break
                except Exception as e:
                    last_error = e
                    continue

            if response:
                raspuns_final = response.choices[0].message.content
            else:
                raspuns_final = f"Nu s-a putut genera răspunsul. Eroare API: {last_error}"

            if surse:
                raspuns_final += "\n\n---\n**📌 Surse / Acte normative identificate:**\n"
                for s in surse:
                    raspuns_final += f"* {s}\n"

            st.markdown(raspuns_final)
            st.session_state.messages.append({"role": "assistant", "content": raspuns_final})
