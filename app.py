import math
import os
import re
import unicodedata
from collections import Counter

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Configuração da página e modelos
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mafalda - Conselheira de Apoio à Neurofibromatose",
    page_icon="💙",
    layout="wide",
)

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_HISTORY_MESSAGES = 20
TOP_K_CHUNKS = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---------------------------------------------------------------------------
# Protocolos de Emergência (Portugal)
# ---------------------------------------------------------------------------
EMERGENCY_MESSAGE = """
🚨 **Se isto é uma emergência ou sentes que estás em perigo, procura ajuda imediata:**

- **112** — Número Europeu de Emergência
- **SNS 24 — 808 24 24 24** (inclui aconselhamento psicológico)
- **SOS Voz Amiga — 213 544 545** (apoio emocional)
- **Conversa Amiga — 808 237 327** (apoio emocional)
- **SOS Criança — 116 111** (crianças e jovens)

Não estás sozinho/a. Se puderes, fala também com alguém de confiança perto de ti.
"""

EMERGENCY_PATTERNS = [
    r"suicid\w*", r"matar[- ]?me", r"quero morrer", r"acabar com (tudo|a minha vida)",
    r"(me|-me) magoar", r"automutil\w*", r"overdose",
    r"nao consigo respirar", r"convuls\w*", r"desmai\w*", r"inconsciente",
    r"perdi a visao", r"perda subita de (visao|audicao)", r"paralisia",
    r"dor (forte )?no peito", r"\bavc\b", r"hemorragia", r"sangramento (intenso|forte)",
]

STOPWORDS = {
    "para", "como", "mais", "mas", "que", "com", "uma", "uns", "umas", "por", "dos", "das",
    "nos", "nas", "pelo", "pela", "isso", "isto", "esta", "este", "estes", "estas", "esse",
    "essa", "ser", "sao", "tem", "tenho", "pode", "posso", "qual", "quais", "quando", "onde",
    "sobre", "entre", "muito", "tambem", "seu", "sua", "meu", "minha", "foi", "era", "the",
    "and", "from", "with", "that", "this",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]{3,}", normalize(text)) if t not in STOPWORDS]


def detect_emergency(text: str) -> bool:
    norm = normalize(text)
    return any(re.search(p, norm) for p in EMERGENCY_PATTERNS)


# ---------------------------------------------------------------------------
# Gestão e Indexação de Documentos Próprios (Base de Conhecimento RAG)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def extract_pdf_cached(name: str, data: bytes) -> dict:
    import io
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return {"pages": [], "error": "PDF protegido por palavra-passe."}
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((i, text))
        if not pages:
            return {"pages": [], "error": "Não foi possível extrair texto (documento escaneado/imagem)."}
        return {"pages": pages, "error": None}
    except Exception as e:
        return {"pages": [], "error": f"Erro ao ler PDF: {e}"}


def build_chunks(files) -> list[dict]:
    all_chunks = []
    for f in files:
        data = f.getvalue()
        res = extract_pdf_cached(f.name, data)
        if res["error"]:
            continue
        for page_no, text in res["pages"]:
            start = 0
            while start < len(text):
                piece = text[start : start + CHUNK_SIZE].strip()
                if piece:
                    tokens = tokenize(piece)
                    all_chunks.append({
                        "source": f.name,
                        "page": page_no,
                        "text": piece,
                        "tf": Counter(tokens),
                        "len": max(len(tokens), 1),
                    })
                if start + CHUNK_SIZE >= len(text):
                    break
                start += CHUNK_SIZE - CHUNK_OVERLAP
    return all_chunks


def retrieve_excerpts(chunks: list[dict], query: str, k: int = TOP_K_CHUNKS) -> list[dict]:
    q_tokens = set(tokenize(query))
    if not chunks or not q_tokens:
        return []
    n = len(chunks)
    df = Counter()
    for c in chunks:
        for t in q_tokens:
            if t in c["tf"]:
                df[t] += 1
    scored = []
    for c in chunks:
        score = 0.0
        for t in q_tokens:
            tf = c["tf"].get(t, 0)
            if tf:
                idf = math.log(1 + n / (1 + df[t]))
                score += (1 + math.log(tf)) * idf
        if score > 0:
            scored.append((score / math.sqrt(c["len"]), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


# ---------------------------------------------------------------------------
# Painel de Controlo (Barra Lateral)
# ---------------------------------------------------------------------------
st.sidebar.title("🛠️ Painel de Controlo")

api_key = st.sidebar.text_input(
    "Chave API do Google Gemini",
    value=os.environ.get("GEMINI_API_KEY", ""),
    type="password",
    help="Insere a tua chave obtida no Google AI Studio.",
)
model_name = st.sidebar.text_input("Modelo Gemini", value=DEFAULT_MODEL)

client = genai.Client(api_key=api_key) if api_key else None

st.sidebar.divider()
st.sidebar.subheader("📚 Ala de Conhecimento Próprio")
st.sidebar.markdown(
    "Carrega documentos internos, guias traduzidos, minutas ou relatórios de referência. "
    "A Mafalda cruzará esta base própria com os portais internacionais (CTF, APNF, ERN Genturis)."
)

uploaded_files = st.sidebar.file_uploader(
    "Adicionar ficheiros à base de dados", type=["pdf"], accept_multiple_files=True
)

current_chunks = []
if uploaded_files:
    current_chunks = build_chunks(uploaded_files)
    st.sidebar.success(f"Base Própria Ativa: {len(uploaded_files)} documento(s) indexado(s) ({len(current_chunks)} segmentos).")

st.sidebar.divider()
if st.sidebar.button("🗑️ Limpar Conversa", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ---------------------------------------------------------------------------
# System Prompt (Instruções da Mafalda)
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = """
És a Mafalda, uma conselheira empática e de referência, especializada em Neurofibromatose
Tipo 1 (NF1), Tipo 2 (NF2) e Schwannomatose. A tua missão é apoiar doentes de todas as idades,
familiares, cuidadores e profissionais de saúde, servindo como porto seguro contra a desinformação.
Responde sempre em português europeu.

DIRETRIZES DE ATUAÇÃO:
1. ESTATUTO NÃO CLÍNICO: Não és médica nem profissional de saúde mental. Não fazes diagnósticos definitivos nem prescrições. Recomenda sempre o acompanhamento clínico especializado.
2. BASE DE CONHECIMENTO PRÓPRIA: Quando existirem "EXCERTOS DA BASE PRÓPRIA", utiliza-os com prioridade absoluta para fundamentar a resposta. 
3. PORTAIS INTERNACIONAIS: Complementa sempre com diretrizes de entidades de referência (Children’s Tumor Foundation, APNF, ERN Genturis, EURORDIS).
4. RIGOR E CITAÇÕES: Indica sempre a origem da informação (ex.: [Fonte: nome_do_ficheiro.pdf, p. X] ou [Porto Seguro APNF/CTF]). Nunca inventes dados.
5. EMERGÊNCIAS: Perante sinais de alarme médico grave ou crise psicológica profunda, orienta de imediato para o 112 ou SNS 24 (808 24 24 24).
""".strip()


# ---------------------------------------------------------------------------
# Construção do Contexto para a API
# ---------------------------------------------------------------------------
def build_api_contents(messages: list[dict], excerpts: list[dict]) -> list[types.Content]:
    recent = messages[-MAX_HISTORY_MESSAGES:]
    contents: list[types.Content] = []
    for i, m in enumerate(recent):
        role = "user" if m["role"] == "user" else "model"
        parts = []
        is_last = (i == len(recent) - 1)
        if is_last and role == "user" and excerpts:
            block = "\n\n".join(f"[Fonte: {c['source']}, p. {c['page']}]\n{c['text']}" for c in excerpts)
            parts.append(types.Part.from_text(
                text=(
                    "EXCERTOS DA BASE PRÓPRIA (utiliza estes dados para fundamentar a resposta):\n"
                    "<<<INÍCIO DOS DADOS>>>\n"
                    f"{block}\n"
                    "<<<FIM DOS DADOS>>>"
                )
            ))
        parts.append(types.Part.from_text(text=m["content"]))
        contents.append(types.Content(role=role, parts=parts))
    while contents and contents[0].role != "user":
        contents.pop(0)
    return contents


def stream_ai_reply(contents: list[types.Content], model: str):
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
        ),
    ):
        if chunk.text:
            yield chunk.text


# ---------------------------------------------------------------------------
# Interface Principal do Chat
# ---------------------------------------------------------------------------
st.title("💙 Mafalda — Conselheira de Apoio à Neurofibromatose")
st.warning(
    "⚠️ A Mafalda **não é médica** nem substitui o acompanhamento clínico. "
    "Em caso de emergência médica ou necessidade urgente, liga **112**."
)
st.markdown(
    "*Estou pronta para ajudar a cruzar a tua base documental própria com o conhecimento científico internacional.*"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escreve a tua questão sobre Neurofibromatose..."):
    if not client:
        st.error("Por favor, insere a Chave API do Google Gemini na barra lateral.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if detect_emergency(prompt):
                st.error(EMERGENCY_MESSAGE)

            # Recuperar excertos da base própria carregada pelo utilizador
            query_text = " ".join([m["content"] for m in st.session_state.messages[-2:] if m["role"] == "user"])
            excerpts = retrieve_excerpts(current_chunks, query_text) if current_chunks else []

            try:
                api_contents = build_api_contents(st.session_state.messages, excerpts)
                with st.spinner("A Mafalda está a cruzar os documentos da base própria..."):
                    reply = st.write_stream(stream_ai_reply(api_contents, model_name))
                
                if not reply:
                    reply = "Não consegui gerar uma resposta com base nos dados disponíveis. Por favor, reformula a questão."
                    st.markdown(reply)

                if excerpts:
                    with st.expander("📄 Segmentos da Base Própria Consultados"):
                        for c in excerpts:
                            st.markdown(f"**{c['source']} — p. {c['page']}**")
                            st.caption(c["text"][:350] + ("…" if len(c["text"]) > 350 else ""))

                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Erro de comunicação com a API: {e}")
                st.session_state.messages.pop()
