import streamlit as st
import pandas as pd
from gnews import GNews
import google.generativeai as genai
import os
import re
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Carrega modelo Spacy ---
@st.cache_resource
def carregar_spacy():
    try:
        return spacy.load("pt_core_news_sm")
    except OSError:
        from spacy.cli import download
        download("pt_core_news_sm")
        return spacy.load("pt_core_news_sm")

nlp = carregar_spacy()

# --- Configuração da API Gemini ---
def configurar_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ Chave de API do Gemini não encontrada. Configure em `secrets.toml`.")
        st.stop()
    genai.configure(api_key=api_key)

configurar_gemini()

# --- Funções auxiliares ---
def carregar_lista_deputados(arquivo):
    try:
        if arquivo.name.endswith(".csv"):
            df = pd.read_csv(arquivo)
        elif arquivo.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(arquivo)
        else:
            st.error("Formato de arquivo não suportado. Use .csv, .xls ou .xlsx.")
            return None, None

        colunas = [c.lower() for c in df.columns]
        if "nome parlamentar" in colunas:
            col = df.columns[colunas.index("nome parlamentar")]
        elif "nome" in colunas:
            col = df.columns[colunas.index("nome")]
        else:
            st.error("Não foi possível encontrar coluna de nome ('nome parlamentar' ou 'nome').")
            return None, None
        return df, col
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None, None


def buscar_noticias(nome):
    st.info(f"🔍 Buscando notícias sobre **{nome}**...")
    google_news = GNews(language='pt', country='BR', max_results=10)
    noticias = google_news.get_news(f"deputado {nome}")
    if not noticias:
        st.warning("Nenhuma notícia encontrada.")
        return None, None

    texto = " ".join([f"{n['title']} {n['description']}" for n in noticias if n['description']])
    prompt = "\n".join([f"- Título: {n['title']}\n  Descrição: {n['description']}" for n in noticias])
    return texto, prompt


def resumir_noticias(prompt, nome):
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt_completo = f"""
    Você é um analista político. Resuma, em até 3 parágrafos, as principais informações sobre o deputado {nome}
    com base nas seguintes notícias:

    {prompt}

    Resumo:
    """
    resposta = model.generate_content(prompt_completo)
    return resposta.text


def limpar_texto_para_nuvem(texto, nome):
    palavras_nome = nome.lower().split()
    stop_words = nlp.Defaults.stop_words.union(palavras_nome)
    doc = nlp(texto.lower())
    tokens = [t.lemma_ for t in doc if t.text not in stop_words and not t.is_punct and not t.is_space]
    return " ".join(tokens)


def gerar_nuvem(texto, nome):
    texto_limpo = limpar_texto_para_nuvem(texto, nome)
    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(texto_limpo)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


# --- Interface Streamlit ---
st.title("📰 Analisador de Notícias de Parlamentares")
st.write("Este app busca notícias recentes sobre deputados e gera um resumo automático usando a API Gemini, além de uma nuvem de palavras.")

arquivo = st.file_uploader("📂 Envie a planilha de deputados (.csv, .xls ou .xlsx)", type=["csv", "xls", "xlsx"])

if arquivo:
    df, col = carregar_lista_deputados(arquivo)
    if df is not None:
        deputado = st.selectbox("Escolha um deputado:", df[col].dropna().unique())
        if st.button("Buscar e Analisar Notícias"):
            texto, prompt = buscar_noticias(deputado)
            if texto:
                resumo = resumir_noticias(prompt, deputado)
                st.subheader("🧭 Resumo das Notícias (via Gemini)")
                st.write(resumo)
                st.subheader("☁️ Nuvem de Palavras")
                gerar_nuvem(texto, deputado)
else:
    st.info("Envie uma planilha para começar.")
