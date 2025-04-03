#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
import streamlit as st

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Carrega variáveis de ambiente do .env
load_dotenv()

# Chaves e configurações básicas
FIRELIES_API_KEY = os.getenv("FIRELIES_API_KEY", "")
FIRELIES_GRAPHQL_ENDPOINT = os.getenv("FIRELIES_GRAPHQL_ENDPOINT", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Configurações do modelo (poderiam vir do .env também)
MODEL_ID = "gemini-2.0-flash"
TEMPERATURE = 0.7


# =================== FUNÇÕES AUXILIARES ====================

def get_fireflies_transcripts():
    """
    Busca lista de transcrições existentes no Fireflies via GraphQL.
    Retorna uma lista de dicionários contendo 'id' e 'title'.
    """
    if not FIRELIES_API_KEY or not FIRELIES_GRAPHQL_ENDPOINT:
        st.error("Faltam configurações de acesso à API Fireflies.")
        return []

    # Query GraphQL simples para obter transcrições (ajuste conforme o schema real)
    query = """
    query {
      transcripts {
        id
        title
      }
    }
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIRELIES_API_KEY}"
    }

    try:
        resp = requests.post(
            FIRELIES_GRAPHQL_ENDPOINT,
            json={"query": query},
            headers=headers
        )
        resp.raise_for_status()  # Lança exceção se status_code >= 400
        data = resp.json()

        # Verifica se teve algum erro no retorno GraphQL
        if "errors" in data:
            st.error("Erro(s) retornado(s) pelo GraphQL: {}".format(data["errors"]))
            return []

        transcripts = data.get("data", {}).get("transcripts", [])
        return transcripts

    except requests.RequestException as e:
        st.error(f"Falha na requisição ao Fireflies: {e}")
        return []


def get_transcript_text_by_id(transcript_id):
    """
    Busca as sentenças do transcript específico e retorna
    o texto completo (concatenado).
    """
    query = """
    query GetTranscript($id: String!) {
      transcript(id: $id) {
        sentences {
          text
        }
      }
    }
    """
    variables = {"id": transcript_id}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIRELIES_API_KEY}"
    }

    try:
        resp = requests.post(
            FIRELIES_GRAPHQL_ENDPOINT,
            json={"query": query, "variables": variables},
            headers=headers
        )
        resp.raise_for_status()
        data = resp.json()

        if "errors" in data:
            st.error("Erro(s) retornado(s) pelo GraphQL: {}".format(data["errors"]))
            return ""

        transcript = data.get("data", {}).get("transcript", {})
        sentences = transcript.get("sentences", [])
        # Junta todas as sentenças em uma string
        text = "\n".join([s["text"] for s in sentences])
        return text

    except requests.RequestException as e:
        st.error(f"Falha na requisição ao Fireflies: {e}")
        return ""


def generate_sales_insights(transcript_text):
    """
    Chama a API Google Generative AI (Gemini) para gerar insights
    a partir do texto de transcrição fornecido.
    """
    if not GOOGLE_API_KEY:
        return "Chave de API do Google não configurada."

    # Configura a biblioteca do Google Generative AI
    genai.configure(api_key=GOOGLE_API_KEY)

    # Define o prompt de análise de vendas
    prompt = f"""
Você é um especialista em análise de vendas com vasta experiência.
Sua tarefa é analisar a seguinte transcrição de uma interação de vendas:

\"\"\"
{transcript_text}
\"\"\"

Com base na transcrição, liste **INSIGHTS ACIONÁVEIS** de vendas. Organize sua resposta nas seguintes categorias:

- **Oportunidades de Venda**
- **Necessidades Explícitas**
- **Necessidades Implícitas**
- **Objeções**
- **Upsell/Cross-sell**
- **Estratégias Comerciais**
- **Sentimento Geral**

Responda em formato markdown.
"""

    try:
        # Cria instância do modelo e configura
        model = genai.GenerativeModel(MODEL_ID)
        config = GenerationConfig(temperature=TEMPERATURE)

        response = model.generate_content(prompt, generation_config=config)

        # Se houver bloqueio por alguma razão
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"**Bloqueado pelo modelo**. Razão: {response.prompt_feedback.block_reason}"

        # Verifica se há candidatos
        if not response.candidates:
            return "Nenhum candidato de resposta retornado."

        candidate = response.candidates[0]
        # Tenta extrair o texto principal
        if hasattr(response, 'text') and response.text.strip():
            return response.text.strip()

        # Fallback para candidate.content.parts
        if (hasattr(candidate, 'content') and
            hasattr(candidate.content, 'parts') and
                candidate.content.parts):
            return candidate.content.parts[0].text.strip()

        return "Não foi possível extrair insights do modelo."

    except Exception as e:
        return f"Ocorreu uma exceção ao gerar insights: {e}"


# ================== APLICAÇÃO STREAMLIT ====================

def main():
    st.set_page_config(page_title="Neural Sales - Monolítico", layout="wide")
    st.title("Neural Sales - Web App (Monolítico)")

    st.write("""
    **Versão Inicial**  
    Esta aplicação lista transcrições do Fireflies e permite gerar insights via modelo Gemini do Google.
    """)

    # Botão para recarregar a lista de transcrições
    if st.button("Carregar Lista de Transcrições"):
        st.session_state["transcripts"] = get_fireflies_transcripts()

    # Garante que haja a chave no session_state
    if "transcripts" not in st.session_state:
        st.session_state["transcripts"] = []

    transcripts = st.session_state["transcripts"]
    if not transcripts:
        st.info("Nenhuma transcrição listada ou botão 'Carregar Lista de Transcrições' não foi clicado.")
        return

    st.subheader("Transcrições Disponíveis")
    for transcript in transcripts:
        transcript_id = transcript["id"]
        title = transcript.get("title", "Título não informado")

        col1, col2 = st.columns([3,1])
        with col1:
            st.write(f"**ID**: {transcript_id}")
            st.write(f"**Título**: {title}")

        with col2:
            if st.button("Gerar Insights", key=f"insights_{transcript_id}"):
                # Buscar texto completo daquela transcrição
                text = get_transcript_text_by_id(transcript_id)
                if not text.strip():
                    st.warning("Transcrição vazia ou não encontrada.")
                else:
                    # Chamar o modelo para gerar insights
                    insights = generate_sales_insights(text)
                    st.success("Insights Gerados!")
                    st.markdown(insights)

        st.markdown("---")  # Separador visual


if __name__ == "__main__":
    main()
