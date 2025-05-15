# gerenciador_conversas.py

import json
import os
from datetime import datetime
import streamlit as st  # Precisamos do st para st.success, st.error, st.warning

# Diretório para salvar as conversas
CONVERSATIONS_DIR = "conversas_salvas"
if not os.path.exists(CONVERSATIONS_DIR):
    os.makedirs(CONVERSATIONS_DIR)


def salvar_conversa_json(mensagens, nome_base="conversa", modelo_usado="desconhecido"):
    """Salva a lista de mensagens em um arquivo JSON."""
    if not mensagens:
        st.warning("Nenhuma mensagem para salvar.")
        return None  # Retorna None se não salvou

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Poderia adicionar um input para o usuário dar um nome/título
    # Para simplificar, vamos usar o nome_base e timestamp
    nome_arquivo_base = f"{nome_base.replace(' ', '_')}_{timestamp}.json"
    nome_arquivo_completo = os.path.join(CONVERSATIONS_DIR, nome_arquivo_base)

    conversa_para_salvar = {
        "modelo_usado": modelo_usado,
        "timestamp_salvo": datetime.now().isoformat(),
        "mensagens": mensagens  # Assumindo que 'mensagens' já é uma lista de tuplas/dicionários
    }

    try:
        with open(nome_arquivo_completo, "w", encoding="utf-8") as f:
            json.dump(conversa_para_salvar, f, ensure_ascii=False, indent=4)
        st.success(f"Conversa salva em: {nome_arquivo_completo}")
        return nome_arquivo_completo  # Retorna o nome do arquivo salvo
    except Exception as e:
        st.error(f"Erro ao salvar a conversa: {e}")
        return None


def carregar_conversa_json(nome_arquivo_completo):
    """
    Carrega uma conversa de um arquivo JSON.
    Retorna a lista de mensagens e metadados, ou None em caso de erro.
    """
    try:
        with open(nome_arquivo_completo, "r", encoding="utf-8") as f:
            conversa_carregada = json.load(f)
        # Retorna o dicionário completo para que o app.py possa decidir o que fazer com os metadados
        return conversa_carregada
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {nome_arquivo_completo}")
        return None
    except json.JSONDecodeError:
        st.error(
            f"Erro ao decodificar JSON do arquivo: {nome_arquivo_completo}")
        return None
    except Exception as e:
        st.error(
            f"Erro ao carregar a conversa de {nome_arquivo_completo}: {e}")
        return None


def listar_conversas_salvas():
    """Retorna uma lista dos nomes dos arquivos de conversa salvos."""
    if not os.path.exists(CONVERSATIONS_DIR):
        return []
    return [f for f in os.listdir(CONVERSATIONS_DIR) if f.endswith(".json")]
