import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv
import os
import time
import pandas as pd
from PyPDF2 import PdfReader, errors as pypdf_errors
import docx
from docx.opc.exceptions import PackageNotFoundError
from openpyxl.utils.exceptions import InvalidFileException
import db_manager

from modelos_tokens import obter_limite_tokens, limitar_texto_por_tokens, tokens_para_paginas_a4
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

from agno.agent import Agent
from agno.tools.tavily import TavilyTools
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools
from gpt5_handler import GPT5Handler


_ = load_dotenv(find_dotenv())

agent = Agent(
    model=OpenAIChat(id="gpt-4.1-mini-2025-04-14", temperature=0.3),
    tools=[
        TavilyTools(),
        YFinanceTools(stock_price=True,
                      analyst_recommendations=True, stock_fundamentals=True)
    ],
    show_tool_calls=True,  # Mostra as chamadas de ferramenta no terminal, ótimo para debug
    stream=False,  # O agente do Agno ainda não suporta streaming da resposta final
    description="Você é um analista de investimentos especialista que pesquisa preços de ações, recomendações de analistas e fundamentos de ações usando as ferramentas disponíveis.",
    instructions=[
        "Sempre que receber uma pergunta sobre ações, use a ferramenta YFinanceTools.",
        "Para pesquisas gerais na web, use a ferramenta TavilyTools.",
        "Formate sua resposta final usando markdown e utilize tabelas para exibir dados financeiros sempre que possível."
    ],
)


def habilitar_memory_cache():
    from langchain_community.cache import InMemoryCache
    from langchain.globals import set_llm_cache
    set_llm_cache(InMemoryCache())


habilitar_memory_cache()


@st.cache_resource
def load_model(model_name_param, model_info_param, temperature=0.5):
    try:
        if model_info_param["type"] == "ollama":
            return ChatOllama(model=model_name_param, base_url='http://localhost:11434', temperature=temperature)
        elif model_info_param["type"] == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                st.error("Chave API da OpenAI não encontrada no arquivo .env")
                return None
            return ChatOpenAI(model=model_name_param, temperature=temperature)
        elif model_info_param["type"] == "gemini":
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                st.error("Chave API do Google não encontrada no arquivo .env")
                return None
            return ChatGoogleGenerativeAI(
                model=model_name_param,
                temperature=temperature,
                max_tokens=None,
                max_retries=2,
                api_key=gemini_api_key
            )
        elif model_info_param["type"] == "groq":
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                st.error("Chave API da Groq não encontrada no arquivo .env")
                return None
            return ChatGroq(
                model_name=model_name_param,
                temperature=temperature,
                groq_api_key=groq_api_key
            )
        elif model_info_param["type"] == "gpt5":
            if not os.getenv("OPENAI_API_KEY"):
                st.error("Chave API da OpenAI não encontrada no arquivo .env")
                return None
            return GPT5Handler()
        elif model_info_param["type"] == "anthropic":
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                st.error("Chave API da Anthropic não encontrada no arquivo .env")
                return None
            return ChatAnthropic(model=model_name_param, api_key=anthropic_api_key, temperature=temperature)
    except Exception as e:
        st.error(f"Erro ao carregar o modelo {model_name_param}: {e}")
        return None


def limpar_historico():
    st.session_state["mensagens"] = []
    st.session_state["resposta_completa"] = ""
    st.session_state["current_thread_id"] = None
    st.session_state["file_errors"] = []
    if "modelo_carregado_info" in st.session_state:
        st.session_state.modelo_carregado_info = None


def ler_arquivo(uploaded_file):
    file_content = None
    file_name = uploaded_file.name
    st.session_state["file_errors"] = []
    if uploaded_file is None:
        error_msg = f"Erro: Objeto de arquivo para '{file_name}' é nulo."
        st.session_state["file_errors"].append(error_msg)
        print(f"DEBUG: {error_msg}")
        return ""
    uploaded_file.seek(0)
    text_encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    try:
        if file_name.endswith(('.txt', '.py', '.html', '.css', '.js')):
            for encoding in text_encodings_to_try:
                try:
                    file_content = uploaded_file.getvalue().decode(encoding)
                    print(
                        f"DEBUG: Arquivo '{file_name}' decodificado com sucesso usando {encoding}.")
                    break
                except UnicodeDecodeError:
                    print(
                        f"DEBUG: Erro de decodificação com {encoding} para '{file_name}'.")
                    continue
                except Exception as e:
                    error_msg = f"Erro inesperado ao decodificar '{file_name}' com {encoding}: {e}"
                    st.session_state["file_errors"].append(error_msg)
                    print(f"DEBUG: {error_msg}")
                    return ""
            if file_content is None:
                error_msg = f"Não foi possível decodificar o arquivo de texto '{file_name}'."
                st.session_state["file_errors"].append(error_msg)
                print(f"DEBUG: {error_msg}")
                return ""
        elif file_name.endswith('.pdf'):
            try:
                reader = PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                file_content = text
                print(f"DEBUG: Arquivo PDF '{file_name}' lido com sucesso.")
            except pypdf_errors.PdfReadError as e:
                error_msg = f"Erro ao ler o arquivo PDF '{file_name}': {e}"
                st.session_state["file_errors"].append(error_msg)
                print(f"DEBUG: {error_msg}")
                return ""
            except Exception as e:
                error_msg = f"Erro inesperado ao ler o arquivo PDF '{file_name}': {e}"
                st.session_state["file_errors"].append(error_msg)
                print(f"DEBUG: {error_msg}")
                return ""
        elif file_name.endswith('.docx'):
            try:
                doc_file = docx.Document(uploaded_file)
                file_content = "\n".join([p.text for p in doc_file.paragraphs])
                print(f"DEBUG: Arquivo DOCX '{file_name}' lido com sucesso.")
            except PackageNotFoundError as e:
                error_msg = f"Erro ao ler o arquivo DOCX '{file_name}': {e}"
                st.session_state["file_errors"].append(error_msg)
                print(f"DEBUG: {error_msg}")
                return ""
            except Exception as e:
                error_msg = f"Erro inesperado ao ler o arquivo DOCX '{file_name}': {e}"
                st.session_state["file_errors"].append(error_msg)
                print(f"DEBUG: {error_msg}")
                return ""
        elif file_name.endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(uploaded_file)
                file_content = df.to_string()
                print(f"DEBUG: Arquivo Excel '{file_name}' lido com sucesso.")
            except InvalidFileException as e:
                error_msg = f"Erro ao ler o arquivo Excel '{file_name}': {e}"
                st.session_state["file_errors"].append(error_msg)
                print(f"DEBUG: {error_msg}")
                return ""
            except Exception as e:
                error_msg = f"Erro inesperado ao ler o arquivo Excel '{file_name}': {e}"
                st.session_state["file_errors"].append(error_msg)
                print(f"DEBUG: {error_msg}")
                return ""
        elif file_name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
                file_content = df.to_string()
                print(
                    f"DEBUG: Arquivo CSV '{file_name}' lido com sucesso (UTF-8).")
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(uploaded_file, encoding='latin1')
                    file_content = df.to_string()
                    warning_msg = f"CSV '{file_name}' lido com Latin-1 após falha de UTF-8."
                    st.session_state["file_errors"].append(warning_msg)
                    print(f"DEBUG: {warning_msg}")
                except Exception as e:
                    error_msg = f"Erro ao ler o arquivo CSV '{file_name}': {e}"
                    st.session_state["file_errors"].append(error_msg)
                    print(f"DEBUG: {error_msg}")
                    return ""
            except Exception as e:
                error_msg = f"Erro inesperado ao ler o arquivo CSV '{file_name}': {e}"
                st.session_state["file_errors"].append(error_msg)
                print(f"DEBUG: {error_msg}")
                return ""
        else:
            warning_msg = f"Tipo de arquivo '{file_name}' não suportado para leitura."
            st.session_state["file_errors"].append(warning_msg)
            print(f"DEBUG: {warning_msg}")
            return ""
        if file_content is None or not file_content.strip():
            warning_msg = f"O arquivo '{file_name}' está vazio ou não contém texto legível."
            st.session_state["file_errors"].append(warning_msg)
            print(f"DEBUG: {warning_msg}")
            return ""
        return file_content
    except Exception as e:
        error_msg = f"Erro geral inesperado durante a leitura do arquivo '{file_name}': {e}"
        st.session_state["file_errors"].append(error_msg)
        print(f"DEBUG: {error_msg}")
        return ""


# def referencia_ao_anexo(pergunta):
#     termos = [
#         "no anexo", "no arquivo", "no documento", "nos anexos", "nos arquivos",
#         "segundo o anexo", "de acordo com o anexo", "conforme o anexo", "conforme o documento anexo",
#         "anexo diz", "arquivo diz", "conforme arquivo anexo", "documento diz", "planilha", "csv", "txt",
#         "código", "script", "html", "css", "python", "py"
#     ]
#     pergunta_lower = pergunta.lower()
#     return any(termo in pergunta_lower for termo in termos)


st.set_page_config(page_title="Chat com IA", layout="centered")
st.title("Converse com a IA!")

# Modelos (igual ao seu)
modelos_locais = {
    "deepseek-r1:8b": {"type": "ollama"},
    "deepseek-coder:6.7b": {"type": "ollama"},
    "llama3.1": {"type": "ollama"},
    "llama3.2": {"type": "ollama"},
    "qwen3:8b": {"type": "ollama"},
    "qwen2.5:7b": {"type": "ollama"},
    "qwen2.5-coder:7b": {"type": "ollama"},
    "gemma3:4b": {"type": "ollama"},
    "mistral": {"type": "ollama"},
    "phi4-mini-reasoning:3.8b": {"type": "ollama"},
    "phi4-mini:3.8b": {"type": "ollama"},
    "phi3.5:3.8b": {"type": "ollama"},
    "granite3.3:8b": {"type": "ollama"}
}
modelos_openai = {
    "gpt-5": {"type": "gpt5"},
    "gpt-5-mini": {"type": "gpt5"},
    "gpt-5-nano": {"type": "gpt5"},
    "gpt-5-2025-08-07": {"type": "openai"},
    "gpt-5-mini-2025-08-07": {"type": "openai"},
    "gpt-5-nano-2025-08-07": {"type": "openai"},
    "gpt-4.1-2025-04-14": {"type": "openai"},
    "gpt-4.1-mini-2025-04-14": {"type": "openai"},
    "gpt-4.1-nano-2025-04-14": {"type": "openai"},
    "gpt-4o-mini-2024-07-18": {"type": "openai"},
    "o3-2025-04-16": {"type": "openai"},
    "o4-mini-2025-04-16": {"type": "openai"},
    "gpt-4o": {"type": "openai"},
    "gpt-4": {"type": "openai"},
    "gpt-4-turbo": {"type": "openai"},
    "gpt-3.5-turbo": {"type": "openai"},
    "gpt-4o-mini-search-preview-2025-03-11": {"type": "openai"}
}
modelos_gemini = {
    "gemini-1.5-flash": {"type": "gemini"},
    "gemini-1.5-pro": {"type": "gemini"},
    "gemini-2.0-flash": {"type": "gemini"},
    "gemini-2.0-flash-lite": {"type": "gemini"},
    "gemini-2.5-pro": {"type": "gemini"},
    "gemini-2.5-flash": {"type": "gemini"},
    "gemini-2.5-flash-lite-preview-06-17": {"type": "gemini"},
}
modelos_groq = {
    "deepseek-r1-distill-llama-70b": {"type": "groq"},
    "meta-llama/llama-4-maverick-17b-128e-instruct": {"type": "groq"},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"type": "groq"},
    "qwen/qwen3-32b": {"type": "groq"},
}
modelos_anthropic = {
    "Claude Haiku 3.5": {"type": "anthropic"},
    "Claude Sonnet 3.5 2024-10-22": {"type": "anthropic"},
    "Claude Sonnet 3.7": {"type": "anthropic"},
    "Claude Sonnet 4": {"type": "anthropic"},
    "Claude Opus 4": {"type": "anthropic"},
}
todos_modelos = {
    **{"Local: " + k: v for k, v in modelos_locais.items()},
    **{"OpenAI: " + k: v for k, v in modelos_openai.items()},
    **{"Gemini: " + k: v for k, v in modelos_gemini.items()},
    **{"Groq: " + k: v for k, v in modelos_groq.items()},
    **{"Anthropic: " + k: v for k, v in modelos_anthropic.items()}
}
modelo_selecionado_key_default = list(todos_modelos.keys())[0]
if "modelo_selector" in st.session_state and st.session_state["modelo_selector"] in todos_modelos:
    modelo_selecionado_key = st.session_state["modelo_selector"]
else:
    modelo_selecionado_key = modelo_selecionado_key_default
model_info = todos_modelos[modelo_selecionado_key]
model_name = modelo_selecionado_key.split(
    ": ")[1] if ": " in modelo_selecionado_key else modelo_selecionado_key
modelo_tipo = todos_modelos[modelo_selecionado_key]["type"]
llm = load_model(model_name, model_info, temperature=0.5)

if "modelo_carregado_info" not in st.session_state:
    st.session_state.modelo_carregado_info = None
if "mensagens" not in st.session_state:
    st.session_state["mensagens"] = []
if "resposta_completa" not in st.session_state:
    st.session_state["resposta_completa"] = ""
if "parar_geracao" not in st.session_state:
    st.session_state["parar_geracao"] = False
if "current_thread_id" not in st.session_state:
    st.session_state["current_thread_id"] = None
if "file_errors" not in st.session_state:
    st.session_state["file_errors"] = []
if "textos_arquivos" not in st.session_state:
    st.session_state["textos_arquivos"] = []
if "arquivos_processados" not in st.session_state:
    st.session_state["arquivos_processados"] = []

with st.sidebar:
    st.title("Configurações")
    modelo_selecionado_key = st.selectbox(
        "Escolha o modelo", options=list(todos_modelos.keys()), key="modelo_selector",
        index=list(todos_modelos.keys()).index(
            modelo_selecionado_key) if modelo_selecionado_key in todos_modelos else 0
    )
    model_info = todos_modelos[modelo_selecionado_key]
    model_name = modelo_selecionado_key.split(
        ": ")[1] if ": " in modelo_selecionado_key else modelo_selecionado_key
    modelo_tipo = todos_modelos[modelo_selecionado_key]["type"]
    
    # Verificar se é GPT-5 primeiro para desabilitar temperatura
    modelo_selecionado_temp = modelo_selecionado_key.split(":", 1)[1].strip()
    is_gpt5_temp = any(gpt5_model in modelo_selecionado_temp for gpt5_model in ["gpt-5", "gpt-5-mini", "gpt-5-nano"])
    
    # Slider para controlar a temperatura do modelo
    if not is_gpt5_temp:
        temperatura = st.slider(
            "Temperatura do modelo", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Controla a criatividade da resposta. Valores mais baixos = respostas mais conservadoras.")
    else:
        st.sidebar.write("🌡️ Temperatura: 1.0 (fixo para GPT-5)")
        temperatura = 1.0
    
    # Usar a mesma verificação is_gpt5
    is_gpt5 = is_gpt5_temp
    
    if is_gpt5:
        st.sidebar.subheader("🚀 Configurações GPT-5")
        st.sidebar.info("ℹ️ GPT-5 usa temperature fixa = 1.0 (não configurável)")
        
        # Seletor de Verbosity
        verbosity = st.selectbox(
            "Nível de Verbosidade",
            options=["low", "medium", "high"],
            index=1,  # medium como padrão
            help="• Low: Respostas concisas\n• Medium: Detalhamento balanceado\n• High: Respostas detalhadas"
        )
        
        # Opção de Minimal Reasoning
        use_minimal_reasoning = st.checkbox(
            "Reasoning Mínimo ⚡",
            value=False,
            help="Ativa reasoning mínimo para respostas mais rápidas. Ideal para tarefas simples como classificação ou extração."
        )
        
        # Opções de Custom Tools
        st.sidebar.subheader("🔧 Ferramentas Personalizadas")
        use_custom_tools = st.checkbox(
            "Ativar Ferramentas Personalizadas",
            value=False,
            help="Permite uso de free-form function calling e CFG"
        )
        
        if use_custom_tools:
            tool_type = st.selectbox(
                "Tipo de Ferramenta",
                options=["SQL Generator", "Timestamp Generator", "Code Executor", "Custom"],
                help="Escolha o tipo de ferramenta personalizada"
            )
    else:
        verbosity = "medium"
        use_minimal_reasoning = False
        use_custom_tools = False
        tool_type = None
    limite_tokens = obter_limite_tokens(model_name)
    limite_tokens_formatado = f"{limite_tokens:,}".replace(",", ".")
    paginas = f'{int(tokens_para_paginas_a4(limite_tokens)):,}'.replace(
        ",", ".")
    st.sidebar.write(
        f"Limite de tokens para contexto: {limite_tokens_formatado} tokens ≅ {paginas} páginas A4")
    arquivos = st.file_uploader(
        "Anexe arquivos para contexto (PDF, DOCX, Excel, TXT, CSV, PY, HTML, CSS)",
        type=['pdf', 'docx', 'xlsx', 'xls', 'txt', 'csv', 'py', 'html', 'css'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    st.divider()
    st.subheader("Carregar / Gerenciar Conversas")
    conversas_salvas = db_manager.list_conversations()
    opcoes_carregamento = [
        {"id": None, "title": "-- Nova Conversa --", "updated_at": ""}] + conversas_salvas
    current_thread_index = 0
    if st.session_state["current_thread_id"]:
        for i, conv in enumerate(opcoes_carregamento):
            if conv["id"] == st.session_state["current_thread_id"]:
                current_thread_index = i
                break
    selected_conversation_option = st.selectbox(
        "Escolha uma conversa:",
        options=[f"{opt['title']} ({opt['updated_at']})" if opt['id']
                 else opt['title'] for opt in opcoes_carregamento],
        index=current_thread_index,
        key="selectbox_carregar_conversa"
    )
    selected_conversation_id = None
    for opt in opcoes_carregamento:
        if (opt['id'] and selected_conversation_option == f"{opt['title']} ({opt['updated_at']})") or \
           (opt['id'] is None and selected_conversation_option == opt['title']):
            selected_conversation_id = opt['id']
            break
    col_load_delete1, col_load_delete2 = st.columns(2)
    with col_load_delete1:
        if st.button("Carregar Selecionada", key="carregar_conversa_db"):
            if selected_conversation_id:
                dados_conversa_carregada = db_manager.load_conversation(
                    selected_conversation_id)
                if dados_conversa_carregada:
                    limpar_historico()
                    st.session_state["mensagens"] = dados_conversa_carregada.get(
                        "messages", [])
                    st.session_state["current_thread_id"] = selected_conversation_id
                    st.session_state[
                        "modelo_carregado_info"] = f"Conversa carregada: {dados_conversa_carregada['title']} (Modelo: {dados_conversa_carregada['model_used']})"
                    st.rerun()
                else:
                    st.error("Erro ao carregar a conversa.")
            else:
                limpar_historico()
                st.session_state["modelo_carregado_info"] = None
                st.rerun()
    with col_load_delete2:
        if selected_conversation_id and st.button("Excluir Selecionada", key="excluir_conversa_db"):
            if db_manager.delete_conversation(selected_conversation_id):
                st.success("Conversa excluída com sucesso!")
                limpar_historico()
                st.rerun()
            else:
                st.error("Erro ao excluir a conversa.")
    if "modelo_carregado_info" in st.session_state and st.session_state["modelo_carregado_info"]:
        st.info(st.session_state["modelo_carregado_info"])

file_error_placeholder = st.empty()

# --------- PROCESSAMENTO DOS ARQUIVOS ---------
# Só processa arquivos se mudou!
if arquivos:
    arquivos_nomes = [a.name for a in arquivos]
    if st.session_state["arquivos_processados"] != arquivos_nomes:
        textos_arquivos = []
        for arquivo in arquivos:
            st.write(f"- **{arquivo.name}**")
            texto = ler_arquivo(arquivo)
            textos_arquivos.append(texto)
            if texto:
                st.write(texto[:500] + ("..." if len(texto) > 500 else ""))
            else:
                st.write("Conteúdo não lido ou vazio.")
        st.session_state["textos_arquivos"] = textos_arquivos
        st.session_state["arquivos_processados"] = arquivos_nomes
elif not arquivos and not st.session_state["textos_arquivos"]:
    st.session_state["arquivos_processados"] = []

# Exibir arquivos e botao limpar se houver textos
if st.session_state.get("textos_arquivos") and any(st.session_state["textos_arquivos"]):
    st.markdown("### Arquivos anexados:")
    for idx, texto in enumerate(st.session_state["textos_arquivos"]):
        st.write(f"- **Arquivo {idx+1}**")
        st.write(texto[:500] + ("..." if len(texto) > 500 else ""))
    if not st.session_state["file_errors"]:
        st.info(
            "Os arquivos anexados foram processados e estão prontos para serem usados como contexto.")
    if st.button("Limpar Arquivos Anexados", key="clear_uploaded_files"):
        if "file_uploader" in st.session_state:
            del st.session_state["file_uploader"]
        st.session_state["file_errors"] = []
        st.session_state["textos_arquivos"] = []
        st.session_state["arquivos_processados"] = []
        st.rerun()
    st.divider()

if st.session_state["file_errors"]:
    with file_error_placeholder.container():
        for error_msg in st.session_state["file_errors"]:
            st.error(error_msg)

# DEBUG VISUAL
# st.write("DEBUG - system prompt:", st.session_state.get("textos_arquivos"))st.write(system_prompt)

for tipo, conteudo in st.session_state["mensagens"]:
    with st.chat_message(tipo):
        st.markdown(conteudo)

col1, col2 = st.columns([2, 10])
with col1:
    usar_pesquisa_web = st.toggle("🌐 Web", key="usar_pesquisa_web")

prompt = st.chat_input("Mande sua mensagem para a IA...")

if prompt or st.session_state["mensagens"]:
    with st.container():
        if st.button("Limpar Histórico / Nova Conversa", key="limpar_historico_button"):
            limpar_historico()
            st.rerun()

if prompt and llm:
    if st.session_state.get("usar_pesquisa_web"):
        consulta = f"Search tavily for '{prompt}'"
        st.session_state["mensagens"].append(
            ("human", f"[PESQUISA WEB] {prompt}"))
        with st.chat_message("ai"):
            st.info(f"Pesquisando na web por: **{prompt}**")
            try:
                resposta_web = agent.run(consulta)
                resposta_texto = resposta_web.content  # Só o texto
                # Pega só a primeira frase como resposta sucinta
                # resposta_sucinta = resposta_texto.split(".")[0] + "."
                st.markdown(resposta_texto)
            except Exception as e:
                st.error(f"Erro ao consultar Tavily: {e}")
        st.stop()
    if st.session_state.get("selectbox_carregar_conversa") == "-- Nova Conversa --" and "modelo_carregado_info" in st.session_state:
        del st.session_state["modelo_carregado_info"]
    st.session_state["mensagens"].append(("human", prompt))
    with st.chat_message("human"):
        st.markdown(prompt)
    mensagens_para_llm = []
    limite_tokens = obter_limite_tokens(model_name)

    if "textos_arquivos" in st.session_state and any(st.session_state["textos_arquivos"]):
        contexto_arquivos = "\n\n".join(
            filter(None, st.session_state["textos_arquivos"]))
        contexto_limitado = limitar_texto_por_tokens(
            contexto_arquivos, limite_tokens)

        # if referencia_ao_anexo(prompt):
        system_prompt = (
            "Você é um assistente de IA. O usuário enviou o(s) seguinte(s) arquivo(s), cujo conteúdo está abaixo entre as linhas ===INÍCIO=== e ===FIM===. "
            "Use APENAS essas informações para responder à pergunta do usuário. "
            "Se não encontrar a resposta, diga 'Não encontrei essa informação nos arquivos'.\n\n"
            "===INÍCIO===\n"
            f"{contexto_limitado}\n"
            "===FIM===\n"
        )

        mensagens_para_llm.insert(0, SystemMessage(content=system_prompt))
    else:
        system_prompt = (
            "Você é um assistente de IA. Responda normalmente ao usuário."
        )

        mensagens_para_llm.insert(0, SystemMessage(content=system_prompt))

    for tipo, conteudo in st.session_state["mensagens"]:
        if tipo == "human":
            mensagens_para_llm.append(HumanMessage(content=conteudo))
        elif tipo == "ai":
            mensagens_para_llm.append(AIMessage(content=conteudo))
    col_stop = st.empty()
    if col_stop.button("Parar geração", key="stop_button"):
        st.session_state["parar_geracao"] = True
    st.session_state["parar_geracao"] = False
    with st.chat_message("ai"):
        resposta_placeholder = st.empty()
        st.session_state["resposta_completa"] = ""
    progress_bar = st.progress(0)
    with st.spinner("IA gerando resposta..."):
        progress = 0
        direction = 1
        try:
            # Verifica se é GPT-5 para usar novos recursos
            if isinstance(llm, GPT5Handler):
                # Prepara o histórico no formato correto
                conversation_history = []
                for tipo, conteudo in st.session_state["mensagens"][:-1]:  # Exclui a mensagem atual
                    role = "user" if tipo == "human" else "assistant"
                    conversation_history.append({"role": role, "content": conteudo})
                
                # Verifica se deve usar ferramentas personalizadas
                if use_custom_tools and tool_type:
                    if tool_type == "SQL Generator":
                        result = llm.create_sql_query_with_dialect(
                            query_description=prompt,
                            dialect="postgresql",
                            model=model_name
                        )
                        if "tool_call" in result:
                            st.session_state["resposta_completa"] = f"**SQL Gerado:**\n```sql\n{result['tool_call']['input']}\n```\n\n**Explicação:**\n{result.get('response', 'Query SQL gerada com sucesso.')}"
                        else:
                            st.session_state["resposta_completa"] = result.get('response', 'Erro na geração SQL')
                    
                    elif tool_type == "Timestamp Generator":
                        result = llm.create_response_with_custom_tool(
                            input_message=prompt,
                            tool_name="timestamp_generator",
                            tool_description="Gera timestamp no formato YYYY-MM-DD HH:MM",
                            model=model_name,
                            grammar_definition=r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]) (?:[01]\d|2[0-3]):[0-5]\d$",
                            grammar_syntax="regex"
                        )
                        if "tool_call" in result:
                            st.session_state["resposta_completa"] = f"**Timestamp:**\n{result['tool_call']['input']}\n\n{result.get('response', '')}"
                        else:
                            st.session_state["resposta_completa"] = result.get('response', 'Erro na geração do timestamp')
                    
                    else:
                        # Usa resposta padrão para outros tipos
                        if use_minimal_reasoning:
                            result = llm.create_response_with_minimal_reasoning(
                                input_message=prompt,
                                model=model_name,
                                conversation_history=conversation_history
                            )
                        else:
                            result = llm.create_response_with_verbosity(
                                input_message=prompt,
                                model=model_name,
                                verbosity=verbosity,
                                conversation_history=conversation_history
                            )
                        st.session_state["resposta_completa"] = result.get('response', 'Erro na resposta')
                
                else:
                    # Usa resposta padrão com verbosity ou minimal reasoning
                    if use_minimal_reasoning:
                        result = llm.create_response_with_minimal_reasoning(
                            input_message=prompt,
                            model=model_name,
                            conversation_history=conversation_history
                        )
                        # Adiciona indicador de reasoning mínimo
                        usage_info = result.get('usage', {})
                        tokens_info = f" (⚡ {usage_info.get('output_tokens', 'N/A')} tokens)"
                        st.session_state["resposta_completa"] = result.get('response', 'Erro na resposta') + f"\n\n*Resposta rápida{tokens_info}*"
                    else:
                        result = llm.create_response_with_verbosity(
                            input_message=prompt,
                            model=model_name,
                            verbosity=verbosity,
                            conversation_history=conversation_history
                        )
                        # Adiciona informações de verbosity e tokens
                        usage_info = result.get('usage', {})
                        tokens_info = f" ({usage_info.get('output_tokens', 'N/A')} tokens)"
                        verbosity_emoji = {"low": "📝", "medium": "📄", "high": "📚"}
                        st.session_state["resposta_completa"] = result.get('response', 'Erro na resposta') + f"\n\n*{verbosity_emoji.get(verbosity, '📄')} Verbosity: {verbosity}{tokens_info}*"
                
                # Simula progresso para GPT-5 (resposta não é streaming)
                for i in range(10):
                    if st.session_state["parar_geracao"]:
                        st.session_state["resposta_completa"] += "\n\n**Geração interrompida pelo usuário.**"
                        break
                    progress = (i + 1) * 0.1
                    progress_bar.progress(progress)
                    time.sleep(0.1)
                
                # Exibe a resposta final
                resposta_placeholder.markdown(st.session_state["resposta_completa"])
                
            else:
                # Código original para modelos não-GPT5
                for chunk in llm.stream(mensagens_para_llm):
                if st.session_state["parar_geracao"]:
                    st.session_state["resposta_completa"] += "\n\n**Geração interrompida pelo usuário.**"
                    resposta_placeholder.markdown(
                        st.session_state["resposta_completa"])
                    break
                if hasattr(chunk, 'content') and chunk.content is not None:
                    st.session_state["resposta_completa"] += chunk.content
                    resposta_placeholder.markdown(
                        st.session_state["resposta_completa"] + "▌")
                progress += direction * 0.05
                if progress >= 1:
                    direction = -1
                elif progress <= 0:
                    direction = 1
                progress_bar.progress(progress)
                time.sleep(0.02)
        except Exception as e:
            st.error(f"Erro ao gerar resposta do modelo: {e}")
            st.session_state["resposta_completa"] += "\n\nDesculpe, ocorreu um erro ao gerar a resposta."
            resposta_placeholder.markdown(
                st.session_state["resposta_completa"])
        finally:
            progress_bar.empty()
            resposta_placeholder.markdown(
                st.session_state["resposta_completa"])
            st.session_state["parar_geracao"] = False
    if st.session_state["resposta_completa"] and not st.session_state["resposta_completa"].endswith("**Geração interrompida pelo usuário.**"):
        st.session_state["mensagens"].append(
            ("ai", st.session_state["resposta_completa"]))
    elif st.session_state["resposta_completa"].endswith("**Geração interrompida pelo usuário.**"):
        st.session_state["mensagens"].append(
            ("ai", st.session_state["resposta_completa"]))
    if st.session_state["mensagens"]:
        if st.session_state["current_thread_id"] is None:
            initial_prompt = ""
            for msg_type, msg_content in st.session_state["mensagens"]:
                if msg_type == "human":
                    initial_prompt = msg_content
                    break
            conversation_title = initial_prompt[:100] + \
                "..." if len(initial_prompt) > 100 else initial_prompt
            if not conversation_title:
                conversation_title = "Conversa Sem Título"
            new_thread_id = db_manager.save_conversation(
                st.session_state["mensagens"],
                model_used=model_name,
                title=conversation_title
            )
            if new_thread_id:
                st.session_state["current_thread_id"] = new_thread_id
                st.success(
                    f"Nova conversa salva com sucesso! ID: {new_thread_id}")
            else:
                st.error("Erro ao salvar a nova conversa.")
        else:
            if db_manager.update_conversation(
                st.session_state["current_thread_id"],
                st.session_state["mensagens"],
                model_used=model_name
            ):
                st.success("Conversa atualizada com sucesso!")
            else:
                st.error("Erro ao atualizar a conversa.")
    st.rerun()
