import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import AIMessageChunk
from dotenv import load_dotenv, find_dotenv
import os
import time  # Importado time para a animação da barra de progresso
# Importa a função para obter o limite de tokens
from modelos_tokens import obter_limite_tokens, limitar_texto_por_tokens, tokens_para_paginas_a4
from langchain_google_genai import ChatGoogleGenerativeAI

import pandas as pd
from PyPDF2 import PdfReader
import docx
import openpyxl

import gerenciador_conversas as gc

# Carrega variáveis de ambiente no início
_ = load_dotenv(find_dotenv())

# No início do seu app.py, após os imports, você pode querer limpar
# a informação de modelo carregado se não for uma recarga de uma conversa
if "modelo_carregado_info" not in st.session_state:
    st.session_state.modelo_carregado_info = None

# Habilita cache em memória para LLMs


def habilitar_memory_cache():
    """Habilita o cache em memória para as respostas do LLM."""
    from langchain_community.cache import InMemoryCache
    from langchain.globals import set_llm_cache
    set_llm_cache(InMemoryCache())


habilitar_memory_cache()

# Configuração da página do Streamlit
st.set_page_config(page_title="Chat com IA", layout="centered")
st.title("Converse com a IA!")

# Definição dos modelos disponíveis (locais e OpenAI)
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
    "phi4-mini:3.8b": {"type": "ollama"},  # recurso de chamada de função
    "phi3.5:3.8b": {"type": "ollama"},
    "granite3.3:8b": {"type": "ollama"}
}

modelos_openai = {
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
    "gemini-2.5-pro-preview-03-25": {"type": "gemini"},
    "gemini-2.5-flash-preview-04-17": {"type": "gemini"},
    "gemini-2.0-flash-preview-image-generation": {"type": "gemini"},

}

# Combina todos os modelos em um dicionário
todos_modelos = {
    **{"Local: " + k: v for k, v in modelos_locais.items()},
    **{"OpenAI: " + k: v for k, v in modelos_openai.items()},
    **{"Gemini: " + k: v for k, v in modelos_gemini.items()}
}


# Função para carregar o modelo LLM (com cache)
# Esta função aceita o parâmetro 'temperature'


@st.cache_resource
def load_model(model_name, model_info, temperature=0.5):
    try:
        if model_info["type"] == "ollama":
            return ChatOllama(model=model_name, base_url='http://localhost:11434', temperature=temperature)
        elif model_info["type"] == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                st.error("Chave API da OpenAI não encontrada no arquivo .env")
                return None
            return ChatOpenAI(model=model_name, temperature=temperature)
        elif model_info["type"] == "gemini":
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not os.getenv("GEMINI_API_KEY"):
                st.error("Chave API do Google não encontrada no arquivo .env")
                return None
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_tokens=None,
                max_retries=2,
                api_key=gemini_api_key
            )
    except Exception as e:
        st.error(f"Erro ao carregar o modelo {model_name}: {e}")
        return None

# Função para limpar o histórico de mensagens


def limpar_historico():
    """Limpa o histórico de mensagens e a resposta completa na sessão."""
    st.session_state["mensagens"] = []
    st.session_state["resposta_completa"] = ""

    if "modelo_carregado_info" in st.session_state:
        st.session_state.modelo_carregado_info = None


# Inicialização do histórico e variável de resposta na sessão
if "mensagens" not in st.session_state:
    st.session_state["mensagens"] = []
if "resposta_completa" not in st.session_state:
    st.session_state["resposta_completa"] = ""
# Inicializa a variável de estado para parar a geração
if "parar_geracao" not in st.session_state:
    st.session_state["parar_geracao"] = False


# ========== SIDEBAR ==========
with st.sidebar:
    st.title("Configurações")
    # Selectbox para escolher o modelo
    # Usando uma key única para evitar o erro de duplicidade
    modelo_selecionado_key = st.sidebar.selectbox(
        "Escolha o modelo", options=list(todos_modelos.keys()), key="modelo_selector")

    # Slider para controlar a temperatura do modelo
    temperatura = st.slider(
        "Temperatura do modelo",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Controla a criatividade da resposta. Valores mais baixos = respostas mais conservadoras.")

    # modelos = list(limites_modelos.keys())
    modelo_selecionado = modelo_selecionado_key.split(":", 1)[1].strip()

    limite_tokens = obter_limite_tokens(modelo_selecionado)
    limite_tokens_formatado = f"{limite_tokens:,}".replace(",", ".")
    paginas = f'{int(tokens_para_paginas_a4(limite_tokens)):,}'.replace(
        ",", ".")
    st.sidebar.write(
        f"Limite de tokens para contexto: {limite_tokens_formatado} tokens ≅ {paginas} páginas A4")

    # Upload de arquivos para RAG
    arquivos = st.file_uploader(
        "Anexe arquivos para RAG (PDF, DOCX, Excel, TXT ou CSV)",
        type=['pdf', 'docx', 'xlsx', 'xls', 'txt', 'csv'],
        accept_multiple_files=True,
        key="file_uploader"  # Adicionado key para o file_uploader também, boa prática
    )

    st.divider()
    st.subheader("Carregar Conversa Salva")
    arquivos_salvos = gc.listar_conversas_salvas()

    if arquivos_salvos:
        # Adiciona uma opção "Nova Conversa" ou "Não carregar"
        opcoes_carregamento = ["-- Nova Conversa --"] + arquivos_salvos
        arquivo_selecionado_para_carregar = st.selectbox(
            "Escolha uma conversa para carregar:",
            options=opcoes_carregamento,
            index=0,  # Padrão para "Nova Conversa"
            key="selectbox_carregar_conversa"
        )

        # Botão para carregar só aparece se um arquivo real for selecionado
        if arquivo_selecionado_para_carregar != "-- Nova Conversa --":
            if st.button("Carregar Conversa Selecionada", key="carregar_json_selecionado"):
                caminho_completo = os.path.join(
                    gc.CONVERSATIONS_DIR, arquivo_selecionado_para_carregar)
                dados_conversa_carregada = gc.carregar_conversa_json(
                    caminho_completo)

                if dados_conversa_carregada:
                    # Limpa o histórico atual antes de carregar um novo
                    limpar_historico()  # Sua função limpar_historico()
                    st.session_state["mensagens"] = dados_conversa_carregada.get(
                        "mensagens", [])
                    st.session_state[
                        "modelo_carregado_info"] = f"Conversa carregada: {arquivo_selecionado_para_carregar}"
                    # Você pode querer definir o selectbox do modelo para o modelo_da_conversa se ele existir
                    # Isso pode ser complexo devido à forma como o selectbox é preenchido
                    st.rerun()
    else:
        st.write("Nenhuma conversa salva encontrada.")

    # Exibir informação do modelo da conversa carregada, se houver
    if "modelo_carregado_info" in st.session_state and st.session_state["modelo_carregado_info"]:
        st.info(st.session_state["modelo_carregado_info"])


# Carregar o modelo selecionado após a escolha do usuário na sidebar
# Usa a variável do selectbox corrigido (modelo_selecionado_key)
model_info = todos_modelos[modelo_selecionado_key]
# Extrai o nome real do modelo removendo o prefixo "Local: " ou "OpenAI: "
model_name = modelo_selecionado_key.split(
    ": ")[1] if ": " in modelo_selecionado_key else modelo_selecionado_key

# Carrega a instância do LLM, passando a temperatura
# A função load_model agora tem um valor padrão para temperature, mas passamos o valor do slider
llm = load_model(model_name, model_info, temperature=temperatura)


# ========== Extração dos arquivos enviados ==========
def ler_arquivo(uploaded_file):
    """Lê o conteúdo de um arquivo enviado, suportando vários formatos."""
    try:
        if uploaded_file.name.endswith('.txt'):
            # Usa with para garantir que o arquivo seja fechado corretamente
            # Use getvalue() para arquivos em memória
            return uploaded_file.getvalue().decode('utf-8')
        elif uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        elif uploaded_file.name.endswith('.docx'):
            doc_file = docx.Document(uploaded_file)
            return "\n".join([p.text for p in doc_file.paragraphs])
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            # Lê a primeira planilha por padrão
            df = pd.read_excel(uploaded_file)
            return df.to_string()
        elif uploaded_file.name.endswith('.csv'):
            # Tente ler com utf-8, se falhar, tente latin1
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin1')
            return df.to_string()
        else:
            return "Tipo de arquivo não suportado"
    except Exception as e:
        st.error(f"Erro ao ler o arquivo {uploaded_file.name}: {e}")
        return ""


def referencia_ao_anexo(pergunta):
    termos = [
        "no anexo", "no arquivo", "no documento", "nos anexos", "nos arquivos",
        "segundo o anexo", "de acordo com o anexo", "conforme o anexo", "conforme o documento anexo",
        "anexo diz", "arquivo diz", "conforme arquivo anexo", "documento diz", "anexo", "arquivo", "documento", "planilha", "csv", "txt"
    ]
    pergunta_lower = pergunta.lower()
    # Procura por palavras-chave seguidas de preposição ou no meio da frase
    return any(
        termo in pergunta_lower
        for termo in termos
    )


# Exibir mensagens anteriores do histórico
for tipo, conteudo in st.session_state["mensagens"]:
    with st.chat_message(tipo):
        st.markdown(conteudo)

# Entrada do usuário via chat_input
prompt = st.chat_input("Mande sua mensagem para a IA...")

# Exibir arquivos enviados e seus conteúdos resumidos
textos_arquivos = []
if arquivos:
    st.markdown("### Arquivos enviados:")
    for arquivo in arquivos:
        st.write(f"- **{arquivo.name}**")
        texto = ler_arquivo(arquivo)
        textos_arquivos.append(texto)
        # Mostra só o começo do texto para não sobrecarregar a interface
        # Adicionado verificação para garantir que o texto não é None ou vazio
        if texto:
            st.write(texto[:500] + ("..." if len(texto) > 500 else ""))
        else:
            st.write("Conteúdo não lido ou vazio.")

# Botão para limpar o histórico (exibido somente se houver mensagens ou prompt)
if prompt or st.session_state["mensagens"]:
    # Use um container para o botão para melhor organização
    with st.container():
        # Adicionado key única para o botão Limpar Histórico
        if st.button("Limpar Histórico", key="limpar_historico_button"):
            limpar_historico()
            st.rerun()  # Recarrega a página para limpar a interface

# Processar o prompt do usuário se houver e o LLM estiver carregado
if prompt and llm:

    if "selectbox_carregar_conversa" in st.session_state and \
       st.session_state.selectbox_carregar_conversa == "-- Nova Conversa --" and \
       "modelo_carregado_info" in st.session_state:
        # Limpa a info se iniciando nova conversa
        del st.session_state["modelo_carregado_info"]

    # Adiciona mensagem do usuário ao histórico da sessão
    st.session_state["mensagens"].append(("human", prompt))

    # Exibe a mensagem do usuário imediatamente na interface
    with st.chat_message("human"):
        st.markdown(prompt)

    # --- PREPARAÇÃO DAS MENSAGENS PARA O LLM USANDO OBJETOS LANGCHAIN ---
    mensagens_para_llm = []
    for tipo, conteudo in st.session_state["mensagens"]:
        if tipo == "human":
            mensagens_para_llm.append(HumanMessage(content=conteudo))
        elif tipo == "ai":
            mensagens_para_llm.append(AIMessage(content=conteudo))

    # Obter limite de tokens para o modelo
    limite_tokens = obter_limite_tokens(modelo_selecionado)

    # Verifica se a lista não está vazia e contém algum texto
    # Verifica se a lista não está vazia e contém algum texto
    if textos_arquivos and any(textos_arquivos):
        contexto_arquivos = "\n\n".join(filter(None, textos_arquivos))
        contexto_limitado = limitar_texto_por_tokens(
            contexto_arquivos, limite_tokens)

        # Só insere o contexto dos anexos se a pergunta mencionar o anexo
        if referencia_ao_anexo(prompt):
            system_prompt = (
                "Você é um assistente de IA. Utilize EXCLUSIVAMENTE as informações (extraídas de arquivos enviados pelo usuário) para responder à pergunta. "
                "Se não encontrar a resposta, diga 'Não encontrei essa informação nos arquivos'.\n\n"
                "Foque EXCLUSIVAMENTE nas informações dos arquivos enviados. "
                "Não use informações externas ou conhecimento prévio.\n\n"
                f"INFORMAÇÕES DOS ARQUIVOS:\n{contexto_limitado}\n\n"
            )
            mensagens_para_llm.insert(0, SystemMessage(content=system_prompt))
        else:
            # Prompt padrão, sem contexto dos arquivos
            system_prompt = (
                "Você é um assistente de IA. Responda normalmente ao usuário."
            )
            mensagens_para_llm.insert(0, SystemMessage(content=system_prompt))
    else:
        # Prompt padrão, sem contexto dos arquivos
        system_prompt = (
            "Responda normalmente ao usuário."
        )
        mensagens_para_llm.insert(0, SystemMessage(content=system_prompt))

    # --- FIM DA PREPARAÇÃO DAS MENSAGENS ---

    # Botão para interromper geração
    # Usando um container e key única para o botão Parar geração
    col_stop = st.empty()
    # Adicionado key única para o botão Parar geração
    if col_stop.button("Parar geração", key="stop_button"):
        st.session_state["parar_geracao"] = True

    # Antes do loop de streaming, resetar a flag parar_geracao para a nova interação
    st.session_state["parar_geracao"] = False

    # Criar espaço para a resposta da IA
    with st.chat_message("ai"):
        resposta_placeholder = st.empty()
        # Reinicia a resposta completa para a nova interação
        st.session_state["resposta_completa"] = ""

    # Cria barra de progresso vazia (agora dentro do bloco if prompt and llm)
    progress_bar = st.progress(0)

    # Bloco principal para gerar a resposta da IA com spinner e barra de progresso
    with st.spinner("IA gerando resposta..."):
        progress = 0
        direction = 1  # Para animação da barra de progresso

        try:
            # O método .stream() do Langchain deve funcionar para ambos Ollama e OpenAI
            # Passamos a lista de objetos de mensagem do Langchain
            # <-- AGORA USANDO mensagens_para_llm
            for chunk in llm.stream(mensagens_para_llm):
                # Verifica se a flag de parada foi ativada
                if st.session_state["parar_geracao"]:
                    st.session_state["resposta_completa"] += "\n\n**Geração interrompida pelo usuário.**"
                    resposta_placeholder.markdown(
                        st.session_state["resposta_completa"])
                    break  # Sai do loop de streaming

                # Verifica se o chunk tem conteúdo e se é do tipo esperado (AIMessageChunk ou similar)
                if hasattr(chunk, 'content') and chunk.content is not None:
                    st.session_state["resposta_completa"] += chunk.content
                    # Atualiza o placeholder com a resposta parcial e um cursor
                    resposta_placeholder.markdown(
                        st.session_state["resposta_completa"] + "▌")

                # Atualiza barra de progresso animada (simulada)
                progress += direction * 0.05  # Ajuste a velocidade da animação
                if progress >= 1:
                    direction = -1
                elif progress <= 0:
                    direction = 1
                progress_bar.progress(progress)
                time.sleep(0.02)  # Pausa curta para a animação ser visível

        except Exception as e:
            st.error(f"Erro ao gerar resposta do modelo: {e}")
            st.session_state["resposta_completa"] += "\n\nDesculpe, ocorreu um erro ao gerar a resposta."
            resposta_placeholder.markdown(
                st.session_state["resposta_completa"])

        finally:
            # Garante que a barra de progresso e o spinner sejam removidos ao final
            progress_bar.empty()
            # Exibe a resposta final completa (sem o cursor)
            resposta_placeholder.markdown(
                st.session_state["resposta_completa"])
            # Reseta a flag de parada após a geração (completa ou interrompida)
            st.session_state["parar_geracao"] = False

    # Adicionar a resposta completa da IA ao histórico da sessão
    # Isso é feito APENAS após a resposta completa ser recebida/streamada ou interrompida
    # Garante que só adiciona se houver algum conteúdo gerado
    if st.session_state["resposta_completa"] and not st.session_state["resposta_completa"].endswith("**Geração interrompida pelo usuário.**"):
        st.session_state["mensagens"].append(
            ("ai", st.session_state["resposta_completa"]))
    elif st.session_state["resposta_completa"].endswith("**Geração interrompida pelo usuário.**"):
        # Adiciona a resposta parcial com a mensagem de interrupção ao histórico
        st.session_state["mensagens"].append(
            ("ai", st.session_state["resposta_completa"]))


# Exibir botões de salvar SOMENTE se houver mensagens no histórico
if st.session_state["mensagens"]:
    col1, col2, col3 = st.columns(3)

    # Verifica se há pelo menos uma mensagem da IA para salvar
    if any(tipo == "ai" for tipo, _ in st.session_state["mensagens"]):
        with col1:
            # Botão para salvar toda a conversa (apenas as respostas da IA)
            # Adicionado key única para o botão Salvar toda a conversa
            if st.button("Salvar toda a conversa", key="salvar_toda_conversa"):
                respostas_ia = [
                    conteudo for tipo, conteudo in st.session_state["mensagens"] if tipo == "ai"]
                if respostas_ia:
                    try:
                        # Salva em modo 'append' ('a') para adicionar a conversas anteriores
                        with open("conversa_completa.txt", "a", encoding="utf-8") as f:
                            for i, resposta in enumerate(respostas_ia):
                                # Opcional: numera as respostas
                                f.write(f"Resposta IA {i+1}:\n")
                                f.write(resposta.strip() + "\n\n")
                        st.success(
                            "Toda a conversa foi salva com sucesso em conversa_completa.txt!")
                    except Exception as e:
                        st.error(f"Erro ao salvar a conversa completa: {e}")

        with col2:
            # Botão para salvar a última resposta da IA
            # Adicionado key única para o botão Salvar última resposta
            if st.button("Salvar última resposta", key="salvar_ultima_resposta"):
                ultimas_respostas = [
                    msg[1] for msg in st.session_state["mensagens"] if msg[0] == "ai"]
                if ultimas_respostas:
                    try:
                        # Salva em modo 'write' ('w') para sobrescrever o arquivo anterior
                        with open("ultima_resposta.txt", "w", encoding="utf-8") as f:
                            f.write(ultimas_respostas[-1].strip())
                        st.success(
                            "Última resposta salva com sucesso em ultima_resposta.txt!")
                    except Exception as e:
                        st.error(f"Erro ao salvar a última resposta: {e}")

    # Botão para salvar a conversa atual em JSON
    with col3:
        if st.button("Salvar Conversa Atual (JSON)", key="salvar_json_conversa_atual"):
            modelo_atual = st.session_state.get(
                "modelo_selecionado_key", "desconhecido")
            # Passa o nome do modelo para ser salvo nos metadados
            gc.salvar_conversa_json(
                st.session_state["mensagens"], modelo_usado=modelo_atual)
