# chat_IA: Interface de Chat com Múltiplos Modelos de IA

Este projeto oferece uma interface de chat flexível que permite aos usuários interagir com diversos modelos de IA, tanto locais (via Ollama) quanto remotos (OpenAI e Gemini). A aplicação, construída com Streamlit, facilita a experimentação e comparação entre diferentes modelos, além de oferecer funcionalidades como upload de arquivos para contexto RAG (Retrieval-Augmented Generation) e salvamento/carregamento de conversas.

## Funcionalidades Principais

*   **Suporte a Múltiplos Modelos:** Integração com modelos locais (Ollama), OpenAI e Gemini, permitindo a escolha do modelo diretamente na interface.
*   **Interface Intuitiva:** Interface de chat amigável construída com Streamlit para fácil interação.
*   **RAG (Retrieval-Augmented Generation):** Capacidade de anexar arquivos (PDF, DOCX, Excel, TXT, CSV) para fornecer contexto adicional ao modelo de IA.
*   **Gerenciamento de Conversas:**
    *   Salvamento e carregamento de conversas em formato JSON para facilitar a retomada de sessões anteriores.
    *   Opção para salvar o histórico completo ou apenas a última resposta da IA em arquivos de texto.
*   **Controle de Temperatura:** Ajuste da temperatura do modelo para controlar a criatividade das respostas.
*   **Animação e Feedback:** Barra de progresso animada durante a geração de respostas e botão para interromper a geração.
*   **Cache de Memória:** Cache em memória para respostas do LLM, otimizando o desempenho.
*   **Limpeza de Histórico:** Limpa o histórico de mensagens e a resposta completa na sessão.
*   **Limite de Tokens:** Exibe o limite de tokens para contexto, com base no modelo selecionado.

## Como Usar

1.  **Pré-requisitos:**
    *   Python 3.7+
    *   [Ollama](https://ollama.com/) instalado para modelos locais
    *   Chaves de API da OpenAI e/ou Gemini (se desejar usar esses modelos)
2.  **Instalação:**

    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd chat_IA
    pip install -r requirements.txt
    ```
3.  **Configuração:**

    *   Crie um arquivo `.env` na raiz do projeto.
    *   Adicione suas chaves de API (se necessário):

        ```
        OPENAI_API_KEY=SUA_CHAVE_OPENAI
        GEMINI_API_KEY=SUA_CHAVE_GEMINI
        ```
4.  **Execução:**

    ```bash
    streamlit run app.py
    ```

    A aplicação será aberta automaticamente no seu navegador.
5.  **Utilização:**
    *   Selecione um modelo de IA na barra lateral.
    *   Ajuste a temperatura do modelo conforme desejado.
    *   Anexe arquivos para fornecer contexto adicional (opcional).
    *   Digite sua mensagem na caixa de chat e pressione Enter.
    *   Use os botões para salvar conversas, limpar o histórico ou interromper a geração.

## Arquitetura

O projeto é estruturado da seguinte forma:

*   `app.py`: Arquivo principal do Streamlit, responsável pela interface do usuário e pela lógica de interação com os modelos de IA.
*   `modelos_tokens.py`: Módulo que contém informações sobre os limites de tokens dos modelos e funções para limitar o texto de acordo com esses limites.
*   `gerenciador_conversas.py`: Módulo responsável por salvar e carregar conversas em formato JSON.
    *   **Funcionalidades:**
        *   Salva conversas em arquivos JSON, incluindo metadados como o modelo utilizado e o timestamp.
        *   Carrega conversas de arquivos JSON.
        *   Lista os arquivos de conversa salvos no diretório `conversas_salvas`.
*   `requirements.txt`: Lista de dependências do projeto.
*   `.env`: Arquivo para armazenar as chaves de API (não versionado).
*   `conversas_salvas/`: Diretório onde as conversas salvas em JSON são armazenadas.

## Dependências

*   streamlit
*   langchain
*   langchain\_openai
*   langchain\_ollama
*   python-dotenv
*   PyPDF2
*   docx
*   openpyxl
*   pandas
*   langchain\_google\_genai

## Variáveis de Ambiente

*   `OPENAI_API_KEY`: Chave de API da OpenAI (opcional).
*   `GEMINI_API_KEY`: Chave de API do Google Gemini (opcional).

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

[MIT](LICENSE)

## Notas Adicionais

*   Certifique-se de ter o Ollama instalado e configurado corretamente para usar os modelos locais.
*   As chaves de API da OpenAI e Gemini são opcionais, mas necessárias para usar esses modelos.
*   O arquivo `.env` não deve ser versionado para proteger suas chaves de API.
*   O projeto utiliza cache em memória para otimizar o desempenho, mas você pode desativá-lo se necessário.
*   As conversas salvas são armazenadas no diretório `conversas_salvas/`.
