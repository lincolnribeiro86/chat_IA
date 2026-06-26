# chat_IA

Chat com IA — interface web moderna com suporte a múltiplos provedores de LLM, busca na web por tool-calling, visão (imagens), RAG com embeddings e persistência de conversas.

**Stack:** FastAPI (backend) + React + Vite + TypeScript + Tailwind + shadcn/ui (frontend) + PostgreSQL + Docker.

---

## Recursos

| Recurso | Detalhe |
|---|---|
| **Multi-provedor** | Ollama local/cloud, OpenAI, Gemini, Groq, Anthropic (API Key + assinatura), OpenRouter |
| **Busca na web** | Tavily + Firecrawl via function-calling automático e botão manual |
| **Visão** | Anexar prints/imagens (PNG, JPG, WEBP) — modelos vision |
| **Arquivos** | PDF, DOCX, TXT, MD, CSV, Excel, PY, HTML, etc. |
| **RAG** | Embeddings + Chroma para documentos grandes (fallback: truncamento) |
| **Streaming** | SSE token a token com indicador de tool-calling |
| **Conversas** | Salvar, carregar, exportar (Markdown/JSON), excluir — PostgreSQL |
| **Auth** | Single-user com senha, JWT httpOnly cookie |
| **Configuração** | Chaves de API editáveis na UI (sem editar .env) |
| **Custo** | Badge de tokens e custo estimado por resposta |
| **Tema** | Claro/escuro |
| **Docker** | Compose para local (Windows) e VPS |

---

## Início rápido

### Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac/Linux)
- Pelo menos uma API key de provedor **ou** [Ollama](https://ollama.com/) rodando localmente

### 1. Clonar e configurar

```bash
git clone https://github.com/lincolnribeiro86/chat_IA
cd chat_IA
cp .env.example .env
```

Edite `.env` e preencha no mínimo `APP_PASSWORD` e `JWT_SECRET`. As chaves de API são opcionais — você pode adicioná-las depois pela UI.

### 2. Subir com Docker

```bash
docker compose up --build
```

Acesse: **http://localhost** → faça login com a senha do `.env`.

> **Windows + Ollama local:** use `OLLAMA_BASE_URL=http://host.docker.internal:11434` no `.env`.

### 3. Rodar sem Docker (desenvolvimento)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (outro terminal)
cd frontend
npm install
npm run dev    # http://localhost:5173
```

---

## Provedores e autenticação

### Ollama (local)
Instale o [Ollama](https://ollama.com/), baixe modelos (`ollama pull llama3.2`) e configure:
```
OLLAMA_BASE_URL=http://localhost:11434
```

### Ollama Cloud
```
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=<sua-chave>
```

### OpenAI
```
OPENAI_API_KEY=sk-...
```

### Anthropic (API Key)
```
ANTHROPIC_API_KEY=sk-ant-...
```

### Claude via Assinatura (experimental)
Requer Claude Code CLI instalado e autenticado:
```bash
npm install -g @anthropic-ai/claude-code
claude setup-token   # copie o token gerado
```
```
CLAUDE_CODE_OAUTH_TOKEN=<token>
```
No seletor de modelos, escolha **"Claude (Assinatura)"**.

### Google Gemini
```
GEMINI_API_KEY=...
```

### Groq
```
GROQ_API_KEY=gsk_...
```

### OpenRouter
```
OPENROUTER_API_KEY=sk-or-...
```
Os modelos do OpenRouter são listados dinamicamente na UI.

### Busca na web
```
TAVILY_API_KEY=tvly-...    # recomendado
FIRECRAWL_API_KEY=fc-...   # complementar (scraping de páginas)
```

---

## Estrutura do projeto

```
chat_IA/
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── main.py              # FastAPI entrypoint
│   ├── config.py            # Configurações (pydantic-settings)
│   ├── auth.py              # JWT + cookie
│   ├── tokens.py            # Limites de contexto e pricing
│   ├── api/                 # Endpoints: chat (SSE), models, conversations, files, settings
│   ├── providers/           # Um módulo por provedor + registry
│   ├── tools/               # web_search (Tavily + Firecrawl)
│   ├── files/               # readers.py (text/pdf/docx/md/...) + images.py
│   ├── rag/                 # chunking + Chroma vectorstore
│   └── persistence/         # PostgreSQL (db.py + repository.py)
└── frontend/
    ├── src/
    │   ├── App.tsx           # Root — auth gate + layout
    │   ├── pages/Login.tsx
    │   ├── components/       # ChatWindow, Sidebar, MessageList, MessageInput, ...
    │   ├── hooks/            # useChat (SSE), useConversations
    │   ├── lib/api.ts        # Cliente HTTP + SSE
    │   └── types/            # TypeScript types
    └── nginx.conf            # Proxy /api → backend + SPA fallback
```

---

## Configuração via UI

Na barra superior, clique em ⚙️ (Configurações) para:
- Inserir/trocar chaves de API sem editar o `.env`
- Alterar a senha de acesso
- Verificar quais provedores estão configurados (✓/○)

---

## Deploy em VPS

```bash
# Na VPS
git clone https://github.com/lincolnribeiro86/chat_IA
cd chat_IA
cp .env.example .env
nano .env   # preencher credenciais
docker compose up -d --build
```

Para HTTPS, coloque um reverse proxy (Caddy/nginx) na frente apontando para a porta 80.

---

## Variáveis de ambiente

Veja [`.env.example`](.env.example) para a lista completa comentada.

---

## Licença

MIT
