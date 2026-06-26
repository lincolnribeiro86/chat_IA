"""Chat endpoint — SSE streaming with tool-calling loop."""
from __future__ import annotations
import json
import logging
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from auth import require_auth
from config import settings
from providers.registry import build_provider, get_model_info, CATALOG
from tools.registry import bind_tools_if_supported
from files.images import build_vision_content
from tokens import estimate_cost, truncate_by_tokens, get_context_limit
from rag.chunking import chunk_text
from rag.vectorstore import build_retriever, retrieve
from persistence import repository as repo

logger = logging.getLogger(__name__)
router = APIRouter()

_catalog_ids = {m["id"] for m in CATALOG}


class FileAttachment(BaseModel):
    name: str
    type: str           # "text" | "image"
    content: str = ""   # for text files
    data_uri: str = ""  # for images


class ChatMessage(BaseModel):
    role: str   # user | assistant
    content: str


class ChatRequest(BaseModel):
    model_id: str
    messages: list[ChatMessage]
    files: list[FileAttachment] = []
    temperature: float = 0.5
    enable_web_search: bool = False
    force_web_search: bool = False
    conversation_id: str | None = None
    api_keys: dict[str, str] = {}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_claude_sub(provider, lc_messages: list) -> AsyncIterator[str]:
    """Convert claude_subscription provider to SSE tokens."""
    from providers.claude_subscription import ClaudeSubProvider
    msgs_dicts = []
    for m in lc_messages:
        if isinstance(m, SystemMessage):
            msgs_dicts.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            msgs_dicts.append({"role": "user", "content": content})
        elif isinstance(m, AIMessage):
            msgs_dicts.append({"role": "assistant", "content": m.content})

    async for token in provider.astream_text(msgs_dicts):
        yield _sse({"type": "token", "content": token})


@router.post("/chat")
async def chat(req: ChatRequest, _user=Depends(require_auth)):
    async def event_generator() -> AsyncIterator[bytes]:
        try:
            # ── Resolve API keys (DB > env) ───────────────────────────────
            resolved_keys: dict[str, str] = {}
            for k in ["openai_api_key", "anthropic_api_key", "gemini_api_key", "groq_api_key",
                      "openrouter_api_key", "deepseek_api_key", "ollama_base_url", "ollama_api_key",
                      "tavily_api_key", "firecrawl_api_key", "claude_code_oauth_token"]:
                db_val = repo.get_setting(k)
                resolved_keys[k] = req.api_keys.get(k) or db_val or getattr(settings, k, None) or ""

            # ── Build provider ─────────────────────────────────────────────
            try:
                llm = build_provider(req.model_id, resolved_keys, req.temperature)
            except ValueError as e:
                yield _sse({"type": "error", "message": str(e)}).encode()
                return

            model_info = get_model_info(req.model_id) or {}
            is_gpt5 = getattr(llm, "_IS_GPT5", False)
            is_claude_sub = getattr(llm, "_IS_CLAUDE_SUB", False)

            # ── Prepare file context ───────────────────────────────────────
            text_files = [f for f in req.files if f.type == "text" and f.content]
            image_uris = [f.data_uri for f in req.files if f.type == "image" and f.data_uri]

            file_context = ""
            retriever = None

            if text_files:
                combined = "\n\n".join(f"=== {f.name} ===\n{f.content}" for f in text_files)
                ctx_limit = get_context_limit(req.model_id)

                if len(combined) > settings.rag_threshold_chars:
                    # Use RAG for large documents
                    session_id = str(uuid.uuid4())
                    all_chunks = []
                    for f in text_files:
                        all_chunks.extend(chunk_text(f.content, source=f.name))
                    try:
                        retriever = build_retriever(all_chunks, session_id)
                        user_query = req.messages[-1].content if req.messages else ""
                        file_context = retrieve(retriever, user_query)
                    except Exception as e:
                        logger.warning(f"RAG failed, falling back to truncation: {e}")
                        file_context = truncate_by_tokens(combined, ctx_limit)
                else:
                    file_context = combined

            # ── Build LangChain messages ───────────────────────────────────
            system_parts = ["Você é um assistente de IA útil e preciso."]
            if file_context:
                system_parts.append(
                    f"\nO usuário anexou arquivos. Use o conteúdo abaixo para responder:\n"
                    f"===INÍCIO DOS ARQUIVOS===\n{file_context}\n===FIM DOS ARQUIVOS==="
                )
            if req.force_web_search:
                system_parts.append(
                    "\nO usuário quer que você faça uma busca na web. "
                    "Use a ferramenta web_search para obter informações atuais."
                )

            lc_messages: list = [SystemMessage(content="\n".join(system_parts))]
            for i, m in enumerate(req.messages):
                if m.role == "user":
                    # Last message may have vision content
                    if i == len(req.messages) - 1 and image_uris and model_info.get("supports_vision"):
                        content = build_vision_content(m.content, image_uris)
                    else:
                        content = m.content
                    lc_messages.append(HumanMessage(content=content))
                elif m.role == "assistant":
                    lc_messages.append(AIMessage(content=m.content))

            # ── Tool binding ───────────────────────────────────────────────
            tools: list = []
            if req.enable_web_search or req.force_web_search:
                if not is_claude_sub and not is_gpt5:
                    llm, tools = bind_tools_if_supported(llm, model_info)

            # ── Stream response ────────────────────────────────────────────
            full_response = ""
            usage_data: dict = {}

            if is_claude_sub:
                # Claude subscription — async generator yields SSE strings
                async for event in _stream_claude_sub(llm, lc_messages):
                    try:
                        payload = json.loads(event[6:].strip())
                        full_response += payload.get("content", "")
                    except Exception:
                        pass
                    yield event.encode()

            elif is_gpt5:
                # GPT-5 Responses API — non-streaming, simulate progress
                llm.set_model(req.model_id)
                msgs_dicts = []
                for m in lc_messages:
                    if isinstance(m, SystemMessage):
                        msgs_dicts.insert(0, {"role": "system", "content": m.content})
                    elif isinstance(m, HumanMessage):
                        content = m.content if isinstance(m.content, str) else str(m.content)
                        msgs_dicts.append({"role": "user", "content": content})
                    elif isinstance(m, AIMessage):
                        msgs_dicts.append({"role": "assistant", "content": m.content})

                result = llm.invoke_with_verbosity(msgs_dicts)
                full_response = result.get("content", "")
                usage_data = result.get("usage", {})
                # Emit as single token (non-streaming)
                yield _sse({"type": "token", "content": full_response}).encode()

            else:
                # Standard LangChain streaming with tool-calling loop
                max_tool_iterations = 5
                for _ in range(max_tool_iterations):
                    tool_calls_made = []
                    current_chunk = ""

                    async for chunk in llm.astream(lc_messages):
                        # Accumulate usage from last chunk
                        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                            u = chunk.usage_metadata
                            usage_data = {
                                "input_tokens": u.get("input_tokens", 0),
                                "output_tokens": u.get("output_tokens", 0),
                            }

                        if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                            for tc in chunk.tool_call_chunks:
                                tool_calls_made.append(tc)
                            continue

                        if hasattr(chunk, "content") and chunk.content:
                            current_chunk += chunk.content
                            full_response += chunk.content
                            yield _sse({"type": "token", "content": chunk.content}).encode()

                    if not tool_calls_made:
                        break

                    # Execute tool calls
                    ai_msg = AIMessage(content=current_chunk, tool_calls=[
                        {"id": tc.get("id", str(uuid.uuid4())),
                         "name": tc.get("name", ""),
                         "args": json.loads(tc.get("args", "{}") or "{}")}
                        for tc in tool_calls_made
                    ])
                    lc_messages.append(ai_msg)

                    tool_map = {t.name: t for t in tools}
                    for tc in ai_msg.tool_calls:
                        tool_id = tc["id"]
                        tool_name = tc["name"]
                        tool_args = tc["args"]

                        yield _sse({"type": "tool_start", "tool_id": tool_id,
                                    "tool_name": tool_name, "tool_args": tool_args}).encode()

                        tool_fn = tool_map.get(tool_name)
                        if tool_fn:
                            try:
                                result = tool_fn.invoke(tool_args)
                            except Exception as e:
                                result = f"Tool error: {e}"
                        else:
                            result = f"Unknown tool: {tool_name}"

                        yield _sse({"type": "tool_result", "tool_id": tool_id,
                                    "tool_name": tool_name, "tool_result": str(result)}).encode()

                        lc_messages.append(
                            ToolMessage(content=str(result), tool_call_id=tool_id)
                        )

            # ── Emit usage/cost ────────────────────────────────────────────
            if usage_data:
                cost = estimate_cost(
                    req.model_id,
                    usage_data.get("input_tokens", 0),
                    usage_data.get("output_tokens", 0),
                )
                usage_data["cost_usd"] = cost
                yield _sse({"type": "usage", "usage": usage_data}).encode()

            yield _sse({"type": "done"}).encode()

        except Exception as e:
            logger.exception("Chat stream error")
            yield _sse({"type": "error", "message": str(e)}).encode()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
