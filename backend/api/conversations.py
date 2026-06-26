import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from auth import require_auth
from persistence import repository as repo

router = APIRouter()


class SaveRequest(BaseModel):
    messages: list
    model_used: str
    title: str = "Nova Conversa"


class UpdateRequest(BaseModel):
    messages: list
    model_used: str


@router.get("/conversations")
def list_conversations(_user=Depends(require_auth)):
    return repo.list_conversations()


@router.post("/conversations")
def save_conversation(req: SaveRequest, _user=Depends(require_auth)):
    conv_id = repo.save_conversation(req.messages, req.model_used, req.title)
    if not conv_id:
        raise HTTPException(503, "Database not available")
    return {"id": conv_id}


@router.get("/conversations/{conv_id}")
def load_conversation(conv_id: str, _user=Depends(require_auth)):
    data = repo.load_conversation(conv_id)
    if not data:
        raise HTTPException(404, "Conversation not found")
    return data


@router.put("/conversations/{conv_id}")
def update_conversation(conv_id: str, req: UpdateRequest, _user=Depends(require_auth)):
    ok = repo.update_conversation(conv_id, req.messages, req.model_used)
    return {"ok": ok}


@router.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str, _user=Depends(require_auth)):
    ok = repo.delete_conversation(conv_id)
    return {"ok": ok}


@router.get("/conversations/{conv_id}/export")
def export_conversation(conv_id: str, fmt: str = "markdown", _user=Depends(require_auth)):
    data = repo.load_conversation(conv_id)
    if not data:
        raise HTTPException(404, "Conversation not found")

    messages = data.get("messages", [])

    if fmt == "json":
        return {"data": json.dumps({"title": data["title"], "model": data["model_used"], "messages": messages}, ensure_ascii=False, indent=2), "mime": "application/json", "ext": "json"}

    # Markdown
    lines = [f"# {data['title']}", f"*Modelo: {data['model_used']}*\n"]
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            lines.append(f"**Você:** {content}\n")
        elif role == "assistant":
            lines.append(f"**IA:** {content}\n")
    return {"data": "\n".join(lines), "mime": "text/markdown", "ext": "md"}
