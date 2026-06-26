"""FastAPI entrypoint."""
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from auth import verify_password, create_access_token, COOKIE_NAME
from config import settings
from persistence import repository as repo

from api.chat import router as chat_router
from api.models import router as models_router
from api.conversations import router as conv_router
from api.files import router as files_router
from api.settings import router as settings_router

app = FastAPI(title="chat_IA API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth routes ───────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    password: str


@app.post("/api/auth/login")
async def login(req: LoginRequest, response: Response):
    # Check DB-overridden password first, then env
    stored = repo.get_setting("app_password") or settings.app_password
    if req.password != stored:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong password")
    token = create_access_token()
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=settings.jwt_expire_hours * 3600,
    )
    return {"ok": True}


@app.post("/api/auth/logout")
async def logout(response: Response):
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}


@app.get("/api/auth/me")
async def me(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    from auth import decode_token
    if token and decode_token(token):
        return {"authenticated": True}
    return {"authenticated": False}


# ── Feature routers ───────────────────────────────────────────────────────────
app.include_router(chat_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(conv_router, prefix="/api")
app.include_router(files_router, prefix="/api")
app.include_router(settings_router, prefix="/api")


@app.get("/api/health")
def health():
    from persistence.db import db_available
    return {"status": "ok", "db": db_available()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
