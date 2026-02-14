"""
DSP UI Generator - Configuration-based UI server.
Serves different UI experiences (login + chatbot flows) from YAML configs.
"""
import os
import yaml
import httpx
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from jose import jwt, JWTError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dsp-ui-gen")

# ---------------------------------------------------------------------------
# Pydantic models for config schema
# ---------------------------------------------------------------------------

class ThemeConfig(BaseModel):
    primary_color: str = "#2563eb"
    secondary_color: str = "#1e40af"
    accent_color: str = "#3b82f6"
    background_color: str = "#f8fafc"
    text_color: str = "#1e293b"
    logo_url: str = ""
    favicon_url: str = ""

class AuthConfig(BaseModel):
    type: str = "simple"                    # simple | ldap | jwt_endpoint
    endpoint: str = ""                      # POST endpoint for auth
    username_field: str = "username"         # field name sent to auth endpoint
    password_field: str = "password"         # field name sent to auth endpoint
    token_response_field: str = "access_token"  # field in response containing JWT
    extra_fields: Dict[str, Any] = {}       # extra fields sent with auth request
    jwt_secret: str = ""                    # for local simple auth fallback
    jwt_algorithm: str = "HS256"
    session_duration_minutes: int = 480

class ChatEndpoint(BaseModel):
    name: str = "default"
    url: str = ""                           # e.g. http://fd2:8080/query
    method: str = "POST"
    streaming: bool = False
    headers: Dict[str, str] = {}            # extra headers; {{token}} is replaced
    body_template: Dict[str, Any] = {}      # template; {{message}} / {{token}} replaced
    response_field: str = "response"        # field in JSON response containing answer

class ChatConfig(BaseModel):
    system_prompt: str = "You are a helpful AI assistant."
    placeholder: str = "Type your message..."
    welcome_message: str = "Hello! How can I help you today?"
    endpoints: list[ChatEndpoint] = []
    show_endpoint_selector: bool = False    # let user pick endpoint in UI

class AppConfig(BaseModel):
    name: str = "AI Assistant"
    description: str = ""
    theme: ThemeConfig = ThemeConfig()
    auth: AuthConfig = AuthConfig()
    chat: ChatConfig = ChatConfig()

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
CONFIGS: Dict[str, AppConfig] = {}
CONFIG_DIR = Path(__file__).parent / "configs"

def load_configs():
    """Load all YAML configs from the configs/ directory."""
    CONFIGS.clear()
    if not CONFIG_DIR.exists():
        logger.warning("No configs/ directory found")
        return
    for f in CONFIG_DIR.glob("*.yaml"):
        try:
            raw = yaml.safe_load(f.read_text(encoding="utf-8"))
            cfg = AppConfig(**raw)
            slug = f.stem  # filename without extension = route slug
            CONFIGS[slug] = cfg
            logger.info(f"Loaded config: {slug} -> {cfg.name}")
        except Exception as e:
            logger.error(f"Failed to load {f.name}: {e}")

def get_config(slug: str) -> AppConfig:
    if slug not in CONFIGS:
        raise HTTPException(404, f"UI config '{slug}' not found. Available: {list(CONFIGS.keys())}")
    return CONFIGS[slug]

# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------
LOCAL_JWT_SECRET = os.getenv("UI_GEN_JWT_SECRET", "ui-gen-dev-secret-change-me")

def create_local_token(username: str, duration_min: int = 480) -> str:
    """Create a local session JWT (used for simple auth only)."""
    payload = {
        "sub": username,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=duration_min),
    }
    return jwt.encode(payload, LOCAL_JWT_SECRET, algorithm="HS256")

def decode_session_token(token: str) -> Optional[dict]:
    """Decode local session token. Returns claims or None."""
    try:
        return jwt.decode(token, LOCAL_JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        return None

def get_session(request: Request) -> Optional[dict]:
    """Extract session info from cookie."""
    token = request.cookies.get("session_token")
    if not token:
        return None
    return decode_session_token(token)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    load_configs()
    yield

app = FastAPI(title="DSP UI Generator", version="1.0.0", lifespan=lifespan)

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ---------------------------------------------------------------------------
# Routes – index
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """List available UI configs."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "configs": CONFIGS,
    })

@app.post("/reload")
async def reload_configs():
    """Reload all configs from disk."""
    load_configs()
    return {"status": "ok", "configs": list(CONFIGS.keys())}

# ---------------------------------------------------------------------------
# Routes – login
# ---------------------------------------------------------------------------
@app.get("/{slug}/login", response_class=HTMLResponse)
async def login_page(request: Request, slug: str):
    cfg = get_config(slug)
    session = get_session(request)
    if session:
        return RedirectResponse(f"/{slug}/chat", status_code=302)
    return templates.TemplateResponse("login.html", {
        "request": request, "cfg": cfg, "slug": slug, "error": None,
    })

@app.post("/{slug}/login")
async def login_submit(request: Request, slug: str,
                        username: str = Form(...), password: str = Form(...)):
    cfg = get_config(slug)
    auth = cfg.auth
    backend_token = None
    error = None

    if auth.type == "simple":
        # Simple mode: accept any non-empty credentials, issue local token
        if not username or not password:
            error = "Username and password are required."
        else:
            backend_token = None  # no backend token in simple mode

    elif auth.type in ("ldap", "jwt_endpoint"):
        # Forward credentials to the configured auth endpoint
        if not auth.endpoint:
            error = "Auth endpoint not configured."
        else:
            try:
                payload = {
                    auth.username_field: username,
                    auth.password_field: password,
                    **auth.extra_fields,
                }
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(auth.endpoint, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    backend_token = data.get(auth.token_response_field)
                    if not backend_token:
                        error = "Auth succeeded but no token returned."
                else:
                    error = f"Authentication failed (HTTP {resp.status_code})."
            except httpx.RequestError as e:
                error = f"Auth service unreachable: {e}"
    else:
        error = f"Unknown auth type: {auth.type}"

    if error:
        return templates.TemplateResponse("login.html", {
            "request": request, "cfg": cfg, "slug": slug, "error": error,
        })

    # Create local session token
    session_token = create_local_token(username, auth.session_duration_minutes)
    response = RedirectResponse(f"/{slug}/chat", status_code=302)
    response.set_cookie("session_token", session_token, httponly=True, samesite="lax",
                         max_age=auth.session_duration_minutes * 60)
    # Store backend JWT separately so chat can use it
    if backend_token:
        response.set_cookie("backend_token", backend_token, httponly=True, samesite="lax",
                             max_age=auth.session_duration_minutes * 60)
    return response

# ---------------------------------------------------------------------------
# Routes – chat
# ---------------------------------------------------------------------------
@app.get("/{slug}/chat", response_class=HTMLResponse)
async def chat_page(request: Request, slug: str):
    cfg = get_config(slug)
    session = get_session(request)
    if not session:
        return RedirectResponse(f"/{slug}/login", status_code=302)
    return templates.TemplateResponse("chat.html", {
        "request": request, "cfg": cfg, "slug": slug,
        "username": session.get("sub", "User"),
        "endpoints": [e.name for e in cfg.chat.endpoints],
        "show_endpoint_selector": cfg.chat.show_endpoint_selector,
    })

@app.post("/{slug}/chat/send")
async def chat_send(request: Request, slug: str):
    """Proxy a chat message to the configured FD2/backend endpoint."""
    cfg = get_config(slug)
    session = get_session(request)
    if not session:
        raise HTTPException(401, "Not authenticated")

    body = await request.json()
    user_message = body.get("message", "")
    endpoint_name = body.get("endpoint", "default")
    backend_token = request.cookies.get("backend_token", "")

    # Find the target endpoint config
    ep = next((e for e in cfg.chat.endpoints if e.name == endpoint_name), None)
    if not ep and cfg.chat.endpoints:
        ep = cfg.chat.endpoints[0]
    if not ep:
        return JSONResponse({"response": "No chat endpoint configured."})

    # Build request body from template
    def substitute(obj, msg: str, token: str):
        """Recursively substitute {{message}} and {{token}} in body template."""
        if isinstance(obj, str):
            return obj.replace("{{message}}", msg).replace("{{token}}", token)
        if isinstance(obj, dict):
            return {k: substitute(v, msg, token) for k, v in obj.items()}
        if isinstance(obj, list):
            return [substitute(i, msg, token) for i in obj]
        return obj

    req_body = substitute(ep.body_template, user_message, backend_token) if ep.body_template else {"query": user_message}
    req_headers = {k: v.replace("{{token}}", backend_token) for k, v in ep.headers.items()}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.request(ep.method, ep.url, json=req_body, headers=req_headers)
        if resp.status_code == 200:
            data = resp.json()
            # Extract answer from response using dot-notation field path
            answer = data
            for part in ep.response_field.split("."):
                if isinstance(answer, dict):
                    answer = answer.get(part, "")
                else:
                    break
            return JSONResponse({"response": str(answer)})
        else:
            return JSONResponse({"response": f"Backend error (HTTP {resp.status_code}): {resp.text[:300]}"})
    except httpx.RequestError as e:
        return JSONResponse({"response": f"Could not reach backend: {e}"})

# ---------------------------------------------------------------------------
# Routes – logout
# ---------------------------------------------------------------------------
@app.get("/{slug}/logout")
async def logout(slug: str):
    response = RedirectResponse(f"/{slug}/login", status_code=302)
    response.delete_cookie("session_token")
    response.delete_cookie("backend_token")
    return response

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8200, reload=True)
