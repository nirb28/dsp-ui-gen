"""
DSP UI Generator - Configuration-based UI server.
Serves different UI experiences (login + chatbot flows) from YAML configs.
"""
import os
import json
import re
import secrets
import yaml
import httpx
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode, parse_qs, urlparse

from dotenv import load_dotenv
load_dotenv()

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

class ResponseDisplayItem(BaseModel):
    field: str                              # dot-notation path into response JSON
    label: str = ""                         # display label
    type: str = "text"                      # text | code | table | json | markdown

class ChatEndpoint(BaseModel):
    name: str = "default"
    url: str = ""                           # e.g. http://fd2:8080/query
    method: str = "POST"
    streaming: bool = False
    headers: Dict[str, str] = {}            # extra headers; {{token}} is replaced
    body_template: Dict[str, Any] = {}      # template; {{message}} / {{token}} replaced
    response_field: str = "response"        # field in JSON response containing answer
    response_display: List[ResponseDisplayItem] = []  # optional rich display items

class InputConfig(BaseModel):
    type: str = "text"                      # text | code | multiline
    rows: int = 1                           # default rows for textarea
    max_rows: int = 6                       # max auto-expand rows
    language: str = ""                      # hint for code mode (python, sas, sql, etc.)
    label: str = ""                         # optional label above input
    submit_on_enter: bool = True            # Enter submits (false = Shift+Enter submits)

class SampleRequest(BaseModel):
    label: str                              # display label in the selector
    message: str                            # pre-filled message text
    endpoint: str = ""                     # optional: pre-select this endpoint

class ChatConfig(BaseModel):
    system_prompt: str = "You are a helpful AI assistant."
    placeholder: str = "Type your message..."
    welcome_message: str = "Hello! How can I help you today?"
    input: InputConfig = InputConfig()      # input control configuration
    endpoints: list[ChatEndpoint] = []
    show_endpoint_selector: bool = False    # let user pick endpoint in UI
    samples: List[SampleRequest] = []       # optional sample requests shown in UI

class ParamConfig(BaseModel):
    name: str                               # e.g. "project"
    label: str = ""                         # display label in UI
    default: Optional[str] = None           # default value; None means required (no default)

class AppConfig(BaseModel):
    name: str = "AI Assistant"
    description: str = ""
    params: List[ParamConfig] = []          # URL/request parameters
    theme: ThemeConfig = ThemeConfig()
    auth: AuthConfig = AuthConfig()
    chat: ChatConfig = ChatConfig()

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
CONFIGS: Dict[str, AppConfig] = {}
CONFIG_DIR = Path(__file__).parent / "configs"

def list_ct_manifests() -> List[Dict[str, Any]]:
    """List CT manifest files from configured directory."""
    items: List[Dict[str, Any]] = []
    if not CT_MANIFESTS_DIR:
        return items
    manifest_dir = Path(CT_MANIFESTS_DIR)
    if not manifest_dir.exists() or not manifest_dir.is_dir():
        logger.warning(f"CT_MANIFESTS_DIR does not exist or is not a directory: {manifest_dir}")
        return items

    for f in sorted(manifest_dir.glob("*.json")):
        item: Dict[str, Any] = {
            "name": f.stem,
            "file_name": f.name,
            "title": f.stem,
            "modules": None,
            "environment": "",
            "valid": True,
        }
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            item["title"] = raw.get("name") or raw.get("project_name") or raw.get("project_id") or f.stem
            modules = raw.get("modules")
            if isinstance(modules, list):
                item["modules"] = len(modules)
            env = raw.get("environment")
            if isinstance(env, str):
                item["environment"] = env
        except Exception:
            item["valid"] = False
        items.append(item)
    return items

def get_ct_manifest_path(name: str) -> Path:
    """Resolve a safe CT manifest file path from manifest name."""
    if not CT_MANIFESTS_DIR:
        raise HTTPException(400, "CT_MANIFESTS_DIR is not configured in .env")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise HTTPException(400, "Invalid manifest name")

    manifest_dir = Path(CT_MANIFESTS_DIR)
    return manifest_dir / f"{name}.json"

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

# ---------------------------------------------------------------------------
# Environment-based settings
# ---------------------------------------------------------------------------
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
FD_BASE_URL = os.getenv("FD_BASE_URL", "http://127.0.0.1:8080")
CT_BASE_URL = os.getenv("CT_BASE_URL", "http://127.0.0.1:8000")
CT_MANIFESTS_DIR = os.getenv("CT_MANIFESTS_DIR", "")
TRACE_ENABLED = os.getenv("UI_GEN_TRACE", os.getenv("TRACE", "false")).strip().lower() in {"1", "true", "yes", "on"}
TRACE_HIDE_TOKEN = os.getenv("UI_GEN_TRACE_HIDE_TOKEN", "false").strip().lower() in {"1", "true", "yes", "on"}
SSL_VERIFY = os.getenv("UI_GEN_SSL_VERIFY", os.getenv("SSL_VERIFY", "true")).strip().lower() in {"1", "true", "yes", "on"}
BACKEND_TOKENS: Dict[str, Dict[str, Any]] = {}


def _curl_escape_single_quotes(value: str) -> str:
    """Escape single quotes for safe use in single-quoted curl arguments."""
    return value.replace("'", "'\"'\"'")


def build_curl_command(method: str, url: str, headers: Dict[str, str], body: Any) -> str:
    """Build a curl command that can be re-used for debugging."""
    cmd_parts = [f"curl -X {method.upper()} '{_curl_escape_single_quotes(url)}'"]
    for k, v in headers.items():
        display_v = "<token hidden>" if TRACE_HIDE_TOKEN and k.lower() == "authorization" else v
        cmd_parts.append(f"  -H '{_curl_escape_single_quotes(str(k))}: {_curl_escape_single_quotes(str(display_v))}'")
    if body is not None:
        body_json = json.dumps(body, ensure_ascii=False)
        cmd_parts.append(f"  --data-raw '{_curl_escape_single_quotes(body_json)}'")
    return " \\\n".join(cmd_parts)


def _mask_token_in_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a trace dict with Authorization header and token field masked."""
    if not TRACE_HIDE_TOKEN:
        return trace
    import copy
    t = copy.deepcopy(trace)
    hdrs = t.get("headers", {})
    for k in list(hdrs.keys()):
        if k.lower() == "authorization":
            hdrs[k] = "<token hidden>"
    if "token" in t:
        t["token"] = "<token hidden>"
    if "curl" in t:
        t["curl"] = build_curl_command(t.get("method", "POST"), t.get("url", ""), hdrs, t.get("body"))
    return t


def store_backend_token(token: str, max_age_seconds: int) -> str:
    """Store backend token server-side and return a compact reference key for cookies."""
    ref = secrets.token_urlsafe(32)
    BACKEND_TOKENS[ref] = {
        "token": token,
        "expires_at": datetime.now(timezone.utc) + timedelta(seconds=max_age_seconds),
    }
    return ref


def get_backend_token_by_ref(ref: str) -> str:
    """Resolve backend token by reference key and cleanup expired records."""
    item = BACKEND_TOKENS.get(ref)
    if not item:
        return ""
    if datetime.now(timezone.utc) >= item["expires_at"]:
        BACKEND_TOKENS.pop(ref, None)
        return ""
    return str(item.get("token", ""))


def delete_backend_token_ref(ref: str) -> None:
    """Delete backend token reference from in-memory store."""
    if ref:
        BACKEND_TOKENS.pop(ref, None)

def verify_admin(request: Request):
    """Verify admin credentials from session cookie or raise 401."""
    token = request.cookies.get("admin_session")
    if token:
        try:
            claims = jwt.decode(token, LOCAL_JWT_SECRET, algorithms=["HS256"])
            if claims.get("role") == "admin":
                return claims
        except JWTError:
            pass
    return None

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
# Routes – manifest tester
# ---------------------------------------------------------------------------
@app.get("/manifest-tester", response_class=HTMLResponse)
async def manifest_tester_page(request: Request):
    return templates.TemplateResponse("manifest_tester.html", {"request": request})

@app.get("/api/projects")
async def api_list_projects(fd_base: str = None):
    """Proxy to Front Door to list available projects."""
    fd_base = fd_base or FD_BASE_URL
    try:
        async with httpx.AsyncClient(timeout=15, verify=SSL_VERIFY) as client:
            resp = await client.get(
                f"{fd_base.rstrip('/')}/admin/projects",
                headers={"Accept": "application/json"},
            )
        return JSONResponse(resp.json(), status_code=resp.status_code)
    except httpx.RequestError as e:
        raise HTTPException(502, f"Could not reach Front Door at {fd_base}: {e}")

@app.get("/api/manifest/{name}")
async def api_get_manifest(name: str, ct_base: str = None,
                           resolve_env: bool = False):
    """Proxy to Control Tower to get a specific manifest."""
    ct_base = ct_base or CT_BASE_URL
    try:
        async with httpx.AsyncClient(timeout=15, verify=SSL_VERIFY) as client:
            resp = await client.get(
                f"{ct_base.rstrip('/')}/manifests/{name}",
                params={"resolve_env": str(resolve_env).lower()},
            )
        return JSONResponse(resp.json(), status_code=resp.status_code)
    except httpx.RequestError as e:
        raise HTTPException(502, f"Could not reach Control Tower at {ct_base}: {e}")

@app.post("/api/activate-manifest")
async def activate_manifest(request: Request):
    """Build a dynamic AppConfig from manifest module data and store it."""
    data = await request.json()
    project_name = data["project_name"]
    fd_base = data["fd_base"].rstrip("/")
    auth_cfg = data.get("auth", {})
    endpoints_data = data.get("endpoints", [])
    display_name = data.get("display_name", project_name)

    slug = f"mt-{project_name}"

    auth = AuthConfig(
        type="jwt_endpoint",
        endpoint=f"{fd_base}/{project_name}/{auth_cfg.get('path', 'auth/token').lstrip('/')}",
        username_field=auth_cfg.get("username_field", "username"),
        password_field=auth_cfg.get("password_field", "password"),
        token_response_field=auth_cfg.get("token_response_field", "access_token"),
    )

    chat_endpoints = []
    for ep in endpoints_data:
        chat_endpoints.append(ChatEndpoint(
            name=ep["name"],
            url=f"{fd_base}/{project_name}/{ep['path'].lstrip('/')}",
            method=ep.get("method", "POST"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer {{token}}"},
            body_template=ep.get("body_template", {"query": "{{message}}"}),
            response_field=ep.get("response_field", "choices.0.message.content"),
        ))

    cfg = AppConfig(
        name=display_name,
        description=f"Manifest tester: {project_name}",
        theme=ThemeConfig(
            primary_color="#f59e0b", secondary_color="#d97706",
            accent_color="#fbbf24", background_color="#fffbeb",
        ),
        auth=auth,
        chat=ChatConfig(
            placeholder="Enter your test input...",
            welcome_message=f"Connected to '{project_name}'. Select an endpoint and start testing.",
            endpoints=chat_endpoints,
            show_endpoint_selector=len(chat_endpoints) > 1,
        ),
    )

    CONFIGS[slug] = cfg
    logger.info(f"Activated manifest config: {slug} -> {display_name} ({len(chat_endpoints)} endpoints)")
    return JSONResponse({"slug": slug, "login_url": f"/{slug}/login"})

# ---------------------------------------------------------------------------
# Routes – admin (must be before /{slug}/* to avoid route conflicts)
# ---------------------------------------------------------------------------
@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Admin login form."""
    if verify_admin(request):
        return RedirectResponse("/admin", status_code=302)
    return templates.TemplateResponse("admin_login.html", {
        "request": request, "error": None,
    })

@app.post("/admin/login")
async def admin_login_submit(request: Request,
                              username: str = Form(...), password: str = Form(...)):
    """Validate admin credentials and set session cookie."""
    if secrets.compare_digest(username, ADMIN_USERNAME) and secrets.compare_digest(password, ADMIN_PASSWORD):
        token = jwt.encode({
            "sub": username, "role": "admin",
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=8),
        }, LOCAL_JWT_SECRET, algorithm="HS256")
        response = RedirectResponse("/admin", status_code=302)
        response.set_cookie("admin_session", token, httponly=True, samesite="lax", max_age=8*3600)
        return response
    return templates.TemplateResponse("admin_login.html", {
        "request": request, "error": "Invalid credentials.",
    })

@app.get("/admin/logout")
async def admin_logout():
    response = RedirectResponse("/admin/login", status_code=302)
    response.delete_cookie("admin_session")
    return response

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    """Admin interface to view, edit, and reload configs."""
    if not verify_admin(request):
        return RedirectResponse("/admin/login", status_code=302)
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "configs": CONFIGS,
        "manifests": list_ct_manifests(),
        "ct_manifest_dir": CT_MANIFESTS_DIR,
    })

@app.get("/admin/config/{slug}")
async def admin_get_config(request: Request, slug: str):
    """Return raw YAML content for a config."""
    if not verify_admin(request):
        raise HTTPException(401, "Not authenticated")
    yaml_path = CONFIG_DIR / f"{slug}.yaml"
    if yaml_path.exists():
        return JSONResponse({"slug": slug, "yaml": yaml_path.read_text(encoding="utf-8"), "source": "file"})
    # Dynamic config (e.g. manifest tester) – serialize from memory
    if slug in CONFIGS:
        cfg = CONFIGS[slug]
        return JSONResponse({"slug": slug, "yaml": yaml.dump(cfg.model_dump(), default_flow_style=False, sort_keys=False), "source": "dynamic"})
    raise HTTPException(404, f"Config '{slug}' not found")

@app.put("/admin/config/{slug}")
async def admin_save_config(request: Request, slug: str):
    """Save edited YAML content to disk and reload that config."""
    if not verify_admin(request):
        raise HTTPException(401, "Not authenticated")
    body = await request.json()
    yaml_content = body.get("yaml", "")
    if not yaml_content.strip():
        raise HTTPException(400, "Empty YAML content")
    try:
        raw = yaml.safe_load(yaml_content)
        cfg = AppConfig(**raw)
    except Exception as e:
        raise HTTPException(400, f"Invalid config: {e}")
    yaml_path = CONFIG_DIR / f"{slug}.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    CONFIGS[slug] = cfg
    logger.info(f"Admin saved config: {slug} -> {cfg.name}")
    return JSONResponse({"status": "ok", "slug": slug, "name": cfg.name})

@app.get("/admin/manifests")
async def admin_list_manifests(request: Request):
    """List CT manifests from configured directory."""
    if not verify_admin(request):
        raise HTTPException(401, "Not authenticated")
    return JSONResponse({
        "manifest_dir": CT_MANIFESTS_DIR,
        "manifests": list_ct_manifests(),
    })

@app.get("/admin/manifest/{name}")
async def admin_get_manifest(request: Request, name: str):
    """Return raw JSON content for a CT manifest."""
    if not verify_admin(request):
        raise HTTPException(401, "Not authenticated")
    manifest_path = get_ct_manifest_path(name)
    if not manifest_path.exists():
        raise HTTPException(404, f"Manifest '{name}' not found")

    return JSONResponse({
        "name": name,
        "content": manifest_path.read_text(encoding="utf-8"),
        "format": "json",
        "source": "ct_manifest",
        "path": str(manifest_path),
    })

@app.put("/admin/manifest/{name}")
async def admin_save_manifest(request: Request, name: str):
    """Save JSON content for a CT manifest file."""
    if not verify_admin(request):
        raise HTTPException(401, "Not authenticated")
    body = await request.json()
    content = body.get("content", "")
    if not content.strip():
        raise HTTPException(400, "Empty manifest content")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid JSON: {e}")

    manifest_path = get_ct_manifest_path(name)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    title = parsed.get("name") or parsed.get("project_name") or parsed.get("project_id") or name
    logger.info(f"Admin saved CT manifest: {manifest_path}")
    return JSONResponse({"status": "ok", "name": name, "title": title})

@app.post("/admin/manifest/{name}/delete")
async def admin_delete_manifest(request: Request, name: str):
    """Delete a CT manifest file."""
    if not verify_admin(request):
        raise HTTPException(401, "Not authenticated")
    manifest_path = get_ct_manifest_path(name)
    if manifest_path.exists():
        manifest_path.unlink()
    logger.info(f"Admin deleted CT manifest: {manifest_path}")
    return JSONResponse({"status": "ok"})

@app.post("/admin/config/{slug}/delete")
async def admin_delete_config(request: Request, slug: str):
    """Delete a config file and remove from memory."""
    if not verify_admin(request):
        raise HTTPException(401, "Not authenticated")
    yaml_path = CONFIG_DIR / f"{slug}.yaml"
    if yaml_path.exists():
        yaml_path.unlink()
    CONFIGS.pop(slug, None)
    logger.info(f"Admin deleted config: {slug}")
    return JSONResponse({"status": "ok"})

@app.post("/admin/reload")
async def admin_reload(request: Request):
    """Reload all configs from disk."""
    if not verify_admin(request):
        raise HTTPException(401, "Not authenticated")
    load_configs()
    return JSONResponse({
        "status": "ok",
        "configs": list(CONFIGS.keys()),
        "manifests": len(list_ct_manifests()),
    })

# ---------------------------------------------------------------------------
# Routes – login
# ---------------------------------------------------------------------------
def resolve_params(cfg: AppConfig, request: Request) -> Dict[str, str]:
    """Build param values from config defaults, cookie, then query string overrides."""
    params = {p.name: (p.default if p.default is not None else "") for p in cfg.params}
    # Cookie-stored params: only fill in params that are still empty (no default set)
    cookie_params = request.cookies.get("ui_params")
    if cookie_params:
        try:
            for k, v in json.loads(cookie_params).items():
                if k in params and not params[k] and v:
                    params[k] = v
        except Exception:
            pass
    # Query string overrides always win
    for p in cfg.params:
        qval = request.query_params.get(p.name)
        if qval:
            params[p.name] = qval
    return params


def get_missing_required_params(cfg: AppConfig, params: Dict[str, str]) -> List[str]:
    """Return names of params that have no default and no resolved value."""
    missing = []
    for p in cfg.params:
        if p.default is None and not params.get(p.name):
            missing.append(p.label or p.name)
    return missing


def load_config_from_path(config_path: str) -> AppConfig:
    """Load an AppConfig from an arbitrary YAML file path."""
    path = Path(config_path)
    if not path.exists():
        raise HTTPException(404, f"Config file not found: {config_path}")
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise HTTPException(400, "Config file must be a .yaml or .yml file")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return AppConfig(**raw)
    except Exception as e:
        raise HTTPException(400, f"Failed to load config from '{config_path}': {e}")


def get_config_for_request(slug: str, request: Request) -> AppConfig:
    """Resolve AppConfig by slug only (slug routes never use config_path)."""
    return get_config(slug)


def require_config_path(request: Request) -> AppConfig:
    """Load AppConfig from the required config_path query parameter."""
    config_path = request.query_params.get("config_path")
    if not config_path:
        raise HTTPException(400, "config_path query parameter is required for this route")
    return load_config_from_path(config_path)

# ---------------------------------------------------------------------------
# Routes – /ui/* (config_path-based, no slug)
# ---------------------------------------------------------------------------
UI_SLUG = "_ui"  # virtual slug used for /ui/* routes in templates

@app.get("/ui/login", response_class=HTMLResponse)
async def ui_login_page(request: Request):
    """Login page loaded from config_path query parameter (no slug needed)."""
    cfg = require_config_path(request)
    session = get_session(request)
    params = resolve_params(cfg, request)
    cp = request.query_params.get("config_path")
    if session:
        redirect_qs = dict(params)
        redirect_qs["config_path"] = cp
        return RedirectResponse(f"/ui/chat?{urlencode(redirect_qs)}", status_code=302)
    missing = get_missing_required_params(cfg, params)
    param_error = f"Required parameter(s) not set: {', '.join(missing)}" if missing else None
    return templates.TemplateResponse("login.html", {
        "request": request, "cfg": cfg, "slug": UI_SLUG,
        "login_action": f"/ui/login?{request.url.query}",
        "error": param_error,
        "params": params,
    })

@app.post("/ui/login")
async def ui_login_submit(request: Request,
                           username: str = Form(...), password: str = Form(...)):
    """Login form submission for config_path-based UI."""
    cfg = require_config_path(request)
    auth = cfg.auth
    backend_token = None
    error = None
    cp = request.query_params.get("config_path", "")

    if auth.type == "simple":
        if not username or not password:
            error = "Username and password are required."
        else:
            backend_token = None

    elif auth.type in ("ldap", "jwt_endpoint"):
        if not auth.endpoint:
            error = "Auth endpoint not configured."
        else:
            try:
                params = resolve_params(cfg, request)
                auth_url = auth.endpoint
                for k, v in params.items():
                    auth_url = auth_url.replace("{{param." + k + "}}", v)
                    auth_url = auth_url.replace("{{" + k + "}}", v)
                payload = {
                    auth.username_field: username,
                    auth.password_field: password,
                    **auth.extra_fields,
                }
                async with httpx.AsyncClient(timeout=15, verify=SSL_VERIFY) as client:
                    resp = await client.post(auth_url, json=payload)
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
            "request": request, "cfg": cfg, "slug": UI_SLUG,
            "login_action": f"/ui/login?{request.url.query}",
            "error": error,
            "params": resolve_params(cfg, request),
        })

    session_token = create_local_token(username, auth.session_duration_minutes)
    params = resolve_params(cfg, request)
    redirect_qs_params = dict(params)
    redirect_qs_params["config_path"] = cp
    redirect_url = f"/ui/chat?{urlencode(redirect_qs_params)}"
    response = RedirectResponse(redirect_url, status_code=302)
    response.set_cookie("session_token", session_token, httponly=True, samesite="lax",
                         max_age=auth.session_duration_minutes * 60)
    if backend_token:
        existing_ref = request.cookies.get("backend_token_ref", "")
        delete_backend_token_ref(existing_ref)
        token_max_age = auth.session_duration_minutes * 60
        token_ref = store_backend_token(backend_token, token_max_age)
        response.set_cookie("backend_token_ref", token_ref, httponly=True, samesite="lax",
                             max_age=token_max_age)
        response.delete_cookie("backend_token")
    else:
        delete_backend_token_ref(request.cookies.get("backend_token_ref", ""))
        response.delete_cookie("backend_token_ref")
        response.delete_cookie("backend_token")
    if params:
        response.set_cookie("ui_params", json.dumps(params), httponly=True, samesite="lax",
                             max_age=auth.session_duration_minutes * 60)
    return response

@app.get("/ui/chat", response_class=HTMLResponse)
async def ui_chat_page(request: Request):
    """Chat page loaded from config_path query parameter (no slug needed)."""
    cfg = require_config_path(request)
    session = get_session(request)
    cp = request.query_params.get("config_path", "")
    if not session:
        params = resolve_params(cfg, request)
        redirect_qs = dict(params)
        redirect_qs["config_path"] = cp
        return RedirectResponse(f"/ui/login?{urlencode(redirect_qs)}", status_code=302)
    params = resolve_params(cfg, request)
    missing = get_missing_required_params(cfg, params)
    ep_display = {}
    for e in cfg.chat.endpoints:
        if e.response_display:
            ep_display[e.name] = [item.model_dump() for item in e.response_display]
    samples_json = json.dumps([s.model_dump() for s in cfg.chat.samples])
    missing_params_error = f"Required parameter(s) not set: {', '.join(missing)}" if missing else None
    return templates.TemplateResponse("chat.html", {
        "request": request, "cfg": cfg, "slug": UI_SLUG,
        "logout_url": f"/ui/logout?{request.url.query}",
        "chat_send_url": f"/ui/chat/send?{request.url.query}",
        "chat_url": f"/ui/chat?config_path={request.query_params.get('config_path', '')}",
        "username": session.get("sub", "User"),
        "endpoints": [e.name for e in cfg.chat.endpoints],
        "show_endpoint_selector": cfg.chat.show_endpoint_selector,
        "params": params,
        "param_configs": cfg.params,
        "ep_display_json": json.dumps(ep_display),
        "samples_json": samples_json,
        "missing_params_error": missing_params_error,
    })

@app.post("/ui/chat/send")
async def ui_chat_send(request: Request):
    """Proxy chat message for config_path-based UI."""
    cfg = require_config_path(request)
    session = get_session(request)
    if not session:
        raise HTTPException(401, "Not authenticated")

    body = await request.json()
    user_message = body.get("message", "")
    endpoint_name = body.get("endpoint", "default")

    params_check = resolve_params(cfg, request)
    missing = get_missing_required_params(cfg, params_check)
    if missing:
        return JSONResponse(
            {"response": f"Cannot send message: required parameter(s) not set: {', '.join(missing)}. "
                         "Open the \u2699 Params panel and provide the missing values."},
            status_code=200,
        )

    backend_token_ref = request.cookies.get("backend_token_ref", "")
    backend_token = get_backend_token_by_ref(backend_token_ref) if backend_token_ref else ""
    if not backend_token:
        backend_token = request.cookies.get("backend_token", "")

    ep = next((e for e in cfg.chat.endpoints if e.name == endpoint_name), None)
    if not ep and cfg.chat.endpoints:
        ep = cfg.chat.endpoints[0]
    if not ep:
        return JSONResponse({"response": "No chat endpoint configured."})

    params = resolve_params(cfg, request)

    def substitute(obj, msg: str, token: str, params: Dict[str, str]):
        if isinstance(obj, str):
            s = obj.replace("{{message}}", msg).replace("{{token}}", token)
            for k, v in params.items():
                s = s.replace("{{param." + k + "}}", v)
                s = s.replace("{{" + k + "}}", v)
            return s
        if isinstance(obj, dict):
            return {k: substitute(v, msg, token, params) for k, v in obj.items()}
        if isinstance(obj, list):
            return [substitute(i, msg, token, params) for i in obj]
        return obj

    req_body = substitute(ep.body_template, user_message, backend_token, params) if ep.body_template else {"query": user_message}
    req_headers = {k: substitute(v, user_message, backend_token, params) for k, v in ep.headers.items()}
    req_url = substitute(ep.url, user_message, backend_token, params)

    try:
        async with httpx.AsyncClient(timeout=60, verify=SSL_VERIFY) as client:
            resp = await client.request(ep.method, req_url, json=req_body, headers=req_headers)
        if resp.status_code == 200:
            data = resp.json()
            def extract_field(data, field_path: str):
                val = data
                for part in field_path.split("."):
                    if isinstance(val, dict):
                        val = val.get(part, "")
                    elif isinstance(val, list) and part.isdigit():
                        idx = int(part)
                        val = val[idx] if idx < len(val) else ""
                    else:
                        break
                return val
            if ep.response_display:
                display_items = []
                for item in ep.response_display:
                    val = extract_field(data, item.field)
                    display_items.append({"label": item.label or item.field, "type": item.type, "value": val})
                return JSONResponse({"response": str(extract_field(data, ep.response_field)), "display": display_items})
            return JSONResponse({"response": str(extract_field(data, ep.response_field))})
        else:
            return JSONResponse({"response": f"Backend error (HTTP {resp.status_code}): {resp.text[:300]}"})
    except httpx.RequestError as e:
        return JSONResponse({"response": f"Could not reach backend: {e}"})

@app.get("/ui/logout")
async def ui_logout(request: Request):
    cp = request.query_params.get("config_path", "")
    response = RedirectResponse(f"/ui/login?config_path={cp}" if cp else "/", status_code=302)
    delete_backend_token_ref(request.cookies.get("backend_token_ref", ""))
    response.delete_cookie("session_token")
    response.delete_cookie("backend_token_ref")
    response.delete_cookie("backend_token")
    return response

# ---------------------------------------------------------------------------
# Routes – /{slug}/login (slug-based, no config_path)
# ---------------------------------------------------------------------------
@app.get("/{slug}/login", response_class=HTMLResponse)
async def login_page(request: Request, slug: str):
    cfg = get_config(slug)
    session = get_session(request)
    params = resolve_params(cfg, request)
    if session:
        qs = urlencode(params) if params else ""
        return RedirectResponse(f"/{slug}/chat?{qs}" if qs else f"/{slug}/chat", status_code=302)
    missing = get_missing_required_params(cfg, params)
    param_error = f"Required parameter(s) not set: {', '.join(missing)}" if missing else None
    return templates.TemplateResponse("login.html", {
        "request": request, "cfg": cfg, "slug": slug,
        "login_action": f"/{slug}/login?{request.url.query}",
        "error": param_error,
        "params": params,
    })

@app.post("/{slug}/login")
async def login_submit(request: Request, slug: str,
                        username: str = Form(...), password: str = Form(...)):
    cfg = get_config_for_request(slug, request)
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
                # Substitute params in auth endpoint URL
                params = resolve_params(cfg, request)
                auth_url = auth.endpoint
                for k, v in params.items():
                    auth_url = auth_url.replace("{{param." + k + "}}", v)
                    auth_url = auth_url.replace("{{" + k + "}}", v)
                payload = {
                    auth.username_field: username,
                    auth.password_field: password,
                    **auth.extra_fields,
                }
                trace_auth_request = {
                    "method": "POST",
                    "url": auth_url,
                    "headers": {"Content-Type": "application/json"},
                    "body": payload,
                }
                trace_auth_request["curl"] = build_curl_command(
                    trace_auth_request["method"],
                    trace_auth_request["url"],
                    trace_auth_request["headers"],
                    trace_auth_request["body"],
                )
                if TRACE_ENABLED:
                    logger.info(
                        "trace.auth.request.pre_submit=%s",
                        json.dumps(trace_auth_request, ensure_ascii=False),
                    )
                async with httpx.AsyncClient(timeout=15, verify=SSL_VERIFY) as client:
                    resp = await client.post(auth_url, json=payload)
                if TRACE_ENABLED:
                    logger.info(
                        "trace.auth.request=%s trace.auth.response=%s",
                        json.dumps(trace_auth_request, ensure_ascii=False),
                        json.dumps(
                            {
                                "status_code": resp.status_code,
                                "headers": dict(resp.headers),
                                "body": resp.text,
                            },
                            ensure_ascii=False,
                        ),
                    )
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
            "request": request, "cfg": cfg, "slug": slug,
            "login_action": f"/{slug}/login?{request.url.query}",
            "error": error,
            "params": resolve_params(cfg, request),
        })

    # Create local session token
    session_token = create_local_token(username, auth.session_duration_minutes)
    params = resolve_params(cfg, request)
    qs = urlencode(params) if params else ""
    redirect_url = f"/{slug}/chat?{qs}" if qs else f"/{slug}/chat"
    response = RedirectResponse(redirect_url, status_code=302)
    response.set_cookie("session_token", session_token, httponly=True, samesite="lax",
                         max_age=auth.session_duration_minutes * 60)
    # Store backend JWT server-side and keep only a compact ref in cookie
    if backend_token:
        existing_ref = request.cookies.get("backend_token_ref", "")
        delete_backend_token_ref(existing_ref)
        token_max_age = auth.session_duration_minutes * 60
        token_ref = store_backend_token(backend_token, token_max_age)
        response.set_cookie("backend_token_ref", token_ref, httponly=True, samesite="lax",
                             max_age=token_max_age)
        response.delete_cookie("backend_token")
    else:
        delete_backend_token_ref(request.cookies.get("backend_token_ref", ""))
        response.delete_cookie("backend_token_ref")
        response.delete_cookie("backend_token")
    # Persist params in cookie so they survive across requests
    if params:
        response.set_cookie("ui_params", json.dumps(params), httponly=True, samesite="lax",
                             max_age=auth.session_duration_minutes * 60)
    return response

# ---------------------------------------------------------------------------
# Routes – chat
# ---------------------------------------------------------------------------
@app.get("/{slug}/chat", response_class=HTMLResponse)
async def chat_page(request: Request, slug: str):
    cfg = get_config_for_request(slug, request)
    session = get_session(request)
    if not session:
        params = resolve_params(cfg, request)
        redirect_qs = dict(params)
        cp = request.query_params.get("config_path")
        if cp:
            redirect_qs["config_path"] = cp
        qs = urlencode(redirect_qs) if redirect_qs else ""
        return RedirectResponse(f"/{slug}/login?{qs}" if qs else f"/{slug}/login", status_code=302)
    params = resolve_params(cfg, request)
    missing = get_missing_required_params(cfg, params)
    # Build endpoint display config for JS
    ep_display = {}
    for e in cfg.chat.endpoints:
        if e.response_display:
            ep_display[e.name] = [item.model_dump() for item in e.response_display]
    samples_json = json.dumps([s.model_dump() for s in cfg.chat.samples])
    missing_params_error = f"Required parameter(s) not set: {', '.join(missing)}" if missing else None
    return templates.TemplateResponse("chat.html", {
        "request": request, "cfg": cfg, "slug": slug,
        "logout_url": f"/{slug}/logout",
        "chat_send_url": f"/{slug}/chat/send",
        "chat_url": f"/{slug}/chat",
        "username": session.get("sub", "User"),
        "endpoints": [e.name for e in cfg.chat.endpoints],
        "show_endpoint_selector": cfg.chat.show_endpoint_selector,
        "params": params,
        "param_configs": cfg.params,
        "ep_display_json": json.dumps(ep_display),
        "samples_json": samples_json,
        "missing_params_error": missing_params_error,
    })

@app.post("/{slug}/chat/send")
async def chat_send(request: Request, slug: str):
    """Proxy a chat message to the configured FD2/backend endpoint."""
    cfg = get_config_for_request(slug, request)
    session = get_session(request)
    if not session:
        raise HTTPException(401, "Not authenticated")

    body = await request.json()
    user_message = body.get("message", "")
    endpoint_name = body.get("endpoint", "default")

    # Validate required params before proxying
    params_check = resolve_params(cfg, request)
    missing = get_missing_required_params(cfg, params_check)
    if missing:
        return JSONResponse(
            {"response": f"Cannot send message: required parameter(s) not set: {', '.join(missing)}. "
                         "Open the ⚙ Params panel and provide the missing values."},
            status_code=200,
        )

    backend_token_ref = request.cookies.get("backend_token_ref", "")
    backend_token = get_backend_token_by_ref(backend_token_ref) if backend_token_ref else ""
    if not backend_token:
        backend_token = request.cookies.get("backend_token", "")

    # Find the target endpoint config
    ep = next((e for e in cfg.chat.endpoints if e.name == endpoint_name), None)
    if not ep and cfg.chat.endpoints:
        ep = cfg.chat.endpoints[0]
    if not ep:
        return JSONResponse({"response": "No chat endpoint configured."})

    # Collect params from cookie/request
    params = resolve_params(cfg, request)

    # Build request body from template
    def substitute(obj, msg: str, token: str, params: Dict[str, str]):
        """Recursively substitute {{message}}, {{token}}, and {{param.*}} placeholders."""
        if isinstance(obj, str):
            s = obj.replace("{{message}}", msg).replace("{{token}}", token)
            for k, v in params.items():
                s = s.replace("{{param." + k + "}}", v)
                s = s.replace("{{" + k + "}}", v)  # shorthand
            return s
        if isinstance(obj, dict):
            return {k: substitute(v, msg, token, params) for k, v in obj.items()}
        if isinstance(obj, list):
            return [substitute(i, msg, token, params) for i in obj]
        return obj

    req_body = substitute(ep.body_template, user_message, backend_token, params) if ep.body_template else {"query": user_message}
    req_headers = {k: substitute(v, user_message, backend_token, params) for k, v in ep.headers.items()}
    req_url = substitute(ep.url, user_message, backend_token, params)
    trace_request = {
        "method": ep.method.upper(),
        "url": req_url,
        "headers": req_headers,
        "body": req_body,
        "token": backend_token,
    }
    trace_request["curl"] = build_curl_command(
        trace_request["method"],
        trace_request["url"],
        trace_request["headers"],
        trace_request["body"],
    )
    if TRACE_ENABLED:
        logger.info(
            "trace.chat.request.pre_submit=%s",
            json.dumps(_mask_token_in_trace(trace_request), ensure_ascii=False),
        )

    try:
        async with httpx.AsyncClient(timeout=60, verify=SSL_VERIFY) as client:
            resp = await client.request(ep.method, req_url, json=req_body, headers=req_headers)
        trace_response = {
            "status_code": resp.status_code,
            "headers": dict(resp.headers),
            "body": resp.text,
        }
        if TRACE_ENABLED:
            logger.info(
                "trace.chat.request=%s trace.chat.response=%s",
                json.dumps(_mask_token_in_trace(trace_request), ensure_ascii=False),
                json.dumps(trace_response, ensure_ascii=False),
            )
        if resp.status_code == 200:
            data = resp.json()
            def extract_field(data, field_path: str):
                """Extract value via dot-notation, supporting list indices."""
                val = data
                for part in field_path.split("."):
                    if isinstance(val, dict):
                        val = val.get(part, "")
                    elif isinstance(val, list) and part.isdigit():
                        idx = int(part)
                        val = val[idx] if idx < len(val) else ""
                    else:
                        break
                return val

            # If response_display is configured, return rich display items
            if ep.response_display:
                display_items = []
                for item in ep.response_display:
                    val = extract_field(data, item.field)
                    display_items.append({
                        "label": item.label or item.field,
                        "type": item.type,
                        "value": val,
                    })
                result = {"response": str(extract_field(data, ep.response_field)), "display": display_items}
                if TRACE_ENABLED:
                    result["trace"] = {
                        "request": _mask_token_in_trace(trace_request),
                        "response": trace_response,
                    }
                return JSONResponse(result)

            answer = extract_field(data, ep.response_field)
            result = {"response": str(answer)}
            if TRACE_ENABLED:
                result["trace"] = {
                    "request": _mask_token_in_trace(trace_request),
                    "response": trace_response,
                }
            return JSONResponse(result)
        else:
            result = {"response": f"Backend error (HTTP {resp.status_code}): {resp.text[:300]}"}
            if TRACE_ENABLED:
                result["trace"] = {
                    "request": _mask_token_in_trace(trace_request),
                    "response": trace_response,
                }
            return JSONResponse(result)
    except httpx.RequestError as e:
        result = {"response": f"Could not reach backend: {e}"}
        if TRACE_ENABLED:
            result["trace"] = {
                "request": _mask_token_in_trace(trace_request),
                "response": {
                    "error": str(e),
                },
            }
        return JSONResponse(result)

# ---------------------------------------------------------------------------
# Routes – logout
# ---------------------------------------------------------------------------
@app.get("/{slug}/logout")
async def logout(request: Request, slug: str):
    response = RedirectResponse(f"/{slug}/login", status_code=302)
    delete_backend_token_ref(request.cookies.get("backend_token_ref", ""))
    response.delete_cookie("session_token")
    response.delete_cookie("backend_token_ref")
    response.delete_cookie("backend_token")
    return response

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8200, reload=True)
