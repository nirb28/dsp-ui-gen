# DSP UI Generator

Configuration-based UI generator that serves professional login + chatbot experiences from YAML config files. Designed to work with FD2 (Front Door) and JWT authentication services.

## Quick Start

```bash
pip install -r requirements.txt
python server.py
```

Server starts at **http://localhost:8200**. The index page lists all available UI configs.

## How It Works

1. Drop a YAML file into `configs/` — each file defines a complete UI experience
2. The server auto-loads all configs on startup (hot-reload via `POST /reload`)
3. Each config gets its own routes: `/{slug}/login` → `/{slug}/chat` → `/{slug}/logout`
4. The slug is the YAML filename without extension

## Config Schema

```yaml
name: "My Chatbot"              # Display name
description: "Optional desc"    # Shown on login page

theme:
  primary_color: "#2563eb"      # Brand colors
  secondary_color: "#1e40af"
  accent_color: "#3b82f6"
  background_color: "#f8fafc"
  text_color: "#1e293b"
  logo_url: ""                  # Optional logo image URL
  favicon_url: ""

auth:
  type: "simple"                # simple | ldap | jwt_endpoint
  endpoint: ""                  # Auth service URL (for ldap/jwt_endpoint)
  username_field: "username"    # Field name in auth request
  password_field: "password"
  token_response_field: "access_token"  # JWT field in auth response
  extra_fields: {}              # Additional fields sent with auth
  session_duration_minutes: 480

chat:
  system_prompt: "You are a helpful assistant."
  placeholder: "Type your message..."
  welcome_message: "Hello! How can I help?"
  show_endpoint_selector: false # Show dropdown to switch endpoints
  endpoints:
    - name: "default"
      url: "http://localhost:8080/query"
      method: "POST"
      streaming: false
      headers:                  # {{token}} replaced with backend JWT
        Authorization: "Bearer {{token}}"
      body_template:            # {{message}} replaced with user input
        query: "{{message}}"
        configuration_name: "default"
      response_field: "response"  # Dot-notation path to answer in response JSON
```

## Authentication Types

| Type | Description |
|------|-------------|
| `simple` | Accepts any credentials — for local dev/testing |
| `jwt_endpoint` | POSTs credentials to an endpoint, expects JWT back |
| `ldap` | Same as jwt_endpoint but sends `extra_fields` (domain, auth_type) for LDAP backends |

All types create a local session cookie. For `jwt_endpoint`/`ldap`, the backend JWT is stored separately and injected into chat requests via `{{token}}`.

## Example Configs

| Config | Auth | Description |
|--------|------|-------------|
| `simple-chatbot` | simple | Basic chatbot, no real auth |
| `fd2-rag-chatbot` | jwt_endpoint | FD2 RAG with JWT auth, query + retrieve |
| `fd2-multi-endpoint` | jwt_endpoint | Multiple FD2 endpoints with query expansion |
| `ldap-enterprise-chat` | ldap | Enterprise LDAP auth example |

## API Endpoints

- `GET /` — Index page listing all configs
- `POST /reload` — Reload configs from disk
- `GET /{slug}/login` — Login page
- `POST /{slug}/login` — Submit credentials
- `GET /{slug}/chat` — Chat page (requires session)
- `POST /{slug}/chat/send` — Send message to backend
- `GET /{slug}/logout` — Clear session

## Environment Variables

- `UI_GEN_JWT_SECRET` — Secret for local session tokens (default: dev secret)
- `UI_GEN_TRACE` — Enable request/response tracing (`true`/`false`, default: `false`).
  - When enabled, `/{slug}/chat/send` responses include a `trace` object with:
    - `request.method`, `request.url`, `request.headers`, `request.body`, `request.token`
    - `request.curl` (copy/paste-ready curl command)
    - `response.status_code`, `response.headers`, `response.body`
- `UI_GEN_SSL_VERIFY` — Enable TLS certificate verification for outbound HTTP calls (`true`/`false`, default: `true`).

## Integration with DSP Stack

- **dsp_ai_jwt**: Use `jwt_endpoint` auth type pointing to the JWT service
- **dsp_ai_rag2 / FD2**: Configure chat endpoints to point to FD2 query/retrieve URLs
- **Control Tower**: Load manifest to determine auth endpoint and FD2 URLs
- **LDAP**: Use `ldap` auth type with `extra_fields` for domain/auth_type
