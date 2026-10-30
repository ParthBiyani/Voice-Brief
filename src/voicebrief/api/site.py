"""One process, one port: the web app at `/` and the API under `/api`.

`voicebrief serve` runs this. In development the Vite dev server proxies `/api` to the
API; in use, FastAPI serves the built app itself, so there is nothing else to run and a
phone only needs one address.

The API can be protected with `VB_API_TOKEN`. It matters as soon as the app is reachable
beyond this computer: generating a brief spends money and the audio is personal. The
token is accepted as a bearer header, an `X-VB-Token` header, a `vb_token` cookie (which
is what lets an `<audio>` element and an event stream authenticate) or a `?token=` query
parameter, and the web app turns the last into the cookie on first use.
"""

from __future__ import annotations

import hmac
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from voicebrief.api.main import app as api_app
from voicebrief.config import get_settings
from voicebrief.logging import get_logger

log = get_logger(__name__)

OPEN_PATHS = frozenset({"/health"})
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


def is_loopback(host: str) -> bool:
    return host in LOOPBACK


def presented_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return (
        request.headers.get("x-vb-token")
        or request.cookies.get("vb_token")
        or request.query_params.get("token")
    )


class TokenGate(BaseHTTPMiddleware):
    """Reject API calls that do not carry the configured token."""

    def __init__(self, app, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):
        # Under the /api mount the request path still carries the prefix.
        root = request.scope.get("root_path", "")
        route = request.url.path
        if root and route.startswith(root):
            route = route[len(root) :]
        if request.method == "OPTIONS" or route in OPEN_PATHS:
            return await call_next(request)
        presented = presented_token(request) or ""
        if not hmac.compare_digest(presented.encode(), self._token.encode()):
            return JSONResponse({"detail": "access token required"}, status_code=401)
        return await call_next(request)


class SPAStaticFiles(StaticFiles):
    """Serve the built app, falling back to index.html so deep links survive a reload."""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


def default_dist() -> Path:
    configured = get_settings().web_dist
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / "web" / "dist"


def build_site(dist: Path | None = None, *, token: str | None = None) -> FastAPI:
    """The combined application. `token` defaults to VB_API_TOKEN."""
    settings = get_settings()
    secret = (
        token
        if token is not None
        else (settings.api_token.get_secret_value() if settings.api_token else None)
    )
    if secret:
        api_app.add_middleware(TokenGate, token=secret)

    site = FastAPI(title="VoiceBrief site", docs_url=None, redoc_url=None, openapi_url=None)
    site.mount("/api", api_app)

    folder = dist or default_dist()
    if (folder / "index.html").is_file():
        site.mount("/", SPAStaticFiles(directory=folder, html=True), name="web")
    else:
        log.warning("site.no_web_build", looked_in=str(folder), hint="cd web && npm run build")
    return site
