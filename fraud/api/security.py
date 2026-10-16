"""Capa de seguridad del serving REST — framework SAIF (clase 7).

Centraliza los controles de seguridad que la clase de seguridad (SAIF de Google)
mapea sobre un sistema de detección de fraude. Cada pieza de este módulo es una
capa de la matriz *componente → riesgo → control*:

- **Capa 1 · AuthN/AuthZ**: API keys con **roles** (analyst / mlops / admin). No se
  expone el modelo sin control de acceso; además cada rol sólo puede tocar lo suyo.
- **Capa 2 · Validación de entradas**: la hace Pydantic (`schemas.py`); acá sólo
  se referencia para dejar el mapa completo.
- **Capa 3 · Rate limiting**: `slowapi` limita las peticiones por IP+key para
  frenar abuso / scraping masivo / denegación de servicio.
- **Capa 4 · Secretos**: las keys se leen de variables de entorno (nunca en el
  código ni en el repo). El fallback sólo existe para dev/demo local.
- **Capa 8 · Operación y gobernanza**: log de **auditoría** estructurado (JSON) por
  cada evento relevante (quién, qué, cuándo, resultado).

Compatibilidad: si sólo se define `API_KEYS` (como venía), ese token funciona como
**admin** y puede hacer todo — el comportamiento previo no cambia.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

from fastapi import Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from loguru import logger

# --------------------------------------------------------------------------- #
# Roles (Capa 1 — Autorización / RBAC)
# --------------------------------------------------------------------------- #
ROLE_ADMIN = "admin"      # puede todo (incluye analista + mlops)
ROLE_ANALYST = "analyst"  # revisión de transacciones (UC2/UC3)
ROLE_MLOPS = "mlops"      # entrenar/evaluar/deploy de modelos (UC4/UC5/UC6)


def _parse_keys(env_name: str) -> set[str]:
    """Lee una lista de API keys separadas por coma desde una env var."""
    raw = os.getenv(env_name, "")
    return {t.strip() for t in raw.split(",") if t.strip()}


# Capa 4 — Secretos desde el entorno. `API_KEYS` mantiene compatibilidad hacia
# atrás: si no hay nada, cae al token de demo (sólo dev/local, NUNCA producción).
ADMIN_KEYS = _parse_keys("API_KEYS") or {"token-secreto-123"}
ANALYST_KEYS = _parse_keys("ANALYST_API_KEYS")
MLOPS_KEYS = _parse_keys("MLOPS_API_KEYS")


def _mask(api_key: str) -> str:
    """Identificador no sensible del token para los logs (sólo últimos 4)."""
    return f"***{api_key[-4:]}" if api_key and len(api_key) >= 4 else "***"


@dataclass(frozen=True)
class Principal:
    """Quién hace la llamada: su rol y un id enmascarado del token (para auditar)."""

    key_id: str
    role: str


api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


def _role_for(api_key: str | None) -> str | None:
    """Resuelve el rol de una API key (admin tiene prioridad)."""
    if not api_key:
        return None
    if api_key in ADMIN_KEYS:
        return ROLE_ADMIN
    if api_key in MLOPS_KEYS:
        return ROLE_MLOPS
    if api_key in ANALYST_KEYS:
        return ROLE_ANALYST
    return None


def authenticate(api_key: str = Security(api_key_header)) -> Principal:
    """Capa 1: valida la API key y devuelve el `Principal` (rol + id enmascarado).

    403 si la key es inválida o ausente. Deja rastro en el log de auditoría.
    """
    role = _role_for(api_key)
    if role is None:
        audit("auth.denied", reason="api_key_invalida_o_ausente")
        raise HTTPException(status_code=403, detail="API Key inválida o ausente")
    return Principal(key_id=_mask(api_key), role=role)


def require_role(*allowed_roles: str):
    """Dependencia FastAPI que exige uno de `allowed_roles` (admin siempre pasa).

    Uso: ``principal = Depends(require_role(ROLE_MLOPS))``.
    """

    def _dependency(principal: Principal = Depends(authenticate)) -> Principal:
        if principal.role == ROLE_ADMIN or principal.role in allowed_roles:
            return principal
        audit(
            "authz.denied",
            principal=principal,
            required=list(allowed_roles),
        )
        raise HTTPException(
            status_code=403,
            detail=f"Rol '{principal.role}' no autorizado (se requiere: {', '.join(allowed_roles)})",
        )

    return _dependency


# --------------------------------------------------------------------------- #
# Capa 3 — Rate limiting (slowapi). Tolerante: si la lib no está instalada
# (p. ej. dev local sin la dependencia), el rate limiting queda como no-op.
# --------------------------------------------------------------------------- #
# Límite por defecto del endpoint de scoring. Para la demo, bajalo con la env
# RATE_LIMIT (p. ej. "5/minute") y vas a ver el 429 enseguida.
RATE_LIMIT = os.getenv("RATE_LIMIT", "60/minute")

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    def _rate_key(request) -> str:
        """Clave del rate limit: la API key (si viene) o la IP de origen."""
        return request.headers.get("X-API-KEY") or get_remote_address(request)

    limiter = Limiter(key_func=_rate_key)
    SLOWAPI_OK = True
except Exception as exc:  # noqa: BLE001 - rate limiting es opcional en dev local
    limiter = None
    SLOWAPI_OK = False
    logger.debug(f"slowapi no disponible; rate limiting deshabilitado: {exc}")


def rate_limit(limit_value: str = RATE_LIMIT):
    """Decorador de rate limit tolerante: aplica slowapi si está, si no es no-op.

    El endpoint decorado debe declarar un parámetro ``request: Request`` (requisito
    de slowapi). Si la lib no está instalada, el decorador devuelve la función tal cual.
    """

    def _decorator(func):
        if SLOWAPI_OK and limiter is not None:
            return limiter.limit(limit_value)(func)
        return func

    return _decorator


# --------------------------------------------------------------------------- #
# Capa 8 — Auditoría (log estructurado JSON por evento)
# --------------------------------------------------------------------------- #
AUDIT_LOG = os.getenv("AUDIT_LOG", "logs/audit.log")
_audit_ready = False


def _ensure_audit_sink() -> None:
    """Agrega (una vez) un sink de loguru que escribe sólo los registros de auditoría."""
    global _audit_ready
    if _audit_ready:
        return
    try:
        Path(AUDIT_LOG).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            AUDIT_LOG,
            filter=lambda record: record["extra"].get("audit", False),
            rotation="10 MB",
            retention=5,
            enqueue=True,
        )
    except Exception as exc:  # noqa: BLE001 - si no se puede escribir el archivo, auditamos a stderr igual
        logger.debug(f"No se pudo crear el sink de auditoría {AUDIT_LOG}: {exc}")
    _audit_ready = True


def audit(action: str, principal: Principal | None = None, **fields) -> None:
    """Registra un evento de auditoría (quién, qué, resultado) como línea JSON.

    Va al archivo `AUDIT_LOG` y también a stderr (visible en `docker logs`). Nunca
    loguea el token completo: sólo el `key_id` enmascarado.
    """
    _ensure_audit_sink()
    from datetime import UTC, datetime

    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "action": action,
        "role": getattr(principal, "role", None),
        "key_id": getattr(principal, "key_id", None),
        **fields,
    }
    logger.bind(audit=True).info("AUDIT {}", json.dumps(record, default=str))


def read_audit(limit: int = 50) -> list[dict]:
    """Devuelve los últimos `limit` eventos de auditoría parseados (más nuevos primero).

    Lee el archivo `AUDIT_LOG` y extrae el JSON que sigue a ``AUDIT `` en cada línea.
    Lo usa el endpoint `/v1/audit` para que la UI muestre la auditoría sin terminal.
    """
    entries: list[dict] = []
    try:
        with open(AUDIT_LOG, encoding="utf-8") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return entries
    for line in lines[-limit:]:
        idx = line.find("AUDIT ")
        if idx == -1:
            continue
        try:
            entries.append(json.loads(line[idx + len("AUDIT ") :]))
        except json.JSONDecodeError:
            continue
    return list(reversed(entries))
