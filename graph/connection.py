"""Neo4j connection settings shared by the migration runner and the ingestion pipeline.

Phase 2 - owned by the graph-engineer subagent. See PLAN.md section 3, Phase 2.

Credentials are read from the process environment first and from the repo-root
``.env`` (the same file ``docker-compose.yml`` reads) as a fallback.  They are
never hardcoded and never printed: :class:`Neo4jSettings` redacts the password in
both ``repr`` and ``str``.

Only the Neo4j *bolt* driver is used - no HTTP, no APOC - so this works against
the local ``neo4j:5-community`` container and against AuraDB Free (PLAN.md
section 10) without changes.  Community edition has exactly one user database, so
tests namespace themselves with ``run_id`` rather than creating a database.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from neo4j import Driver

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_DATABASE = "neo4j"

__all__ = [
    "DEFAULT_DATABASE",
    "DEFAULT_ENV_FILE",
    "Neo4jSettings",
    "connect",
    "neo4j_driver",
    "parse_env_file",
]


def parse_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """Parse a dotenv-style file into a dict. Missing file -> empty dict.

    Deliberately dependency-free (no python-dotenv): the format we need is
    ``KEY=value`` with ``#`` comments, optional ``export`` prefix and optional
    surrounding quotes.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        out[key] = value
    return out


@dataclass(frozen=True)
class Neo4jSettings:
    """Where to reach Neo4j. The password never appears in repr/str output."""

    uri: str
    username: str
    password: str
    database: str = DEFAULT_DATABASE

    @classmethod
    def from_env(
        cls,
        env_file: str | os.PathLike[str] | None = DEFAULT_ENV_FILE,
        environ: Mapping[str, str] | None = None,
    ) -> "Neo4jSettings":
        """Build settings from ``os.environ``, falling back to ``env_file``.

        Raises ``RuntimeError`` naming the missing variable rather than failing
        later with an opaque authentication error.
        """
        env = dict(os.environ if environ is None else environ)
        file_env = parse_env_file(env_file) if env_file is not None else {}

        def pick(key: str, default: str | None = None) -> str | None:
            value = env.get(key) or file_env.get(key) or default
            return value or None

        uri = pick("NEO4J_URI", "bolt://localhost:7687")
        username = pick("NEO4J_USERNAME", "neo4j")
        password = pick("NEO4J_PASSWORD")
        database = pick("NEO4J_DATABASE", DEFAULT_DATABASE)
        missing = [
            name
            for name, value in (
                ("NEO4J_URI", uri),
                ("NEO4J_USERNAME", username),
                ("NEO4J_PASSWORD", password),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"missing Neo4j credentials: {', '.join(missing)}. "
                f"Set them in the environment or in {env_file} (see .env.example)."
            )
        assert uri and username and password and database  # for type checkers
        return cls(uri=uri, username=username, password=password, database=database)

    # Redaction: these objects end up in log lines and pytest failure output.
    def __repr__(self) -> str:
        return (
            f"Neo4jSettings(uri={self.uri!r}, username={self.username!r}, "
            f"password='***', database={self.database!r})"
        )

    __str__ = __repr__


def connect(settings: Neo4jSettings | None = None, *, verify: bool = True) -> "Driver":
    """Open a driver. The caller owns closing it - prefer :func:`neo4j_driver`."""
    from neo4j import GraphDatabase

    cfg = settings or Neo4jSettings.from_env()
    driver = GraphDatabase.driver(cfg.uri, auth=(cfg.username, cfg.password))
    if verify:
        try:
            driver.verify_connectivity()
        except Exception:
            driver.close()
            raise
    return driver


@contextmanager
def neo4j_driver(
    settings: Neo4jSettings | None = None, *, verify: bool = True
) -> Iterator["Driver"]:
    """Context-managed driver, closed on exit."""
    driver = connect(settings, verify=verify)
    try:
        yield driver
    finally:
        driver.close()
