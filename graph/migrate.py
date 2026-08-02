"""Numbered-migration runner for the Aegis knowledge graph.

Phase 2 - owned by the graph-engineer subagent. See PLAN.md section 3, Phase 2.
CLAUDE.md makes this non-negotiable: *Cypher migrations are numbered files under
graph/migrations/*.

    python -m graph.migrate            # status: what is applied, what is pending
    python -m graph.migrate --apply    # apply everything pending, in order
    python -m graph.migrate --apply --dry-run

Design
------
Applied migrations are recorded as ``(:_AegisMigration {version, name, checksum,
applied_at})`` nodes in the same database they mutate, so "what schema is this
database at?" is answerable from the database itself rather than from a file on
someone's laptop.  Re-running is a no-op, which is what makes it safe to call
from a test fixture or a container entrypoint.

``checksum`` is a SHA-256 of the migration's normalized statements.  If a file
that has already been applied is later edited, the runner refuses to continue
(:class:`MigrationDriftError`) instead of silently leaving the database in a
state no file describes.  Fix drift by writing the *next* numbered migration -
that is the whole point of numbering them.

Each statement runs in its own transaction: Neo4j does not allow schema commands
(``CREATE CONSTRAINT``) and data writes to share one, and a failed multi-statement
migration would otherwise roll back a constraint that later statements assume.
Statements are therefore written to be individually idempotent
(``IF NOT EXISTS`` / ``MERGE``), so a migration that fails halfway can be re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from graph.connection import DEFAULT_DATABASE, Neo4jSettings, neo4j_driver

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from neo4j import Driver

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
SCHEMA_FILE = Path(__file__).resolve().parent / "schema.cypher"

#: Migration file names must be `NNN_snake_case_name.cypher`.
FILENAME_RE = re.compile(r"^(?P<version>\d{3,})_(?P<name>[A-Za-z0-9][A-Za-z0-9_\-]*)\.cypher$")

_MIGRATION_LABEL = "_AegisMigration"
_BOOKKEEPING_CONSTRAINT = (
    f"CREATE CONSTRAINT aegis_migration_version IF NOT EXISTS "
    f"FOR (m:{_MIGRATION_LABEL}) REQUIRE m.version IS UNIQUE"
)

__all__ = [
    "MIGRATIONS_DIR",
    "Migration",
    "MigrationDriftError",
    "apply_migrations",
    "applied_migrations",
    "discover_migrations",
    "pending_migrations",
    "split_statements",
]


class MigrationDriftError(RuntimeError):
    """An already-applied migration file no longer matches what was applied."""


def split_statements(text: str) -> list[str]:
    """Split a .cypher file into individual statements.

    Strips ``//`` line comments and splits on ``;``.  Migrations are hand-written
    schema DDL, so the simple rule holds - but it does mean a literal ``//`` or
    ``;`` inside a string literal would be mis-split.  Keep migrations to DDL and
    parameter-free MERGEs and that never comes up.
    """
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].rstrip()
        if line.strip():
            lines.append(line)
    joined = "\n".join(lines)
    return [s.strip() for s in joined.split(";") if s.strip()]


def _normalize(statements: list[str]) -> str:
    """Whitespace-insensitive canonical form, for checksums and drift checks."""
    return "\n".join(re.sub(r"\s+", " ", s).strip() for s in statements)


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path
    statements: tuple[str, ...]

    @property
    def checksum(self) -> str:
        digest = hashlib.sha256(_normalize(list(self.statements)).encode("utf-8"))
        return digest.hexdigest()

    @property
    def label(self) -> str:
        return f"{self.version:03d}_{self.name}"

    @classmethod
    def from_path(cls, path: Path) -> "Migration":
        match = FILENAME_RE.match(path.name)
        if match is None:
            raise ValueError(
                f"migration filename {path.name!r} must look like 001_initial_schema.cypher"
            )
        statements = split_statements(path.read_text(encoding="utf-8"))
        if not statements:
            raise ValueError(f"migration {path.name!r} contains no statements")
        return cls(
            version=int(match.group("version")),
            name=match.group("name"),
            path=path,
            statements=tuple(statements),
        )


def discover_migrations(directory: Path | str = MIGRATIONS_DIR) -> list[Migration]:
    """All migrations in ``directory``, ordered by version. Duplicate versions raise."""
    directory = Path(directory)
    migrations = [
        Migration.from_path(p)
        for p in sorted(directory.glob("*.cypher"))
        if FILENAME_RE.match(p.name)
    ]
    seen: dict[int, Migration] = {}
    for m in migrations:
        if m.version in seen:
            raise ValueError(
                f"duplicate migration version {m.version:03d}: "
                f"{seen[m.version].path.name} and {m.path.name}"
            )
        seen[m.version] = m
    return sorted(migrations, key=lambda m: m.version)


def ensure_bookkeeping(driver: "Driver", database: str = DEFAULT_DATABASE) -> None:
    """Create the constraint backing the applied-migration ledger. Idempotent."""
    driver.execute_query(_BOOKKEEPING_CONSTRAINT, database_=database)


def applied_migrations(
    driver: "Driver", database: str = DEFAULT_DATABASE
) -> dict[int, dict[str, str]]:
    """Version -> {name, checksum, applied_at} for everything already applied."""
    ensure_bookkeeping(driver, database)
    result = driver.execute_query(
        f"MATCH (m:{_MIGRATION_LABEL}) "
        "RETURN m.version AS version, m.name AS name, m.checksum AS checksum, "
        "toString(m.applied_at) AS applied_at ORDER BY version",
        database_=database,
    )
    return {
        int(r["version"]): {
            "name": r["name"],
            "checksum": r["checksum"],
            "applied_at": r["applied_at"],
        }
        for r in result.records
    }


def _check_drift(migrations: list[Migration], applied: dict[int, dict[str, str]]) -> None:
    for m in migrations:
        record = applied.get(m.version)
        if record is not None and record["checksum"] != m.checksum:
            raise MigrationDriftError(
                f"{m.path.name} was already applied on {record['applied_at']} but its "
                f"contents have changed since. Do not edit an applied migration - add "
                f"the next numbered file instead."
            )


def pending_migrations(
    driver: "Driver",
    database: str = DEFAULT_DATABASE,
    directory: Path | str = MIGRATIONS_DIR,
) -> list[Migration]:
    """Migrations present on disk and not yet recorded in the database."""
    migrations = discover_migrations(directory)
    applied = applied_migrations(driver, database)
    _check_drift(migrations, applied)
    return [m for m in migrations if m.version not in applied]


def apply_migrations(
    driver: "Driver",
    database: str = DEFAULT_DATABASE,
    directory: Path | str = MIGRATIONS_DIR,
    *,
    dry_run: bool = False,
    log: bool = False,
) -> list[Migration]:
    """Apply every pending migration in version order. Returns what was applied."""
    pending = pending_migrations(driver, database, directory)
    if dry_run:
        return pending
    for migration in pending:
        for statement in migration.statements:
            driver.execute_query(statement, database_=database)
        driver.execute_query(
            f"MERGE (m:{_MIGRATION_LABEL} {{version: $version}}) "
            "SET m.name = $name, m.checksum = $checksum, "
            "m.statements = $statements, m.applied_at = datetime()",
            {
                "version": migration.version,
                "name": migration.name,
                "checksum": migration.checksum,
                "statements": len(migration.statements),
            },
            database_=database,
        )
        if log:
            print(f"  applied {migration.label} ({len(migration.statements)} statements)")
    return pending


def _print_status(driver: "Driver", database: str, directory: Path) -> None:
    migrations = discover_migrations(directory)
    applied = applied_migrations(driver, database)
    _check_drift(migrations, applied)
    print(f"migrations in {directory}")
    if not migrations:
        print("  (none)")
    for m in migrations:
        record = applied.get(m.version)
        if record is None:
            print(f"  [ ] {m.label:<28} pending")
        else:
            print(f"  [x] {m.label:<28} applied {record['applied_at']}")
    orphans = sorted(set(applied) - {m.version for m in migrations})
    for version in orphans:
        print(f"  [?] {version:03d}_{applied[version]['name']:<24} in database, no file")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply Aegis Neo4j migrations.")
    parser.add_argument("--apply", action="store_true", help="apply pending migrations")
    parser.add_argument(
        "--dry-run", action="store_true", help="with --apply, list without writing"
    )
    parser.add_argument("--dir", default=str(MIGRATIONS_DIR), help="migrations directory")
    parser.add_argument("--database", default=None, help="override NEO4J_DATABASE")
    args = parser.parse_args(argv)

    settings = Neo4jSettings.from_env()
    database = args.database or settings.database
    directory = Path(args.dir)

    with neo4j_driver(settings) as driver:
        print(f"neo4j: {settings.uri} database={database}")
        if not args.apply:
            _print_status(driver, database, directory)
            return 0
        applied = apply_migrations(
            driver, database, directory, dry_run=args.dry_run, log=True
        )
        if args.dry_run:
            for m in applied:
                print(f"  would apply {m.label} ({len(m.statements)} statements)")
        elif not applied:
            print("  nothing pending - database is up to date")
        print()
        _print_status(driver, database, directory)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
