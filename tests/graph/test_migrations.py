"""The migration runner, and the guarantee that schema.cypher never drifts from it.

CLAUDE.md: "Cypher migrations are numbered files under graph/migrations/."
PLAN.md Phase 2 is done when "every schema change is a numbered migration", so
these tests cover both halves of that: the files are well-formed and ordered, and
applying them to the live database is idempotent and actually creates the
constraints ingestion depends on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from graph.migrate import (
    MIGRATIONS_DIR,
    SCHEMA_FILE,
    Migration,
    MigrationDriftError,
    apply_migrations,
    applied_migrations,
    discover_migrations,
    pending_migrations,
    split_statements,
)
from tests.graph.conftest import TEST_MIGRATION_FLOOR, TEST_MIGRATION_LABEL

# ---------------------------------------------------------------- file layout


def test_migrations_directory_is_not_empty():
    migrations = discover_migrations()
    assert migrations, f"no migrations found in {MIGRATIONS_DIR}"


def test_migrations_are_numbered_uniquely_and_in_order():
    migrations = discover_migrations()
    versions = [m.version for m in migrations]
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)
    assert versions[0] == 1, "numbering starts at 001"


def test_every_migration_statement_is_idempotent():
    """Re-running a half-applied migration must not fail on what already exists."""
    for migration in discover_migrations():
        for statement in migration.statements:
            upper = statement.upper()
            assert "IF NOT EXISTS" in upper or upper.startswith("MERGE"), (
                f"{migration.path.name} has a statement that is not safe to re-run: "
                f"{statement!r}"
            )


def test_no_bare_create_in_migrations():
    """A bare CREATE would break the re-run guarantee the runner promises."""
    for migration in discover_migrations():
        for statement in migration.statements:
            stripped = statement.strip().upper()
            if stripped.startswith("CREATE") and "IF NOT EXISTS" not in stripped:
                pytest.fail(f"{migration.path.name}: bare CREATE - {statement!r}")


def test_split_statements_drops_comments_and_blank_statements():
    text = """
    // a leading comment
    CREATE CONSTRAINT a IF NOT EXISTS FOR (n:X) REQUIRE n.k IS UNIQUE;

    // another
    CREATE CONSTRAINT b IF NOT EXISTS FOR (n:Y) REQUIRE n.k IS UNIQUE;
    """
    statements = split_statements(text)
    assert len(statements) == 2
    assert all("//" not in s for s in statements)


def test_schema_cypher_matches_the_migrations():
    """schema.cypher is documentation; if it disagrees with migrations/, it lies.

    Compared as normalized statement sets, so comments and line wrapping in
    either file are free to differ - only the executable content has to match.
    """

    def normalize(statements: list[str]) -> set[str]:
        return {" ".join(s.split()) for s in statements}

    from_schema = normalize(split_statements(SCHEMA_FILE.read_text(encoding="utf-8")))
    from_migrations: set[str] = set()
    for migration in discover_migrations():
        from_migrations |= normalize(list(migration.statements))

    assert from_schema == from_migrations, (
        "graph/schema.cypher and graph/migrations/ have diverged.\n"
        f"only in schema.cypher: {sorted(from_schema - from_migrations)}\n"
        f"only in migrations/:   {sorted(from_migrations - from_schema)}"
    )


# ------------------------------------------------------------- against a live DB


def test_constraints_exist_after_apply(driver: Any, database: str):
    """The three (run_id, id) uniqueness constraints ingestion MERGEs against."""
    result = driver.execute_query(
        "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties "
        "RETURN name, type, labelsOrTypes, properties",
        database_=database,
    )
    found = {
        r["name"]: (r["type"], tuple(r["labelsOrTypes"] or ()), tuple(r["properties"] or ()))
        for r in result.records
    }
    for name, label in (
        ("aegis_service_key", "Service"),
        ("aegis_pod_key", "Pod"),
        ("aegis_node_key", "Node"),
    ):
        assert name in found, f"{name} missing; found {sorted(found)}"
        kind, labels, props = found[name]
        assert "UNIQUE" in kind.upper()
        assert labels == (label,)
        assert props == ("run_id", "id")


def test_uniqueness_constraint_is_actually_enforced(driver: Any, database: str, run_id: str):
    """Not just declared - the database rejects a duplicate (run_id, id)."""
    from neo4j.exceptions import ConstraintError

    driver.execute_query(
        "CREATE (:Service {run_id: $r, id: 'dupe'})", {"r": run_id}, database_=database
    )
    with pytest.raises(ConstraintError):
        driver.execute_query(
            "CREATE (:Service {run_id: $r, id: 'dupe'})",
            {"r": run_id},
            database_=database,
        )
    driver.execute_query(
        "MATCH (s:Service {run_id: $r}) DETACH DELETE s", {"r": run_id}, database_=database
    )


def test_applying_twice_is_a_no_op(driver: Any, database: str):
    assert apply_migrations(driver, database) == []
    assert pending_migrations(driver, database) == []


def test_every_file_is_recorded_in_the_ledger(driver: Any, database: str):
    apply_migrations(driver, database)
    applied = applied_migrations(driver, database)
    for migration in discover_migrations():
        assert migration.version in applied, f"{migration.label} not in the ledger"
        assert applied[migration.version]["checksum"] == migration.checksum


def _write_probe_migration(directory: Path, version: int, key: str) -> Path:
    path = directory / f"{version:03d}_probe_{key}.cypher"
    path.write_text(
        f"// probe migration, deleted by the test fixture\n"
        f"MERGE (n:{TEST_MIGRATION_LABEL} {{id: '{key}'}});\n",
        encoding="utf-8",
    )
    return path


def test_runner_applies_pending_in_version_order(
    driver: Any, database: str, tmp_path: Path
):
    """Out-of-order filenames still apply low-to-high, and only once."""
    _write_probe_migration(tmp_path, TEST_MIGRATION_FLOOR + 2, "c")
    _write_probe_migration(tmp_path, TEST_MIGRATION_FLOOR, "a")
    _write_probe_migration(tmp_path, TEST_MIGRATION_FLOOR + 1, "b")

    applied = apply_migrations(driver, database, tmp_path)
    assert [m.version for m in applied] == [
        TEST_MIGRATION_FLOOR,
        TEST_MIGRATION_FLOOR + 1,
        TEST_MIGRATION_FLOOR + 2,
    ]

    count = driver.execute_query(
        f"MATCH (n:{TEST_MIGRATION_LABEL}) RETURN count(n) AS c", database_=database
    ).records[0]["c"]
    assert count == 3

    # Second pass: nothing pending, nothing written.
    assert apply_migrations(driver, database, tmp_path) == []
    count_again = driver.execute_query(
        f"MATCH (n:{TEST_MIGRATION_LABEL}) RETURN count(n) AS c", database_=database
    ).records[0]["c"]
    assert count_again == 3


def test_a_new_file_is_picked_up_without_replaying_the_old_ones(
    driver: Any, database: str, tmp_path: Path
):
    _write_probe_migration(tmp_path, TEST_MIGRATION_FLOOR + 10, "first")
    assert len(apply_migrations(driver, database, tmp_path)) == 1

    _write_probe_migration(tmp_path, TEST_MIGRATION_FLOOR + 11, "second")
    applied = apply_migrations(driver, database, tmp_path)
    assert [m.version for m in applied] == [TEST_MIGRATION_FLOOR + 11]


def test_editing_an_applied_migration_is_refused(
    driver: Any, database: str, tmp_path: Path
):
    """Drift detection: the fix for a mistake is the next number, not an edit."""
    path = _write_probe_migration(tmp_path, TEST_MIGRATION_FLOOR + 20, "drift")
    apply_migrations(driver, database, tmp_path)

    path.write_text(
        f"MERGE (n:{TEST_MIGRATION_LABEL} {{id: 'drift', changed: true}});\n",
        encoding="utf-8",
    )
    with pytest.raises(MigrationDriftError, match="contents have changed"):
        pending_migrations(driver, database, tmp_path)


def test_badly_named_files_are_ignored_and_bad_content_raises(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("not a migration", encoding="utf-8")
    (tmp_path / "no_number.cypher").write_text("RETURN 1;", encoding="utf-8")
    assert discover_migrations(tmp_path) == []

    (tmp_path / "001_empty.cypher").write_text("// only a comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no statements"):
        discover_migrations(tmp_path)


def test_duplicate_version_numbers_raise(tmp_path: Path):
    _write_probe_migration(tmp_path, 1, "one")
    _write_probe_migration(tmp_path, 1, "two")
    with pytest.raises(ValueError, match="duplicate migration version"):
        discover_migrations(tmp_path)


def test_checksum_is_whitespace_insensitive(tmp_path: Path):
    a = tmp_path / "001_a.cypher"
    a.write_text("CREATE  CONSTRAINT x IF NOT EXISTS\nFOR (n:X)\nREQUIRE n.k IS UNIQUE;\n")
    first = Migration.from_path(a).checksum
    a.write_text("CREATE CONSTRAINT x IF NOT EXISTS FOR (n:X) REQUIRE n.k IS UNIQUE;\n")
    assert Migration.from_path(a).checksum == first
