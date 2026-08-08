---
name: sync-graph-schema
description: Regenerate and apply a Neo4j migration after the graph schema changes. Use after editing graph/schema.cypher.
disable-model-invocation: true
---

1. Diff graph/schema.cypher against the latest file in graph/migrations/
2. Write the next numbered migration capturing only the delta
3. Apply it to the local dev database
4. Update graph/ingestion_pipeline.py if the change affects what
   ingestion writes
