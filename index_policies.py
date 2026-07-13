#!/usr/bin/env python3
#
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#

"""
Index policy YAML files into a local Milvus Lite vector database.

Chunks each policy (reply_cannot_contain / reply_may_contain / per-risk
description items, plus one policy-level "topic" summary used for topic
similarity), embeds every chunk once with a sentence-transformers model, and
upserts them into Milvus.

This is the only script that needs the heavy ML stack (sentence-transformers /
torch). compare_policies_embeddings_multi.py only ever reads precomputed
vectors back out of Milvus, so re-running comparisons — or changing which
locations a policy applies to — never triggers re-embedding.

Re-running this script for a policy re-embeds only that policy: existing rows
for the same policy_label are deleted and replaced; every other policy already
in the collection is left untouched.

Requirements:
    pip install sentence-transformers pymilvus

Usage:
    python3 index_policies.py <policy1.yaml> [<policy2.yaml> ...]
    python3 index_policies.py <policies_dir/>
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore", message=".*position_ids.*")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    from pymilvus import MilvusClient, DataType
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("index_policies")


def load_policy(file_path: str) -> dict:
    """Load a policy YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def _label_for(path: str, seen: dict) -> str:
    """Build a short, unique, human-readable label for a policy from its filename.
    This label is the join key used by compare_policies_embeddings_multi.py and
    policy_locations.yaml to find this policy's rows in Milvus."""
    stem = Path(path).stem
    count = seen.get(stem, 0)
    seen[stem] = count + 1
    return stem if count == 0 else f"{stem}_{count + 1}"


def _resolve_policy_paths(inputs: list[str]) -> list[Path]:
    """Expand any directories in `inputs` into the .yaml/.yml files they contain
    (recursively), and pass individual file paths through unchanged."""
    resolved: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found = sorted(set(path.rglob("*.yaml")) | set(path.rglob("*.yml")))
            if not found:
                print(f"Warning: no YAML files found under directory '{raw}'", file=sys.stderr)
            resolved.extend(found)
        else:
            resolved.append(path)
    return resolved


def build_rows(policy: dict, label: str, file_path: str) -> list[dict]:
    """Build the (un-embedded) Milvus rows for one policy: one row per
    cannot_contain/may_contain/description chunk, plus one 'topic' row holding
    a whole-policy summary used for topic-level similarity."""
    rows: list[dict] = []

    for risk in policy.get('risks', []):
        risk_name = risk.get('risk', 'unknown')
        risk_id = str(risk.get('risk_id', ''))
        policy_rules = risk.get('policy', {})

        description = (risk.get('description') or '').strip()
        if description:
            rows.append({
                "text": description, "chunk_type": "description",
                "risk_name": risk_name, "risk_id": risk_id,
            })

        for item in policy_rules.get('reply_cannot_contain', []) or []:
            if item and item.strip():
                rows.append({
                    "text": item.strip(), "chunk_type": "cannot_contain",
                    "risk_name": risk_name, "risk_id": risk_id,
                })

        for item in policy_rules.get('reply_may_contain', []) or []:
            if item and item.strip():
                rows.append({
                    "text": item.strip(), "chunk_type": "may_contain",
                    "risk_name": risk_name, "risk_id": risk_id,
                })

    topic_text = f"{policy.get('risk_group', '')} {policy.get('description', '')}"
    for risk in policy.get('risks', []):
        topic_text += f" {risk.get('risk', '')} {risk.get('description', '')}"
    rows.append({
        "text": topic_text.strip(), "chunk_type": "topic",
        "risk_name": "", "risk_id": "",
    })

    for row in rows:
        row["policy_label"] = label
        row["file_path"] = file_path
        is_topic = row["chunk_type"] == "topic"
        row["risk_group"] = policy.get('risk_group', '') if is_topic else ''
        row["policy_description"] = policy.get('description', '') if is_topic else ''

    return rows


def ensure_collection(client: 'MilvusClient', collection_name: str, dim: int) -> None:
    """Create the collection with the shared schema if it doesn't exist yet."""
    if client.has_collection(collection_name):
        return
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("chunk_type", DataType.VARCHAR, max_length=32)
    schema.add_field("risk_name", DataType.VARCHAR, max_length=256)
    schema.add_field("risk_id", DataType.VARCHAR, max_length=64)
    schema.add_field("policy_label", DataType.VARCHAR, max_length=256)
    schema.add_field("file_path", DataType.VARCHAR, max_length=2048)
    schema.add_field("risk_group", DataType.VARCHAR, max_length=256)
    schema.add_field("policy_description", DataType.VARCHAR, max_length=4096)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


def index_policy(client: 'MilvusClient', collection_name: str, model: 'SentenceTransformer',
                  policy: dict, label: str, file_path: str) -> int:
    """Embed and upsert one policy's rows. Deletes any existing rows for this
    policy_label first, so re-indexing a changed policy never leaves stale chunks
    behind, and never touches rows belonging to other policies."""
    rows = build_rows(policy, label, file_path)
    texts = [r["text"] for r in rows]
    embeddings = model.encode(texts, convert_to_tensor=False)

    ensure_collection(client, collection_name, dim=len(embeddings[0]))
    client.load_collection(collection_name)
    client.delete(collection_name=collection_name, filter=f'policy_label == "{label}"')

    for row, vector in zip(rows, embeddings):
        row["embedding"] = vector.tolist()

    client.insert(collection_name=collection_name, data=rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Chunk, embed, and index policy YAML files into a local Milvus Lite database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('policies', nargs='+',
                        help='Policy YAML files and/or directories to index recursively')
    parser.add_argument('--milvus-db', default='policy_embeddings.db',
                        help='Path to the local Milvus Lite database file (default: policy_embeddings.db)')
    parser.add_argument('--collection', default='policy_chunks',
                        help='Milvus collection name to store chunk embeddings in (default: policy_chunks)')
    parser.add_argument('--model', '-m', default='all-MiniLM-L6-v2',
                        help='Sentence transformer model to use (default: all-MiniLM-L6-v2)')
    args = parser.parse_args()

    if not EMBEDDINGS_AVAILABLE:
        print("Error: sentence-transformers is required for this script.", file=sys.stderr)
        print("Install with: pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)
    if not MILVUS_AVAILABLE:
        print("Error: pymilvus is required for this script.", file=sys.stderr)
        print("Install with: pip install pymilvus", file=sys.stderr)
        sys.exit(1)

    for path in args.policies:
        if not Path(path).exists():
            print(f"Error: Path not found: {path}", file=sys.stderr)
            sys.exit(1)

    policy_paths = _resolve_policy_paths(args.policies)
    if not policy_paths:
        print("Error: no policy YAML files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading embedding model: {args.model}...", file=sys.stderr)
    model = SentenceTransformer(args.model)
    client = MilvusClient(uri=args.milvus_db)

    seen_labels: dict = {}
    indexed = 0
    for path in policy_paths:
        label = _label_for(str(path), seen_labels)
        try:
            policy = load_policy(str(path))
        except Exception as e:
            print(f"Error loading '{path}': {e}", file=sys.stderr)
            continue
        num_rows = index_policy(client, args.collection, model, policy, label, str(path))
        print(f"Indexed '{label}' ({path}): {num_rows} chunks", file=sys.stderr)
        indexed += 1

    client.close()
    print(f"Done. Indexed {indexed}/{len(policy_paths)} policies into "
          f"{args.milvus_db!r} (collection={args.collection!r}).", file=sys.stderr)


if __name__ == '__main__':
    main()
