#!/usr/bin/env python3
#
# Adapted from main/granite.trust.policy-tools/scripts/compare_policies_embeddings.py
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#

"""
Compare N Policies using precomputed embeddings from Milvus - Semantic similarity comparison.

Run index_policies.py first to chunk, embed, and upsert policies into a local Milvus Lite database; this
script only reads those precomputed vectors back out and compares them, so it
has no sentence-transformers/torch dependency and never re-embeds anything —
not even when you change which locations a policy applies to, since that
mapping lives entirely in policy_locations.yaml, outside of Milvus.

Every pair of selected policies is compared at a deeper semantic level:
- reply_cannot_contain vs reply_cannot_contain (agreement)
- reply_may_contain vs reply_may_contain (agreement)
- reply_cannot_contain vs reply_may_contain (conflict detection)

You select which policies to compare in one of two ways:
  1. --place <name>   Look up the policy_labels for this deployment location
                       in --locations-config (default: policy_locations.yaml).
  2. <policy.yaml ...> Pass policy YAML files/directories directly; their
                       filename stems are used as policy_labels (must already
                       be indexed under those labels).

The comparison report is always saved to a .md file (it's already valid
markdown — the frontend in webapp/ renders it directly).

Requirements:
    pip install pymilvus

Usage:
    python3 index_policies.py <policy1.yaml> <policy2.yaml> [<policy3.yaml> ...]
    python3 compare_policies_embeddings_multi.py --place us
    python3 compare_policies_embeddings_multi.py <policy1.yaml> <policy2.yaml> [--output <path>] [--threshold 0.7]

Examples:
    python3 index_policies.py policies/example_policies_drinking_beer/policy_files/
    python3 compare_policies_embeddings_multi.py --place eu
"""

import argparse
import itertools
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

try:
    from pymilvus import MilvusClient
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("compare_policies_embeddings_multi")


@dataclass
class PolicyChunk:
    """Represents a chunk of policy content with metadata."""
    text: str
    chunk_type: str  # 'cannot_contain', 'may_contain', or 'description'
    risk_name: str
    risk_id: str
    policy_source: str  # label of the policy this chunk came from


@dataclass
class SimilarityMatch:
    """Represents a similarity match between two chunks."""
    chunk1: PolicyChunk
    chunk2: PolicyChunk
    similarity: float
    match_type: str  # 'CONFLICT', 'AGREEMENT_CANNOT', 'AGREEMENT_MAY'


@dataclass
class PolicyIndex:
    """A policy's chunks and embeddings, fetched pre-computed from Milvus."""
    label: str
    file_path: str
    risk_group: str
    policy_description: str
    chunks: list[PolicyChunk]
    chunk_embeddings: Optional[np.ndarray]  # shape (n_chunks, dim), or None if no chunks
    desc_embedding: np.ndarray  # shape (dim,) — the policy-level "topic" embedding


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _cos_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def _num_risks(idx: PolicyIndex) -> int:
    return len({c.risk_id for c in idx.chunks if c.risk_id})


def load_location_config(path: str) -> dict[str, list[str]]:
    """Load the place -> policy_labels mapping. This is the only place that
    changes when a policy's applicability changes — no re-indexing needed."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    return data.get('locations', {}) or {}


def _label_for(path: str, seen: dict) -> str:
    """Build the same policy_label index_policies.py would have used for this file."""
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


def fetch_policy_rows(client: 'MilvusClient', collection_name: str,
                       labels: list[str]) -> dict[str, list[dict]]:
    """Pull every chunk (including the 'topic' summary row) for the given
    policy_labels out of Milvus in one query."""
    client.load_collection(collection_name)
    label_list = ", ".join(f'"{label}"' for label in labels)
    rows = client.query(
        collection_name=collection_name,
        filter=f'policy_label in [{label_list}]',
        output_fields=["text", "chunk_type", "risk_name", "risk_id", "policy_label",
                        "file_path", "risk_group", "policy_description", "embedding"],
    )
    by_label: dict[str, list[dict]] = {label: [] for label in labels}
    for row in rows:
        by_label.setdefault(row["policy_label"], []).append(row)
    return by_label


def build_policy_index_from_rows(label: str, rows: list[dict]) -> PolicyIndex:
    """Reconstruct a PolicyIndex purely from what's stored in Milvus — no YAML
    file or embedding model needed at compare time."""
    topic_rows = [r for r in rows if r["chunk_type"] == "topic"]
    chunk_rows = [r for r in rows if r["chunk_type"] != "topic"]

    if not topic_rows:
        raise ValueError(
            f"No indexed data found for policy_label={label!r}. "
            f"Run index_policies.py on it first."
        )
    topic_row = topic_rows[0]

    chunks = [
        PolicyChunk(
            text=r["text"], chunk_type=r["chunk_type"],
            risk_name=r["risk_name"], risk_id=r["risk_id"], policy_source=label,
        )
        for r in chunk_rows
    ]
    chunk_embeddings = (
        np.array([r["embedding"] for r in chunk_rows], dtype=np.float32) if chunk_rows else None
    )
    desc_embedding = np.array(topic_row["embedding"], dtype=np.float32)
    file_path = chunk_rows[0]["file_path"] if chunk_rows else topic_row["file_path"]

    return PolicyIndex(
        label=label,
        file_path=file_path,
        risk_group=topic_row["risk_group"],
        policy_description=topic_row["policy_description"],
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        desc_embedding=desc_embedding,
    )


def analyze_topic_similarity(idx1: PolicyIndex, idx2: PolicyIndex) -> dict:
    """Analyze overall topic similarity between two policies using their precomputed embeddings."""
    similarity = _cos_sim(idx1.desc_embedding, idx2.desc_embedding)

    if similarity >= 0.7:
        relationship = "SAME_TOPIC"
        explanation = "Policies are semantically about the same topic"
    elif similarity >= 0.4:
        relationship = "RELATED_TOPICS"
        explanation = "Policies have semantic overlap"
    else:
        relationship = "DIFFERENT_TOPICS"
        explanation = "Policies are semantically distinct - no conflicts expected"

    return {
        'relationship': relationship,
        'explanation': explanation,
        'similarity': round(similarity, 3)
    }


def find_semantic_matches(idx1: PolicyIndex, idx2: PolicyIndex, threshold: float = 0.6) -> list[SimilarityMatch]:
    """
    Find semantic matches between chunks from two policies, using precomputed embeddings.

    Compares:
    1. cannot_contain (P1) vs may_contain (P2) -> CONFLICT
    2. may_contain (P1) vs cannot_contain (P2) -> CONFLICT
    3. cannot_contain (P1) vs cannot_contain (P2) -> AGREEMENT
    4. may_contain (P1) vs may_contain (P2) -> AGREEMENT
    """
    matches = []

    if idx1.chunk_embeddings is None or idx2.chunk_embeddings is None:
        return matches

    cosine_scores = _cos_sim_matrix(idx1.chunk_embeddings, idx2.chunk_embeddings)

    for i, chunk1 in enumerate(idx1.chunks):
        for j, chunk2 in enumerate(idx2.chunks):
            similarity = float(cosine_scores[i][j])

            if similarity >= threshold:
                if chunk1.chunk_type == 'cannot_contain' and chunk2.chunk_type == 'may_contain':
                    match_type = 'CONFLICT_P2_ALLOWS'
                elif chunk1.chunk_type == 'may_contain' and chunk2.chunk_type == 'cannot_contain':
                    match_type = 'CONFLICT_P1_ALLOWS'
                elif chunk1.chunk_type == 'cannot_contain' and chunk2.chunk_type == 'cannot_contain':
                    match_type = 'AGREEMENT_BOTH_FORBID'
                elif chunk1.chunk_type == 'may_contain' and chunk2.chunk_type == 'may_contain':
                    match_type = 'AGREEMENT_BOTH_ALLOW'
                else:
                    continue

                matches.append(SimilarityMatch(
                    chunk1=chunk1,
                    chunk2=chunk2,
                    similarity=round(similarity, 3),
                    match_type=match_type
                ))

    return matches


def compare_pair(idx1: PolicyIndex, idx2: PolicyIndex, threshold: float = 0.6) -> dict:
    """Compare a single pair of (precomputed) policies using embeddings."""
    topic_analysis = analyze_topic_similarity(idx1, idx2)
    all_matches = find_semantic_matches(idx1, idx2, threshold)

    conflicts = [m for m in all_matches if 'CONFLICT' in m.match_type]
    agreements_forbid = [m for m in all_matches if m.match_type == 'AGREEMENT_BOTH_FORBID']
    agreements_allow = [m for m in all_matches if m.match_type == 'AGREEMENT_BOTH_ALLOW']

    conflicts.sort(key=lambda x: x.similarity, reverse=True)
    agreements_forbid.sort(key=lambda x: x.similarity, reverse=True)
    agreements_allow.sort(key=lambda x: x.similarity, reverse=True)

    return {
        'policy1': {
            'label': idx1.label,
            'file': idx1.file_path,
            'risk_group': idx1.risk_group,
            'description': idx1.policy_description,
            'num_risks': _num_risks(idx1),
            'num_chunks': len(idx1.chunks)
        },
        'policy2': {
            'label': idx2.label,
            'file': idx2.file_path,
            'risk_group': idx2.risk_group,
            'description': idx2.policy_description,
            'num_risks': _num_risks(idx2),
            'num_chunks': len(idx2.chunks)
        },
        'topic_analysis': topic_analysis,
        'conflicts': conflicts,
        'agreements_forbid': agreements_forbid,
        'agreements_allow': agreements_allow,
        'summary': {
            'topic_relationship': topic_analysis['relationship'],
            'topic_similarity': topic_analysis['similarity'],
            'total_conflicts': len(conflicts),
            'total_agreements_forbid': len(agreements_forbid),
            'total_agreements_allow': len(agreements_allow),
            'threshold_used': threshold
        }
    }


def determine_stance(pair_result: dict) -> tuple[str, str, str]:
    """Derive the (stance, icon, explanation) tuple for a pairwise result."""
    topic = pair_result['topic_analysis']
    total_conflicts = len(pair_result['conflicts'])
    total_agreements = len(pair_result['agreements_forbid']) + len(pair_result['agreements_allow'])

    if topic['relationship'] == 'DIFFERENT_TOPICS':
        return "UNRELATED", "⚪", "Policies regulate different topics - no direct comparison possible"
    elif total_conflicts == 0 and total_agreements > 0:
        return "ALIGNED", "✅", "Policies have compatible regulatory approaches on this topic"
    elif total_conflicts > 0 and total_agreements == 0:
        return "OPPOSING", "⚔️", "Policies have completely opposing regulatory approaches on this topic"
    elif total_conflicts > total_agreements:
        return "MOSTLY_OPPOSING", "🔶", "Policies have more conflicts than agreements - significantly different regulatory approaches"
    elif total_agreements > total_conflicts:
        return "MOSTLY_ALIGNED", "🔷", "Policies have more agreements than conflicts - similar regulatory approaches with some differences"
    else:
        return "MIXED", "⚖️", "Policies have equal conflicts and agreements - mixed regulatory approaches"


def generate_pair_section(pair_result: dict) -> list[str]:
    """Generate the markdown section for a single pair (topic + stance + conflict/agreement detail)."""
    p1 = pair_result['policy1']
    p2 = pair_result['policy2']
    topic = pair_result['topic_analysis']
    conflicts = pair_result['conflicts']
    agreements_forbid = pair_result['agreements_forbid']
    agreements_allow = pair_result['agreements_allow']

    relationship_icon = {
        'SAME_TOPIC': '🔴',
        'RELATED_TOPICS': '🟡',
        'DIFFERENT_TOPICS': '🟢'
    }.get(topic['relationship'], '⚪')

    stance, stance_icon, stance_explanation = determine_stance(pair_result)

    p1_allows_p2_forbids = len([c for c in conflicts if c.match_type == 'CONFLICT_P1_ALLOWS'])
    p1_forbids_p2_allows = len([c for c in conflicts if c.match_type == 'CONFLICT_P2_ALLOWS'])

    lines = [
        f"## {p1['label']} vs {p2['label']}",
        "",
        f"**Files:** `{p1['file']}` vs `{p2['file']}`",
        "",
        f"**Topic Relationship:** {relationship_icon} **{topic['relationship']}** (similarity: {topic['similarity']})",
        "",
        f"*{topic['explanation']}*",
        "",
        f"**Overall Stance:** {stance_icon} **{stance}**",
        "",
        f"*{stance_explanation}*",
        "",
        "| Conflict Type | Count | Meaning |",
        "|---------------|-------|---------|",
        f"| {p1['label']} allows, {p2['label']} forbids | {p1_allows_p2_forbids} | {p1['label']} is more permissive |",
        f"| {p1['label']} forbids, {p2['label']} allows | {p1_forbids_p2_allows} | {p2['label']} is more permissive |",
        f"| Both policies forbid | {len(agreements_forbid)} | Agreement on restrictions |",
        f"| Both policies allow | {len(agreements_allow)} | Agreement on permissions |",
        "",
    ]

    if topic['relationship'] == 'DIFFERENT_TOPICS' and not conflicts:
        lines.extend([
            "**No semantic conflicts detected.** These policies target different topics and can coexist.",
            "",
        ])
        return lines

    if conflicts:
        lines.extend([
            "### 🔴 Semantic Conflicts Detected",
            "",
            "| # | Similarity | Risk ID (P1) | Risk ID (P2) | Conflict Type |",
            "|---|------------|---------------|---------------|---------------|",
        ])
        for i, match in enumerate(conflicts[:20], 1):
            conflict_desc = f"{p1['label']} forbids, {p2['label']} allows" if match.match_type == 'CONFLICT_P2_ALLOWS' else f"{p1['label']} allows, {p2['label']} forbids"
            lines.append(
                f"| {i} | {match.similarity} | {match.chunk1.risk_id} | {match.chunk2.risk_id} | {conflict_desc} |"
            )

        lines.extend(["", "#### Conflict Details", ""])

        for i, match in enumerate(conflicts[:10], 1):
            lines.extend([f"##### Conflict {i} (Similarity: {match.similarity})", ""])
            if match.match_type == 'CONFLICT_P2_ALLOWS':
                lines.extend([
                    f"**{p1['label']} FORBIDS** (`{match.chunk1.risk_name}` - {match.chunk1.risk_id}):",
                    f"> {match.chunk1.text}",
                    "",
                    f"**{p2['label']} ALLOWS** (`{match.chunk2.risk_name}` - {match.chunk2.risk_id}):",
                    f"> {match.chunk2.text}",
                ])
            else:
                lines.extend([
                    f"**{p1['label']} ALLOWS** (`{match.chunk1.risk_name}` - {match.chunk1.risk_id}):",
                    f"> {match.chunk1.text}",
                    "",
                    f"**{p2['label']} FORBIDS** (`{match.chunk2.risk_name}` - {match.chunk2.risk_id}):",
                    f"> {match.chunk2.text}",
                ])
            lines.extend(["", "---", ""])

    if agreements_forbid:
        lines.extend([
            "### ✅ Agreements (Both Policies Forbid)",
            "",
            "| # | Similarity | Text (P1) | Text (P2) |",
            "|---|------------|-----------|-----------|",
        ])
        for i, match in enumerate(agreements_forbid[:10], 1):
            p1_text = match.chunk1.text[:50] + "..." if len(match.chunk1.text) > 50 else match.chunk1.text
            p2_text = match.chunk2.text[:50] + "..." if len(match.chunk2.text) > 50 else match.chunk2.text
            lines.append(f"| {i} | {match.similarity} | {p1_text} | {p2_text} |")
        lines.append("")

    if agreements_allow:
        lines.extend([
            "### ✅ Agreements (Both Policies Allow)",
            "",
            "| # | Similarity | Text (P1) | Text (P2) |",
            "|---|------------|-----------|-----------|",
        ])
        for i, match in enumerate(agreements_allow[:10], 1):
            p1_text = match.chunk1.text[:50] + "..." if len(match.chunk1.text) > 50 else match.chunk1.text
            p2_text = match.chunk2.text[:50] + "..." if len(match.chunk2.text) > 50 else match.chunk2.text
            lines.append(f"| {i} | {match.similarity} | {p1_text} | {p2_text} |")
        lines.append("")

    return lines


def generate_report(indices: list[PolicyIndex], pair_results: list[dict], threshold: float) -> str:
    """Generate a markdown report covering every policy and every pairwise comparison."""
    lines = [
        "# Policy Comparison Report (Embedding-based Semantic Analysis)",
        "",
        f"Comparing **{len(indices)}** policies across **{len(pair_results)}** pairs "
        f"(similarity threshold: {threshold}).",
        "",
        "## Policies Compared",
        "",
        "| Label | File | Risk Group | Number of Risks | Policy Chunks |",
        "|-------|------|------------|------------------|---------------|",
    ]
    for idx in indices:
        lines.append(
            f"| {idx.label} | `{Path(idx.file_path).name}` | {idx.risk_group} | "
            f"{_num_risks(idx)} | {len(idx.chunks)} |"
        )
    lines.append("")

    # Pairwise topic-similarity and conflict-count matrices give an at-a-glance
    # overview before diving into the per-pair detail sections below.
    labels = [idx.label for idx in indices]
    sim_lookup = {(r['policy1']['label'], r['policy2']['label']): r['topic_analysis']['similarity'] for r in pair_results}
    conflict_lookup = {(r['policy1']['label'], r['policy2']['label']): len(r['conflicts']) for r in pair_results}

    lines.extend([
        "## Topic Similarity Matrix",
        "",
        "| | " + " | ".join(labels) + " |",
        "|---" * (len(labels) + 1) + "|",
    ])
    for row_label in labels:
        cells = []
        for col_label in labels:
            if row_label == col_label:
                cells.append("—")
            else:
                key = (row_label, col_label) if (row_label, col_label) in sim_lookup else (col_label, row_label)
                cells.append(f"{sim_lookup.get(key, 'n/a')}")
        lines.append(f"| **{row_label}** | " + " | ".join(cells) + " |")
    lines.append("")

    lines.extend([
        "## Conflict Count Matrix",
        "",
        "| | " + " | ".join(labels) + " |",
        "|---" * (len(labels) + 1) + "|",
    ])
    for row_label in labels:
        cells = []
        for col_label in labels:
            if row_label == col_label:
                cells.append("—")
            else:
                key = (row_label, col_label) if (row_label, col_label) in conflict_lookup else (col_label, row_label)
                cells.append(f"{conflict_lookup.get(key, 'n/a')}")
        lines.append(f"| **{row_label}** | " + " | ".join(cells) + " |")
    lines.append("")

    lines.extend(["---", ""])

    for pair_result in pair_results:
        lines.extend(generate_pair_section(pair_result))
        lines.extend(["---", ""])

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Compare two or more already-indexed policies using their precomputed Milvus embeddings.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('policies', nargs='*',
                        help='Policy YAML files/directories whose labels to compare '
                             '(must already be indexed via index_policies.py). Omit if using --place.')
    parser.add_argument('--place', help='Compare the policies registered for this deployment location '
                                        'in --locations-config, instead of passing files directly')
    parser.add_argument('--locations-config', default='policy_locations.yaml',
                        help='Path to the place -> policy_labels mapping file (default: policy_locations.yaml)')
    parser.add_argument('--milvus-db', default='policy_embeddings.db',
                        help='Path to the local Milvus Lite database file (default: policy_embeddings.db)')
    parser.add_argument('--collection', default='policy_chunks',
                        help='Milvus collection name to read chunk embeddings from (default: policy_chunks)')
    parser.add_argument('--output', '-o', default='policy_comparison_report',
                        help='Base path (without extension) for the saved .md report '
                             '(default: policy_comparison_report)')
    parser.add_argument('--threshold', '-t', type=float, default=0.6,
                        help='Similarity threshold for matching (default: 0.6)')
    parser.add_argument('--json', action='store_true', help='Also save raw JSON pairwise results')

    args = parser.parse_args()

    if not MILVUS_AVAILABLE:
        print("Error: pymilvus is required for this script.", file=sys.stderr)
        print("Install with: pip install pymilvus", file=sys.stderr)
        sys.exit(1)

    if args.place and args.policies:
        print("Error: pass either --place or explicit policy paths, not both.", file=sys.stderr)
        sys.exit(1)
    if not args.place and not args.policies:
        print("Error: specify --place <name> or two or more policy YAML files/directories.", file=sys.stderr)
        sys.exit(1)

    if args.place:
        try:
            location_map = load_location_config(args.locations_config)
        except OSError as e:
            print(f"Error loading locations config '{args.locations_config}': {e}", file=sys.stderr)
            sys.exit(1)
        labels = location_map.get(args.place)
        if labels is None:
            known = ", ".join(sorted(location_map)) or "(none configured)"
            print(f"Error: unknown place '{args.place}'. Known places: {known}", file=sys.stderr)
            sys.exit(1)
    else:
        for path in args.policies:
            if not Path(path).exists():
                print(f"Error: Path not found: {path}", file=sys.stderr)
                sys.exit(1)
        policy_paths = _resolve_policy_paths(args.policies)
        seen_labels: dict = {}
        labels = [_label_for(str(path), seen_labels) for path in policy_paths]

    if len(labels) < 2:
        print(f"Error: at least 2 policies are required (found {len(labels)}).", file=sys.stderr)
        sys.exit(1)

    client = MilvusClient(uri=args.milvus_db)
    if not client.has_collection(args.collection):
        print(f"Error: collection '{args.collection}' not found in {args.milvus_db!r}. "
              f"Run index_policies.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching precomputed embeddings for {len(labels)} policies: {', '.join(labels)}...", file=sys.stderr)
    rows_by_label = fetch_policy_rows(client, args.collection, labels)
    client.close()

    missing = [label for label, rows in rows_by_label.items() if not rows]
    if missing:
        print(f"Error: these policy_labels aren't indexed yet: {', '.join(missing)}. "
              f"Run index_policies.py on them first.", file=sys.stderr)
        sys.exit(1)

    indices = [build_policy_index_from_rows(label, rows_by_label[label]) for label in labels]

    print(f"Comparing all {len(indices) * (len(indices) - 1) // 2} pairs...", file=sys.stderr)
    pair_results = [
        compare_pair(idx1, idx2, args.threshold)
        for idx1, idx2 in itertools.combinations(indices, 2)
    ]

    report_text = generate_report(indices, pair_results, args.threshold)
    print(report_text)

    base_output = Path(args.output)
    md_path = base_output.with_suffix('.md')
    md_path.write_text(report_text)
    print(f"Report saved to: {md_path}", file=sys.stderr)

    if args.json:
        import json

        def serialize(obj):
            if isinstance(obj, (PolicyChunk, SimilarityMatch)):
                return obj.__dict__
            return obj

        json_path = base_output.with_suffix('.json')
        json_path.write_text(json.dumps({'pairs': pair_results}, indent=2, default=serialize))
        print(f"JSON results saved to: {json_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
