"""
src/utils.py
============
Shared utilities for KG-Flan:
  - Knowledge graph loading (DBpedia, MetaQA KB)
  - Relation candidate retrieval
  - Embedding helpers
  - Evaluation metrics
  - Result serialisation
"""

import json
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────────────────────────────
# 1.  Knowledge graph loaders
# ──────────────────────────────────────────────────────────────────────

def load_dbpedia(pickle_path: str) -> dict:
    """
    Load the DBpedia undirected KG from a pickle file.

    Returns
    -------
    dict  {entity: {relation: [object, ...]}}
    """
    print(f"[DBpedia] Loading from '{pickle_path}'…")
    with open(pickle_path, "rb") as f:
        kg = pickle.load(f)
    print(f"[DBpedia] Loaded {len(kg):,} entities.")
    return kg


def load_metaqa_kb(kb_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Load MetaQA knowledge base (pipe-separated triples).

    Format per line: subject|relation|object
    Returns a bidirectional dict (forward + inverse ~ relations).
    """
    kb: Dict[str, Dict[str, List[str]]] = {}
    with open(kb_path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            s, r, o = parts
            # Forward
            kb.setdefault(s, {}).setdefault(r, []).append(o)
            # Inverse
            kb.setdefault(o, {}).setdefault(f"~{r}", []).append(s)
    total_entities = len(kb)
    total_triples = sum(
        sum(len(v) for v in rels.items())
        for rels in kb.values()
    )
    print(f"[MetaQA KB] {total_entities:,} entities loaded from '{kb_path}'")
    return kb


def get_all_relations(dbpedia: dict, include_inverse: bool = False) -> List[str]:
    """Return all unique relation names in a DBpedia/KB dict."""
    rels: Set[str] = set()
    for entity_rels in dbpedia.values():
        for rel in entity_rels:
            if include_inverse or not rel.startswith("~"):
                rels.add(rel)
    return sorted(rels)


# ──────────────────────────────────────────────────────────────────────
# 2.  Relation candidate retrieval
# ──────────────────────────────────────────────────────────────────────

def get_relation_candidates(entity_set: List[str], kg: dict,
                             fallback_relations: Optional[List[str]] = None) -> List[str]:
    """
    Return all forward relations reachable from the given entity set.

    Falls back to ``fallback_relations`` (or all KG relations) if no
    entity in ``entity_set`` is found in the KG.
    """
    candidates: Set[str] = set()
    for entity in entity_set:
        if entity in kg:
            for rel in kg[entity]:
                if not rel.startswith("~"):
                    candidates.add(rel)

    if not candidates:
        if fallback_relations is not None:
            candidates = set(fallback_relations)
        else:
            candidates = set(get_all_relations(kg))

    return list(candidates)


def build_evidence_graph(entity_set: List[str], selected_rels: List[str],
                          kg: dict, max_triples: int = 15) -> List[Tuple[str, str, str]]:
    """
    Retrieve (subject, relation, object) triples for the selected relations.
    """
    triples: List[Tuple[str, str, str]] = []
    sel_set = set(selected_rels)
    for entity in entity_set:
        if entity not in kg:
            continue
        for rel, objects in kg[entity].items():
            if rel in sel_set:
                objs = objects if isinstance(objects, list) else [objects]
                for obj in objs:
                    triples.append((entity, rel, obj))
                    if len(triples) >= max_triples:
                        return triples
    return triples


def triples_to_string(triples: List[Tuple[str, str, str]]) -> str:
    """Serialise a list of triples to a readable multi-line string."""
    if not triples:
        return "No evidence found."
    return "\n".join(f"({s}, {r}, {o})" for s, r, o in triples)


# ──────────────────────────────────────────────────────────────────────
# 3.  Embedding utilities
# ──────────────────────────────────────────────────────────────────────

def encode_pairs(claims: List[str], relations: List[str],
                 encoder: SentenceTransformer, batch_size: int = 256,
                 show_progress: bool = False) -> np.ndarray:
    """
    Encode claim–relation pairs as  "[claim] [SEP] [relation]"  embeddings.

    Returns
    -------
    np.ndarray  shape (N, embedding_dim)
    """
    texts = [f"{c} [SEP] {r}" for c, r in zip(claims, relations)]
    return encoder.encode(texts, batch_size=batch_size,
                          show_progress_bar=show_progress,
                          convert_to_numpy=True)


def score_candidates(claim: str, candidates: List[str],
                     scorer, encoder: SentenceTransformer,
                     device: str = "cpu") -> List[Tuple[str, float]]:
    """
    Score (claim, relation) pairs with MLPRelationScorer.

    Returns
    -------
    List of (relation, score) sorted descending by score.
    """
    if not candidates:
        return []
    texts = [f"{claim} [SEP] {rel}" for rel in candidates]
    embs = encoder.encode(texts, batch_size=256, show_progress_bar=False,
                          convert_to_numpy=True)
    emb_tensor = torch.tensor(embs, dtype=torch.float32).to(device)
    with torch.no_grad():
        scores = scorer.score(emb_tensor).cpu().numpy()
    return sorted(zip(candidates, scores.tolist()), key=lambda x: -x[1])


# ──────────────────────────────────────────────────────────────────────
# 4.  MetaQA-specific helpers
# ──────────────────────────────────────────────────────────────────────

METAQA_RELATIONS = [
    "directed_by", "written_by", "starred_actors", "release_year",
    "in_language", "has_genre", "has_imdb_rating", "has_imdb_votes", "has_tags",
]

_KW_MAP = [
    (["direct", "director"], "directed_by"),
    (["writ", "author", "script"], "written_by"),
    (["star", "act", "cast", "appear"], "starred_actors"),
    (["year", "when", "releas", "came out"], "release_year"),
    (["language", "spoken", "tongue"], "in_language"),
    (["genre", "type", "kind of film"], "has_genre"),
    (["imdb rating", "rating", "score", "rated"], "has_imdb_rating"),
    (["votes", "how many people"], "has_imdb_votes"),
    (["tag", "keyword", "about"], "has_tags"),
]


def infer_metaqa_relation(question: str) -> str:
    """Heuristic: map question text to the most likely MetaQA relation."""
    q = question.lower()
    for keywords, rel in _KW_MAP:
        if any(k in q for k in keywords):
            return rel
    return "starred_actors"


def extract_metaqa_entities(question: str) -> List[str]:
    """Extract bracket-enclosed entities from a MetaQA question."""
    return re.findall(r"\[([^\]]+)\]", question)


def traverse_kb_nhop(entity_set: List[str], kb: dict,
                      target_rel: str, n_hops: int) -> List[str]:
    """
    Perform n-hop traversal on a KB starting from ``entity_set``,
    returning the final set of answer entities (values of ``target_rel``
    at the last hop).
    """
    frontier = set(entity_set)
    for hop in range(n_hops):
        next_frontier: Set[str] = set()
        for ent in frontier:
            if ent not in kb:
                continue
            for rel, objs in kb[ent].items():
                if hop < n_hops - 1:
                    # Intermediate hop: expand all relations
                    next_frontier.update(objs)
                else:
                    # Final hop: only target relation
                    if rel == target_rel:
                        next_frontier.update(objs)
        frontier = next_frontier
    return list(frontier)


# ──────────────────────────────────────────────────────────────────────
# 5.  Evaluation metrics
# ──────────────────────────────────────────────────────────────────────

def compute_factkg_metrics(preds: List[bool], labels: List[bool]) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1 for FactKG."""
    n = len(preds)
    assert n == len(labels) and n > 0

    tp = sum(1 for p, l in zip(preds, labels) if p and l)
    tn = sum(1 for p, l in zip(preds, labels) if not p and not l)
    fp = sum(1 for p, l in zip(preds, labels) if p and not l)
    fn = sum(1 for p, l in zip(preds, labels) if not p and l)

    accuracy  = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "accuracy":  round(accuracy * 100, 2),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def hits_at_1(pred_answers: List[str], gold_answers: List[str]) -> bool:
    """True if the top predicted answer matches any gold answer (case-insensitive)."""
    if not pred_answers:
        return False
    gold_lower = {g.lower() for g in gold_answers}
    return pred_answers[0].lower() in gold_lower


def compute_metaqa_hits(results: List[Dict]) -> float:
    """
    Compute Hits@1 from a list of {'pred': [...], 'gold': [...]} dicts.
    """
    correct = sum(hits_at_1(r["pred"], r["gold"]) for r in results)
    return correct / len(results) * 100


# ──────────────────────────────────────────────────────────────────────
# 6.  Result I/O
# ──────────────────────────────────────────────────────────────────────

def save_metrics(metrics: dict, path: str):
    """Save a metrics dict as a JSON file (creates parent dirs)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Saved] {path}")


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_factkg_table(metrics: dict):
    print(f"\n{'='*55}")
    print(f"{'FactKG Results':^55}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {metrics['accuracy']:.2f}%")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"{'='*55}")


def print_metaqa_table(hop_metrics: Dict[int, float]):
    print(f"\n{'='*45}")
    print(f"{'MetaQA Hits@1':^45}")
    print(f"{'='*45}")
    values = []
    for hop in sorted(hop_metrics):
        h = hop_metrics[hop]
        values.append(h)
        print(f"  {hop}-hop : {h:.2f}%")
    if values:
        print(f"  {'─'*30}")
        print(f"  Average : {sum(values)/len(values):.2f}%")
    print(f"{'='*45}")
