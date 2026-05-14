"""
src/dataset.py
==============
Dataset classes and data-loading utilities for KG-Flan.

Covers
------
  - RelationPairDataset  : (claim, relation) pairs for MLP scorer training
  - FactKGDataset        : FactKG fact-verification samples
  - MetaQADataset        : MetaQA multi-hop QA samples

Training pair construction
--------------------------
  Positive: claim paired with its ground-truth relation(s).
  Negative: claim paired with randomly sampled non-ground-truth relations.
  Ratio    : neg_pos_ratio = 2  (i.e. 2 negatives per positive).
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────────────────────────────
# 1.  RelationPairDataset
# ──────────────────────────────────────────────────────────────────────

class RelationPairDataset(Dataset):
    """
    Dataset of pre-embedded (claim [SEP] relation) pairs with binary labels.

    Parameters
    ----------
    embeddings  : np.ndarray  shape (N, 384)
    labels      : np.ndarray  shape (N,)  — 1 = relevant, 0 = irrelevant
    """

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        assert len(embeddings) == len(labels), "Embeddings and labels must have equal length."
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]

    @classmethod
    def from_pairs(
        cls,
        pairs: List[Tuple[str, str, int]],
        encoder: SentenceTransformer,
        batch_size: int = 512,
        show_progress: bool = True,
    ) -> "RelationPairDataset":
        """
        Build dataset by encoding (claim, relation, label) triples.

        pairs : list of (claim_text, relation_name, label)
        """
        texts = [f"{claim} [SEP] {relation}" for claim, relation, _ in pairs]
        labels = np.array([label for _, _, label in pairs], dtype=np.float32)

        print(f"[Dataset] Encoding {len(texts):,} pairs …")
        embeddings = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return cls(embeddings, labels)


# ──────────────────────────────────────────────────────────────────────
# 2.  FactKG helpers
# ──────────────────────────────────────────────────────────────────────

class FactKGDataset(Dataset):
    """
    Loads FactKG JSON and exposes (claim, entity_set, label) items.

    Expected JSON format (per item):
        {
            "claim": "...",
            "label": "SUPPORTS" | "REFUTES",
            "entity_set": ["Entity_A", "Entity_B"],
            "relations": ["relation_1", ...]   # ground-truth relations
        }
    """

    LABEL_MAP = {"SUPPORTS": 1, "REFUTES": 0, "True": 1, "False": 0,
                 True: 1, False: 0, 1: 1, 0: 0}

    def __init__(self, json_path: str, n_samples: Optional[int] = None, seed: int = 42):
        with open(json_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            # Some FactKG releases use a dict with "data" key
            data = data.get("data", list(data.values()))

        if n_samples is not None:
            rng = random.Random(seed)
            data = rng.sample(data, min(n_samples, len(data)))

        self.samples = data
        print(f"[FactKGDataset] Loaded {len(self.samples):,} samples from '{json_path}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        return {
            "claim": item["claim"],
            "entity_set": item.get("entity_set", []),
            "relations": item.get("relations", []),
            "label": self.LABEL_MAP.get(item.get("label", item.get("Label", 0)), 0),
        }

    def build_training_pairs(
        self,
        all_relations: List[str],
        neg_pos_ratio: int = 2,
        seed: int = 42,
    ) -> List[Tuple[str, str, int]]:
        """
        Construct (claim, relation, label) pairs for scorer training.
        Positive pairs: ground-truth relations. Negative: random samples.
        """
        rng = random.Random(seed)
        pairs: List[Tuple[str, str, int]] = []
        other_relations = list(all_relations)

        for item in self.samples:
            claim = item["claim"]
            pos_rels = item.get("relations", [])

            # Positive pairs
            for rel in pos_rels:
                pairs.append((claim, rel, 1))

            # Negative pairs
            neg_pool = [r for r in other_relations if r not in pos_rels]
            n_neg = neg_pos_ratio * max(len(pos_rels), 1)
            negs = rng.sample(neg_pool, min(n_neg, len(neg_pool)))
            for rel in negs:
                pairs.append((claim, rel, 0))

        rng.shuffle(pairs)
        pos_count = sum(1 for _, _, l in pairs if l == 1)
        neg_count = len(pairs) - pos_count
        print(f"[FactKGDataset] Training pairs — pos: {pos_count:,}, neg: {neg_count:,}")
        return pairs


# ──────────────────────────────────────────────────────────────────────
# 3.  MetaQA helpers
# ──────────────────────────────────────────────────────────────────────

METAQA_RELATIONS = [
    "directed_by", "written_by", "starred_actors", "release_year",
    "in_language", "has_genre", "has_imdb_rating", "has_imdb_votes", "has_tags",
]


class MetaQADataset(Dataset):
    """
    Loads MetaQA QA file (plain text, one question per line).

    Line format: question\tanswer1|answer2
    Entity mentions are enclosed in square brackets: [Inception]
    """

    def __init__(self, qa_path: str, n_samples: Optional[int] = None, seed: int = 42):
        samples = []
        with open(qa_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                question = parts[0]
                answers = parts[1].split("|")
                samples.append({"question": question, "answers": answers})

        if n_samples is not None:
            rng = random.Random(seed)
            samples = rng.sample(samples, min(n_samples, len(samples)))

        self.samples = samples
        print(f"[MetaQADataset] Loaded {len(self.samples):,} samples from '{qa_path}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

    def build_training_pairs(
        self,
        neg_pos_ratio: int = 2,
        seed: int = 42,
    ) -> List[Tuple[str, str, int]]:
        """Build (question, relation, label) pairs using MetaQA's 9 relations."""
        import re
        rng = random.Random(seed)
        pairs: List[Tuple[str, str, int]] = []

        for item in self.samples:
            question = item["question"]
            # Infer positive relation from question keywords (heuristic)
            pos_rels = _infer_metaqa_relations(question)
            neg_rels = [r for r in METAQA_RELATIONS if r not in pos_rels]

            for rel in pos_rels:
                pairs.append((question, rel, 1))

            n_neg = neg_pos_ratio * max(len(pos_rels), 1)
            negs = rng.sample(neg_rels, min(n_neg, len(neg_rels)))
            for rel in negs:
                pairs.append((question, rel, 0))

        rng.shuffle(pairs)
        return pairs


def _infer_metaqa_relations(question: str) -> List[str]:
    """Heuristic mapping of question keywords → MetaQA relation names."""
    q = question.lower()
    mapping = [
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
    found = []
    for keywords, rel in mapping:
        if any(k in q for k in keywords):
            found.append(rel)
    return found if found else ["starred_actors"]   # safe default


# ──────────────────────────────────────────────────────────────────────
# 4.  DataLoader factory
# ──────────────────────────────────────────────────────────────────────

def make_dataloaders(
    dataset: RelationPairDataset,
    val_split: float = 0.15,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Split dataset into train / val and return DataLoaders."""
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[DataLoader] train={n_train:,} | val={n_val:,} | batch_size={batch_size}")
    return train_loader, val_loader
