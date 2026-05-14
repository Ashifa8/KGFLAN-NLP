"""
train.py
========
Train the MLP Relation Scorer for KG-Flan.

Usage
-----
  python train.py --dataset factkg
  python train.py --dataset metaqa
  python train.py --dataset factkg --epochs 15 --lr 5e-4
"""

import argparse
import json
import os
import time
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset

from src.model import MLPRelationScorer
from src.dataset import RelationPairDataset, FactKGDataset, MetaQADataset, make_dataloaders


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────
# FactKG data loading (pickle format from Kaggle dataset)
# ──────────────────────────────────────────────────────────────────────

def load_factkg_pickle(factkg_path: str, dbpedia_path: str):
    """Load FactKG train split and DBpedia from pickle files."""
    print("[FactKG] Loading training data...")
    with open(os.path.join(factkg_path, "factkg_train.pickle"), "rb") as f:
        train_data = pickle.load(f)

    print("[FactKG] Loading DBpedia KG...")
    with open(os.path.join(dbpedia_path, "dbpedia_2015_undirected_light.pickle"), "rb") as f:
        dbpedia = pickle.load(f)

    print(f"[FactKG] Train claims: {len(train_data):,} | DBpedia entities: {len(dbpedia):,}")
    return train_data, dbpedia


def build_factkg_pairs(train_data: dict, dbpedia: dict,
                       max_claims: int = 5000, neg_ratio: int = 2, seed: int = 42):
    """Build (claim, relation, label) pairs from FactKG pickle data."""
    rng = np.random.default_rng(seed)
    claims = list(train_data.keys())
    rng.shuffle(claims)
    claims = claims[:max_claims]

    # Collect all unique forward relations
    all_relations = list({
        rel for entity in dbpedia for rel in dbpedia[entity]
        if not rel.startswith("~")
    })

    pairs = []
    skipped = 0

    for claim in claims:
        info = train_data[claim]
        entity_set = info.get("Entity_set", [])
        evidence = info.get("Evidence", {})

        # Positive relations from evidence
        pos_rels = set()
        for rel_list in evidence.values():
            for group in rel_list:
                for rel in group:
                    pos_rels.add(rel.lstrip("~"))

        if not pos_rels:
            skipped += 1
            continue

        # Candidate relations for this entity set
        candidates = [
            rel for entity in entity_set if entity in dbpedia
            for rel in dbpedia[entity] if not rel.startswith("~")
        ] or all_relations

        for rel in pos_rels:
            if rel in candidates:
                pairs.append((claim, rel, 1))

        neg_pool = [r for r in candidates if r not in pos_rels]
        n_neg = min(len(pos_rels) * neg_ratio, len(neg_pool))
        if n_neg > 0:
            negs = rng.choice(neg_pool, n_neg, replace=False)
            for rel in negs:
                pairs.append((claim, rel, 0))

    rng.shuffle(pairs)
    pos = sum(1 for _, _, l in pairs if l == 1)
    print(f"[FactKG] Built {len(pairs):,} pairs (pos={pos:,}, neg={len(pairs)-pos:,}) | skipped={skipped}")
    return pairs


# ──────────────────────────────────────────────────────────────────────
# MetaQA data loading
# ──────────────────────────────────────────────────────────────────────

METAQA_RELATIONS = [
    "directed_by", "written_by", "starred_actors", "release_year",
    "in_language", "has_genre", "has_imdb_rating", "has_imdb_votes", "has_tags",
]


def load_metaqa_kb(kb_path: str) -> dict:
    """Load MetaQA knowledge base into a forward dict."""
    kb = {}
    with open(kb_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            subj, rel, obj = parts
            kb.setdefault(subj, {}).setdefault(rel, []).append(obj)
    return kb


def build_metaqa_pairs(qa_paths: list, neg_ratio: int = 2, seed: int = 42) -> list:
    """Build (question, relation, label) pairs from MetaQA QA files."""
    from src.dataset import _infer_metaqa_relations
    rng = random.Random(seed)
    pairs = []

    for qa_path in qa_paths:
        if not os.path.exists(qa_path):
            print(f"[MetaQA] Skipping missing file: {qa_path}")
            continue
        dataset = MetaQADataset(qa_path)
        pairs.extend(dataset.build_training_pairs(neg_ratio, seed))

    rng.shuffle(pairs)
    pos = sum(1 for _, _, l in pairs if l == 1)
    print(f"[MetaQA] Built {len(pairs):,} pairs (pos={pos:,}, neg={len(pairs)-pos:,})")
    return pairs


# ──────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []

    for emb, labels in loader:
        emb, labels = emb.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(emb).squeeze(1)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds_all.extend((out > 0.5).float().cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    return total_loss / len(loader), acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []

    for emb, labels in loader:
        emb, labels = emb.to(device), labels.to(device)
        out = model(emb).squeeze(1)
        loss = criterion(out, labels)

        total_loss += loss.item()
        preds_all.extend((out > 0.5).float().cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    return total_loss / len(loader), acc


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train KG-Flan MLP Relation Scorer")
    parser.add_argument("--dataset", choices=["factkg", "metaqa"], required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    # Override config from CLI
    epochs = args.epochs or cfg["training"]["epochs"]
    lr = args.lr or cfg["training"]["learning_rate"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    device = torch.device(cfg["model"]["device"] if torch.cuda.is_available() else "cpu")

    os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["logging"]["results_dir"], exist_ok=True)

    print(f"\n{'='*60}")
    print(f"KG-Flan — Training Relation Scorer ({args.dataset.upper()})")
    print(f"Device: {device} | Epochs: {epochs} | LR: {lr} | Batch: {batch_size}")
    print(f"{'='*60}\n")

    # ── Build training pairs ──────────────────────────────────────────
    if args.dataset == "factkg":
        train_data, dbpedia = load_factkg_pickle(
            cfg["factkg"]["train_path"].rsplit("/", 1)[0],
            cfg["factkg"]["dbpedia_path"].rsplit("/", 1)[0],
        )
        pairs = build_factkg_pairs(
            train_data, dbpedia,
            max_claims=5000,
            neg_ratio=cfg["training"]["neg_pos_ratio"],
            seed=args.seed,
        )
        ckpt_path = cfg["factkg"]["scorer_ckpt"]

    else:  # metaqa
        qa_paths = [
            cfg["metaqa"]["qa_1hop_test"].replace("test", "train"),
            cfg["metaqa"]["qa_2hop_test"].replace("test", "train"),
        ]
        pairs = build_metaqa_pairs(
            qa_paths,
            neg_ratio=cfg["training"]["neg_pos_ratio"],
            seed=args.seed,
        )
        ckpt_path = cfg["metaqa"]["scorer_ckpt"]

    # ── Encode pairs ──────────────────────────────────────────────────
    encoder_name = cfg["model"]["sentence_encoder"]
    print(f"\n[Encoder] Loading '{encoder_name}'...")
    encoder = SentenceTransformer(encoder_name)

    dataset = RelationPairDataset.from_pairs(pairs, encoder, batch_size=512)
    train_loader, val_loader = make_dataloaders(
        dataset,
        val_split=cfg["training"]["val_split"],
        batch_size=batch_size,
        seed=args.seed,
    )

    # ── Model ─────────────────────────────────────────────────────────
    scorer_cfg = cfg["scorer"]
    model = MLPRelationScorer(
        input_dim=scorer_cfg["input_dim"],
        hidden_dims=scorer_cfg["hidden_dims"],
        dropout=scorer_cfg["dropout"],
    ).to(device)
    print(f"\n[Model] Parameters: {model.count_parameters():,}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Training ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state = None
    log_rows = []

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc:>9.4f} | {va_loss:>8.4f} | {va_acc:>7.4f}  ({elapsed:.1f}s)")

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 5),
            "train_acc": round(tr_acc, 5),
            "val_loss": round(va_loss, 5),
            "val_acc": round(va_acc, 5),
        })

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Save checkpoint ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "input_dim": scorer_cfg["input_dim"],
        "hidden_dims": scorer_cfg["hidden_dims"],
        "dropout": scorer_cfg["dropout"],
        "dataset": args.dataset,
        "best_val_loss": best_val_loss,
    }, ckpt_path)
    print(f"\n[Checkpoint] Saved → {ckpt_path}")

    # ── Save training log ─────────────────────────────────────────────
    import csv
    log_path = os.path.join(cfg["logging"]["results_dir"], f"training_log_{args.dataset}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[Log] Training log → {log_path}")

    best_val_acc = max(r["val_acc"] for r in log_rows)
    print(f"\n[Done] Best Val Acc: {best_val_acc*100:.2f}%  |  Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
