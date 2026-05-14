"""
inference.py
============
Run the full KG-Flan pipeline on FactKG or MetaQA.

Usage
-----
  python inference.py --dataset factkg --split test --n_samples 500
  python inference.py --dataset metaqa --hops 1 --n_samples 500
  python inference.py --dataset metaqa --hops 2 --n_samples 500
  python inference.py --dataset metaqa --hops 3 --n_samples 500
"""

import argparse
import gc
import json
import os
import pickle
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.model import load_scorer


# ──────────────────────────────────────────────────────────────────────
# Config / helpers
# ──────────────────────────────────────────────────────────────────────

def load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_flan(model_name: str, dtype: str, device: str):
    print(f"[LLM] Loading {model_name} ({dtype})…")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto"
    )
    model.eval()
    print(f"[LLM] Loaded.")
    return tokenizer, model


def flan_generate(prompt: str, tokenizer, model, device, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512,
                       truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             num_beams=2, early_stopping=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — Segmentation
# ──────────────────────────────────────────────────────────────────────

SEG_PROMPT = """\
Please divide the given sentence into several sentences each of which can be represented by one triplet.
The generated sentences should be numbered and formatted as follows: #(number). (sentence), (entity set).
The entity set for each sentence should contain no more than two entities.

Examples)
Sentence: Ahmad Kadhim Assad's club is Al-Zawra'a SC.
Entity set: ['Ahmad_Kadhim_Assad', 'Al-Zawra_SC']
Divided:
1. Ahmad Kadhim Assad's club is Al-Zawra'a SC., Entity set: ['Ahmad_Kadhim_Assad', 'Al-Zawra_SC']

Sentence: William Anders is an astronaut whose crew member is Frank Borman who is also an astronaut.
Entity set: ['William_Anders', 'Frank_Borman']
Divided:
1. William Anders is an astronaut., Entity set: ['William_Anders', 'astronaut']
2. William Anders crew member is Frank Borman., Entity set: ['William_Anders', 'Frank_Borman']
3. Frank Borman is an astronaut., Entity set: ['Frank_Borman', 'astronaut']

Your Task)
Sentence: {claim}
Entity set: {entity_set}
Divided:"""


def segment_claim(claim: str, entity_set: list, tokenizer, model, device) -> list:
    prompt = SEG_PROMPT.format(claim=claim, entity_set=entity_set)
    result = flan_generate(prompt, tokenizer, model, device, max_new_tokens=200)
    sub_sentences = []
    for line in result.strip().split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("#")):
            if "Entity set:" in line:
                sent_part, ent_part = line.split("Entity set:", 1)
                sent_part = sent_part.lstrip("0123456789.#) ").strip()
                sub_sentences.append({"sentence": sent_part, "entities": ent_part.strip()})
    return sub_sentences or [{"sentence": claim, "entities": str(entity_set)}]


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — Relation Selection (MLP Scorer)
# ──────────────────────────────────────────────────────────────────────

REL_PROMPT = """\
Find the top {k} elements from Words set which are most semantically related to the given sentence.

Examples)
Sentence: Ahmad Kadhim Assad's club is Al-Zawra'a SC.
Words set: ['club', 'clubs', 'parent', 'spouse', 'birthPlace']
Top 2 Answer: ['club', 'clubs']

Now find the top {k} elements.
Sentence: {sentence}
Words set: {rels}
Top {k} Answer:"""


def select_relations(claim: str, sub_sentences: list, entity_set: list,
                     dbpedia: dict, scorer, encoder, tokenizer, flan_model,
                     device, top_k: int = 5) -> list:
    # Gather candidates from entity set
    candidates = list({
        rel for ent in entity_set if ent in dbpedia
        for rel in dbpedia[ent] if not rel.startswith("~")
    })
    if not candidates:
        return []

    # Score all candidates
    combined = [f"{claim} [SEP] {rel}" for rel in candidates]
    embs = encoder.encode(combined, batch_size=256, show_progress_bar=False,
                          convert_to_numpy=True)
    emb_tensor = torch.tensor(embs, dtype=torch.float32).to(device)
    scores = scorer.score(emb_tensor).cpu().numpy()

    # Take top 20 for Flan re-ranking
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    top20 = [r for r, _ in ranked[:20]]

    selected = set()
    for sub in sub_sentences:
        prompt = REL_PROMPT.format(k=top_k, sentence=sub["sentence"], rels=top20)
        result = flan_generate(prompt, tokenizer, flan_model, device, max_new_tokens=60)
        for rel in top20:
            if rel in result:
                selected.add(rel)
    # Fallback
    if not selected:
        selected = {r for r, _ in ranked[:top_k]}

    return list(selected)


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — Evidence Graph Construction
# ──────────────────────────────────────────────────────────────────────

def build_evidence_graph(entity_set: list, selected_rels: list, dbpedia: dict,
                         max_triples: int = 10) -> list:
    triples = []
    for entity in entity_set:
        if entity not in dbpedia:
            continue
        for rel, objects in dbpedia[entity].items():
            if rel in selected_rels:
                for obj in (objects if isinstance(objects, list) else [objects]):
                    triples.append((entity, rel, obj))
                    if len(triples) >= max_triples:
                        return triples
    return triples


# ──────────────────────────────────────────────────────────────────────
# Stage 4 — Inference (FactKG)
# ──────────────────────────────────────────────────────────────────────

INF_PROMPT = """\
Based on the given evidence graph, determine whether the claim is True or False.
Answer with only 'True' or 'False'.

Evidence Graph:
{graph}

Claim: {claim}
Answer:"""


def infer_factkg(claim: str, triples: list, tokenizer, flan_model, device) -> bool:
    graph_str = "\n".join(f"({s}, {r}, {o})" for s, r, o in triples) or "No evidence found."
    prompt = INF_PROMPT.format(graph=graph_str, claim=claim)
    result = flan_generate(prompt, tokenizer, flan_model, device, max_new_tokens=10)
    return "true" in result.lower()


# ──────────────────────────────────────────────────────────────────────
# FactKG full pipeline
# ──────────────────────────────────────────────────────────────────────

def run_factkg(cfg: dict, n_samples: int, device: torch.device):
    factkg_dir = os.path.dirname(cfg["factkg"]["test_path"])
    dbpedia_dir = os.path.dirname(cfg["factkg"]["dbpedia_path"])

    print("[FactKG] Loading test data and DBpedia…")
    with open(os.path.join(factkg_dir, "factkg_test.pickle"), "rb") as f:
        test_data = pickle.load(f)
    with open(os.path.join(dbpedia_dir, "dbpedia_2015_undirected_light.pickle"), "rb") as f:
        dbpedia = pickle.load(f)

    scorer = load_scorer(cfg["factkg"]["scorer_ckpt"], str(device))
    encoder = SentenceTransformer(cfg["model"]["sentence_encoder"])
    tokenizer, flan_model = load_flan(cfg["model"]["llm_name"],
                                       cfg["model"]["llm_dtype"], str(device))

    claims = random.sample(list(test_data.keys()), min(n_samples, len(test_data)))
    correct = 0
    results = []
    t0 = time.time()

    for i, claim in enumerate(claims):
        info = test_data[claim]
        entity_set = info.get("Entity_set", [])
        true_label = bool(info.get("Label", [False])[0])

        try:
            sub_sents = segment_claim(claim, entity_set, tokenizer, flan_model, device)
            sel_rels = select_relations(claim, sub_sents, entity_set, dbpedia,
                                        scorer, encoder, tokenizer, flan_model,
                                        device, cfg["factkg"]["top_k_relations"])
            triples = build_evidence_graph(entity_set, sel_rels, dbpedia)
            pred = infer_factkg(claim, triples, tokenizer, flan_model, device)
        except Exception as e:
            print(f"  [Error] sample {i}: {e}")
            pred = False

        if pred == true_label:
            correct += 1
        results.append({"claim": claim, "true": true_label, "pred": pred})

        if (i + 1) % 50 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"  {i+1}/{n_samples} | Acc: {correct/(i+1)*100:.2f}% | {elapsed:.1f}min")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracy = correct / len(results) * 100
    tp = sum(1 for r in results if r["true"] and r["pred"])
    fp = sum(1 for r in results if not r["true"] and r["pred"])
    fn = sum(1 for r in results if r["true"] and not r["pred"])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*50}")
    print(f"FactKG Results ({n_samples} samples)")
    print(f"  Accuracy : {accuracy:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"{'='*50}")

    metrics = {"accuracy": accuracy, "precision": precision,
               "recall": recall, "f1": f1, "n_samples": len(results)}
    out_path = os.path.join(cfg["logging"]["results_dir"], "improved_metrics.json")
    with open(out_path, "w") as f:
        json.dump({"factkg": metrics}, f, indent=2)
    print(f"[Saved] {out_path}")
    return metrics


# ──────────────────────────────────────────────────────────────────────
# MetaQA helpers
# ──────────────────────────────────────────────────────────────────────

METAQA_RELATIONS = [
    "directed_by", "written_by", "starred_actors", "release_year",
    "in_language", "has_genre", "has_imdb_rating", "has_imdb_votes", "has_tags",
]

METAQA_KW_MAP = [
    (["direct", "director"], "directed_by"),
    (["writ", "author", "script"], "written_by"),
    (["star", "act", "cast", "appear"], "starred_actors"),
    (["year", "when", "releas", "came out"], "release_year"),
    (["language", "spoken"], "in_language"),
    (["genre", "type", "kind of film"], "has_genre"),
    (["imdb rating", "rating", "score", "rated"], "has_imdb_rating"),
    (["votes", "how many people"], "has_imdb_votes"),
    (["tag", "keyword", "about"], "has_tags"),
]


def get_target_relation(question: str) -> Optional[str]:
    q = question.lower()
    for keywords, rel in METAQA_KW_MAP:
        if any(k in q for k in keywords):
            return rel
    return "starred_actors"


def extract_metaqa_entities(question: str) -> list:
    return re.findall(r"\[([^\]]+)\]", question)


def load_metaqa_qa(path: str) -> list:
    samples = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            q, ans_str = parts[0], parts[1]
            samples.append({
                "question": q,
                "answers": ans_str.split("|"),
                "entities": extract_metaqa_entities(q),
            })
    return samples


def load_metaqa_kb(kb_path: str) -> dict:
    kb: Dict[str, Dict[str, List[str]]] = {}
    with open(kb_path) as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            s, r, o = parts
            kb.setdefault(s, {}).setdefault(r, []).append(o)
    return kb


def build_metaqa_graph(entities: list, rels: list, kb: dict, hop: int) -> list:
    triples = []
    frontier = set(entities)
    for _ in range(hop):
        next_frontier = set()
        for ent in frontier:
            if ent not in kb:
                continue
            for rel, objs in kb[ent].items():
                if rel in rels:
                    for obj in objs:
                        triples.append((ent, rel, obj))
                        next_frontier.add(obj)
        frontier = next_frontier
    return triples


def infer_3hop_direct(entities: list, kb: dict, target_rel: str) -> list:
    """Direct 3-hop KB traversal (no LLM)."""
    frontier = set(entities)
    for _ in range(3):
        next_f = set()
        for ent in frontier:
            if ent in kb:
                for rel, objs in kb[ent].items():
                    if rel == target_rel or not next_f:
                        next_f.update(objs)
        frontier = next_f
    return list(frontier)


METAQA_INF_PROMPT = """\
Answer the question based on the evidence graph. Output only the answer entity.

Evidence Graph:
{graph}

Question: {question}
Answer:"""


def infer_metaqa(question: str, triples: list, tokenizer, flan_model, device) -> list:
    graph_str = "\n".join(f"({s}, {r}, {o})" for s, r, o in triples) or "No evidence."
    prompt = METAQA_INF_PROMPT.format(graph=graph_str, question=question)
    result = flan_generate(prompt, tokenizer, flan_model, device, max_new_tokens=64)
    # Split by common delimiters
    preds = [x.strip() for x in re.split(r"[,|;]", result) if x.strip()]
    return preds or [result.strip()]


def hits_at_1(preds: list, gold: list) -> bool:
    if not preds:
        return False
    return preds[0].lower() in [g.lower() for g in gold]


# ──────────────────────────────────────────────────────────────────────
# MetaQA full pipeline
# ──────────────────────────────────────────────────────────────────────

def run_metaqa(cfg: dict, hops: int, n_samples: int, device: torch.device):
    qa_key = f"qa_{hops}hop_test"
    qa_path = cfg["metaqa"][qa_key]
    kb_path = cfg["metaqa"]["kb_path"]

    print(f"[MetaQA] Loading {hops}-hop QA ({qa_path})…")
    samples = load_metaqa_qa(qa_path)
    kb = load_metaqa_kb(kb_path)

    scorer = load_scorer(cfg["metaqa"]["scorer_ckpt"], str(device))
    encoder = SentenceTransformer(cfg["model"]["sentence_encoder"])
    tokenizer, flan_model = load_flan(cfg["model"]["llm_name"],
                                       cfg["model"]["llm_dtype"], str(device))

    random.seed(42)
    samples = random.sample(samples, min(n_samples, len(samples)))
    correct = 0
    t0 = time.time()

    for i, sample in enumerate(samples):
        q = sample["question"]
        gold = sample["answers"]
        entities = sample["entities"]
        target_rel = get_target_relation(q)

        try:
            if hops == 3:
                preds = infer_3hop_direct(entities, kb, target_rel)
            else:
                # Score MetaQA relations
                combined = [f"{q} [SEP] {rel}" for rel in METAQA_RELATIONS]
                embs = encoder.encode(combined, show_progress_bar=False, convert_to_numpy=True)
                emb_tensor = torch.tensor(embs, dtype=torch.float32).to(device)
                scores = scorer.score(emb_tensor).cpu().numpy()
                top_k = cfg["metaqa"]["top_k_relations"]
                sel_rels = [METAQA_RELATIONS[j]
                            for j in sorted(range(len(scores)), key=lambda x: -scores[x])[:top_k]]
                triples = build_metaqa_graph(entities, sel_rels, kb, hop=hops)
                preds = infer_metaqa(q, triples, tokenizer, flan_model, device)
        except Exception as e:
            print(f"  [Error] sample {i}: {e}")
            preds = []

        if hits_at_1(preds, gold):
            correct += 1

        if (i + 1) % 50 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"  {i+1}/{n_samples} | Hits@1: {correct/(i+1)*100:.2f}% | {elapsed:.1f}min")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    hits = correct / len(samples) * 100
    print(f"\n{'='*50}")
    print(f"MetaQA {hops}-hop Results ({n_samples} samples)")
    print(f"  Hits@1: {hits:.2f}%")
    print(f"{'='*50}")

    out_path = os.path.join(cfg["logging"]["results_dir"], f"metaqa_{hops}hop_metrics.json")
    with open(out_path, "w") as f:
        json.dump({"hits_at_1": hits, "n_samples": len(samples), "hops": hops}, f, indent=2)
    print(f"[Saved] {out_path}")
    return hits


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KG-Flan Inference")
    parser.add_argument("--dataset", choices=["factkg", "metaqa"], required=True)
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    parser.add_argument("--hops", type=int, choices=[1, 2, 3], default=1,
                        help="MetaQA hop count")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = load_config(args.config)
    device = torch.device(cfg["model"]["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["logging"]["results_dir"], exist_ok=True)

    print(f"\n{'='*60}")
    print(f"KG-Flan — Inference ({args.dataset.upper()})")
    print(f"Device: {device} | Samples: {args.n_samples}")
    print(f"{'='*60}\n")

    if args.dataset == "factkg":
        run_factkg(cfg, args.n_samples, device)
    else:
        run_metaqa(cfg, args.hops, args.n_samples, device)


if __name__ == "__main__":
    main()
