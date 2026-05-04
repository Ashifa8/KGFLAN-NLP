# KG-Flan: Graph-Enhanced LLM Reasoning with Trained Relation Scoring

> **NLP Course Project** — Reproduction & Extension of KG-GPT  
> FAST-NUCES | Spring 2026  
> Group: 25I-7614 · 25I-7609

--

##  Overview

This project reproduces and extends **KG-GPT** (*Bi et al., 2023*), a framework that grounds large language model reasoning in structured knowledge graphs. The original paper used OpenAI's GPT (paid API) and evaluated on large-scale datasets — making full reproduction expensive. We replaced GPT with **Flan-T5-Large** (open-source, free) and introduced a **trained MLP Relation Scorer** as our core novelty.

We evaluate on two benchmarks:
- **FactKG** — knowledge graph-based fact verification (DBpedia-grounded)
- **MetaQA** — multi-hop question answering over a movie knowledge base

---

##  Architecture

```
          ┌─────────────────────────────────────────┐
          │           KG-Flan Pipeline               │
          │                                         │
  Claim ──►  Step 1: Sentence Segmentation          │
          │         (Flan-T5-Large)                  │
          │              │                           │
          │              ▼                           │
          │  Step 2: Graph Retrieval                 │
          │    [NOVELTY] MLP Relation Scorer         │
          │    → filters candidates                  │
          │    → Flan-T5 picks final relations       │
          │              │                           │
          │              ▼                           │
          │  Step 3: Inference                       │
          │         (Flan-T5-Large)                  │
          └──────────────┬──────────────────────────┘
                         │
                    True / False
```

The three steps mirror the original KG-GPT paper:
1. **Segmentation** — decompose multi-hop claims into atomic sub-sentences
2. **Retrieval** — select relevant KG relations (our novelty replaces GPT here)
3. **Inference** — verify claim against linearized evidence graph

---

##  Novelty: Trained MLP Relation Scorer

The original KG-GPT used GPT-4 for relation retrieval — expensive and closed-source. We replace this with a **lightweight trained MLP scorer** that ranks relation candidates before passing them to Flan-T5.

### Scorer Architecture

```
Input: [CLS(claim + relation)] → SentenceTransformer (all-MiniLM-L6-v2) → 384-dim embedding
                                                                                    │
                                                               ┌────────────────────┘
                                                               ▼
                                                   Linear(384 → 256) → ReLU → Dropout(0.3)
                                                   Linear(256 → 64)  → ReLU → Dropout(0.3)
                                                   Linear(64  → 1)   → Sigmoid
                                                               │
                                                        Relevance Score ∈ [0,1]
```

### Training Details

| Parameter | FactKG Scorer | MetaQA Scorer |
|-----------|--------------|---------------|
| Architecture | MLP (384→256→64→1) | MLP (384→256→64→1) |
| Embedder | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 |
| Optimizer | Adam (lr=0.001) | Adam (lr=0.001) |
| Epochs | 10 | 10 |
| Batch Size | 64 | 64 |
| Training Pairs | ~11,000 | 11,436 |
| Val Accuracy | **85.47%** | **95.63%** |

Training pairs are constructed as:
- **Positive**: (claim, relation) pairs where the relation appears in ground-truth evidence
- **Negative**: (claim, relation) pairs where the relation is in the KG but not in evidence

---

##  Results

### FactKG — Fact Verification

| Method | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|-----|
| BlueBERT (full data) | 59.93% | — | — | — |
| Flan-T5 zero-shot | 62.70% | — | — | — |
| **Ours (w/o Scorer)** | 57.80% | 0.575 | 0.206 | 0.304 |
| BERT (full data) | 65.20% | — | — | — |
| **Ours (with Scorer)** | **65.60%** | **0.815** | **0.296** | **0.434** |
| ChatGPT 12-shot | 68.48% | — | — | — |
| KG-GPT 12-shot (paper) | 72.68% | — | — | — |
| GEAR (full data) | 77.65% | — | — | — |

 Our scorer-augmented system (+7.8% accuracy over no-scorer baseline) matches BERT trained on full data, using only Flan-T5 + a lightweight MLP — no paid API, no full dataset training.

### MetaQA — Multi-hop QA (Hits@1)

| Method | 1-hop | 2-hop | 3-hop | Average |
|--------|-------|-------|-------|---------|
| Without Scorer | 73.8% | 61.2% | 73.4% | 69.47% |
| With Scorer | 73.8% | 61.2% | 73.4% | 69.47% |

> MetaQA has only 9 unique relations in the KB — the scorer has minimal effect at this scale. The system still achieves strong multi-hop performance driven by Flan-T5's reasoning.

### Ablation Study

| Dataset | Without Scorer | With Scorer | Improvement |
|---------|---------------|-------------|-------------|
| FactKG | 57.8% | 65.6% | **+7.80%** |
| MetaQA (avg) | 69.47% | 69.47% | +0.00% |

---

##  Repository Structure

```
KGFlan-NLP/
│
├── nlp-kggpt-25i-7614-25i-6709.ipynb   # Main notebook (full pipeline)
│
├── results/
│   ├── factkg_results.csv               # FactKG evaluation results
│   ├── metaqa_results.csv               # MetaQA evaluation results
│   └── ablation_results.csv             # Ablation: scorer vs no-scorer
│
├── figures/
│   ├── scorer_training.png              # MLP scorer training curves
│   ├── factkg_comparison.png            # FactKG method comparison
│   ├── metaqa_hops.png                  # MetaQA hop-wise results
│   └── ablation_study.png               # Ablation study visualization
│
├── data/
│   ├── kb.txt                           # MetaQA knowledge base
│   ├── qa_test.txt                      # MetaQA test QA pairs
│   └── training_details.json            # Scorer training hyperparameters
│
└── README.md
```

---

##  Setup & Usage

### Requirements

```bash
pip install sentence-transformers transformers torch scikit-learn pandas numpy
```

### Data

| Dataset | Source | Access |
|---------|--------|--------|
| FactKG | [FactKG Paper](https://arxiv.org/abs/2305.06590) | Kaggle: `ozzy37/factkgg-db-pedia` |
| DBpedia | Bundled with FactKG | Same Kaggle dataset |
| MetaQA | [MetaQA Repo](https://github.com/yuyuz/MetaQA) | Public |
| Flan-T5-Large | HuggingFace / Kaggle | `google/flan-t5/pytorch/large/3` |

### Running the Pipeline

Open `nlp-kggpt-25i-7614-25i-6709.ipynb` in **Kaggle** (recommended — GPU + pre-loaded datasets).

The notebook is structured as:
1. **Cell 0** — Imports & setup
2. **Cells 1–4** — Data loading (FactKG + DBpedia + MetaQA)
3. **Cells 5–8** — MLP Scorer: data creation, model definition, training
4. **Cells 9–14** — Full KG-Flan pipeline (segmentation → retrieval → inference)
5. **Cells 15–21** — Evaluation on dev/test sets
6. **Cells 26+** — Final results, ablation, analysis

---

## Design Decisions & Why They Work

### Why Flan-T5 instead of GPT?
GPT-3/4 API costs made full evaluation on FactKG (~10K samples) and MetaQA prohibitively expensive. Flan-T5-Large is instruction-tuned, runs on a single GPU (or even CPU), and achieves competitive zero-shot reasoning.

### Why an MLP Scorer?
The original paper passes *all* KG relations for an entity to GPT for selection — GPT can handle this with a large context window. Flan-T5 has a much smaller context limit. The MLP scorer acts as a **pre-filter**, reducing hundreds of candidate relations down to top-k=5 before Flan-T5 processes them. This makes the pipeline feasible on free-tier hardware.

### Why the Scorer Helps on FactKG but Not MetaQA?
FactKG's DBpedia has hundreds of unique relations — precise filtering matters greatly. MetaQA's KB has only 9 relations total, so even random selection barely affects results.

---

##  Limitations

- **Scale**: Evaluated on subsets (200–500 samples) due to inference time (~4–5 sec/sample on CPU; ~30 min for 500 samples on GPU)
- **Gap vs. paper**: KG-GPT (72.68%) vs. ours (65.60%) — gap is primarily due to GPT-4's superior relation reasoning ability, not the framework design
- **MetaQA scorer**: Minimal KB size makes the trained scorer redundant for this dataset
- **No fine-tuning**: Flan-T5 is used zero-shot; fine-tuning on FactKG training data would likely close the gap further

---

##  Future Work

- Fine-tune Flan-T5 on FactKG training data for the inference step
- Replace all-MiniLM with a domain-adapted embedder (e.g., fine-tuned on KG triples)
- Quality-weighted ensemble: combine scorer confidence with Flan-T5 perplexity
- Scale evaluation to full test sets using batched GPU inference
- Test on additional KG benchmarks (WebQSP, ComplexWebQ)

---

##  References

1. Bi, Z., et al. *KG-GPT: A General Framework for Reasoning on Knowledge Graphs Using Large Language Models.* EMNLP 2023 Findings. [arXiv:2310.11220](https://arxiv.org/abs/2310.11220)
2. Kim, J., et al. *FactKG: Fact Verification via Reasoning on Knowledge Graphs.* ACL 2023. [arXiv:2305.06590](https://arxiv.org/abs/2305.06590)
3. Zhang, Y., et al. *MetaQA: Combining Expert Agents for Multi-Step Question Answering in Tabular and Textual Data.* 2018.
4. Reimers, N. & Gurevych, I. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.
5. Chung, H., et al. *Scaling Instruction-Finetuned Language Models (Flan-T5).* JMLR 2024.

---

## Group Members

| Name | Roll No |
|------|---------|
| Shanzae Khan| 25I-7614 |
| Ashifa Ikram | 25I-7609 |

*FAST-NUCES Islamabad — NLP Course, Spring 2026*
