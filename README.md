# TrustDoc RAG

> **Experimental PDF RAG system for studying how document structure, chunking, dense/sparse retrieval, RRF fusion, and CrossEncoder reranking affect evidence ranking quality.**

TrustDoc RAG는 단순 PDF QA 챗봇이 아니라, PDF 문서 구조와 retrieval pipeline 설계가 근거 검색 품질에 미치는 영향을 비교하는 실험 시스템입니다. Development 성능만으로 pipeline을 선택하지 않고, 별도로 동결한 held-out query에서 일반화 여부를 확인했으며 paired bootstrap, latency measurement, 원본 PDF 기반 systematic error analysis로 개선과 실패 원인을 분석했습니다.

## At a Glance

| Item | Scale |
|---|---:|
| Development queries | 32 |
| Frozen held-out queries | 35 |
| Document/chunking settings | 4 |
| Retrieval pipelines | 5 |
| Chunking strategies | 3 |
| Paired bootstrap resamples | 10,000 |
| Manually audited error cases | 28 |

### Key Results

- Dense candidate reranking showed a positive MRR direction across all four held-out settings.
- Hybrid retrieval outperformed Dense by point estimate on both OS held-out chunking settings.
- Hybrid candidate reranking improved all four development settings, but did not consistently reproduce that direction on frozen held-out queries.
- BM25 remained competitive on terminology-heavy AI-paper questions.
- Chunking effectiveness depended on the downstream retrieval pipeline and on PDF structure, including section boundaries and table remnants.

These are results within this corpus and query sample. Method uncertainty, paired intervals, latency context, and known evaluation limitations are reported below.

## Project Overview

The system retrieves evidence pages from two document domains:

- Operating Systems lecture slides
- Three RAG-related papers: RAG, SELF-RAG, and Ragas

Five pipelines are evaluated under an identical unified benchmark:

1. Dense retrieval
2. BM25 retrieval
3. Dense retrieval + CrossEncoder reranking
4. Dense + BM25 candidate retrieval with Reciprocal Rank Fusion (RRF)
5. Hybrid retrieval + CrossEncoder reranking

The experimental unit is evidence retrieval, not generated-answer quality. Hit@k and MRR measure whether an annotated source page is ranked highly; the serving layer then exposes the same retrieval components through FastAPI and Streamlit.

## Why This Project?

RAG systems can fail even when the correct document exists in the corpus. The important question is often not whether evidence is retrievable at all, but why it is ranked below a semantic neighbor, a repeated lexical match, an adjacent page, or a malformed chunk.

This project therefore focuses on four linked problems:

- separating candidate retrieval quality from final reranking quality;
- measuring how chunk boundaries interact with downstream retrieval;
- checking whether development-set improvements survive unseen queries;
- connecting aggregate metrics to concrete PDF-level failure modes.

## Research Questions

- **RQ1 — Retrieval methods:** How much do dense, sparse, hybrid, and reranked pipelines differ in evidence ranking quality?
- **RQ2 — Reranking generalization:** Does reranking consistently improve retrieval on unseen queries?
- **RQ3 — Chunking interaction:** How does chunking strategy interact with the downstream retrieval and reranking pipeline?
- **RQ4 — Development vs held-out:** Do improvements observed during pipeline exploration generalize to a frozen held-out query set?
- **RQ5 — Failure modes:** What document and pipeline factors explain retrieval improvements and degradations?

## System Overview

```mermaid
flowchart TD
    PDF[PDF documents] --> Parse[PyMuPDF text parsing]
    Parse --> Fixed[Fixed-size chunks]
    Parse --> Paragraph[Paragraph/page chunks]
    Parse --> Section[Section-aware chunks]

    Fixed --> Dense[Dense retrieval]
    Paragraph --> Dense
    Section --> Dense
    Fixed --> BM25[BM25 retrieval]
    Paragraph --> BM25
    Section --> BM25

    Dense --> RRF[Reciprocal Rank Fusion]
    BM25 --> RRF
    Dense --> Direct[Direct evidence ranking]
    BM25 --> Direct
    RRF --> Direct
    Dense --> Rerank[Optional CrossEncoder reranking]
    RRF --> Rerank
    Rerank --> Evaluate[Hit@k / MRR / latency]
    Direct --> Evaluate
```

The answer-generation path uses the retrieved chunks to build a Gemini prompt and returns answer text with source citations. Retrieval evaluation does not call the generation model.

## Experimental Design

### Document and chunking settings

| Setting | Domain | Chunking |
|---|---|---|
| `os_paragraph` | OS lecture slides | Paragraph/page-based |
| `os_fixed` | OS lecture slides | Fixed-size |
| `ai_papers_paragraph` | RAG-related papers | Paragraph/page-based |
| `ai_papers_section` | RAG-related papers | Text-based section-aware |

Section-aware chunking detects extracted headings such as `Introduction`, `Methods`, or numbered subsections. It is not a layout model and does not reconstruct tables or figures.

### Retrieval pipelines

| Pipeline | Candidate generation | Final ranking |
|---|---|---|
| `dense` | Chroma + multilingual sentence embedding | Dense similarity |
| `bm25` | BM25 keyword retrieval | BM25 score |
| `dense_rerank` | Dense top 10 | CrossEncoder |
| `hybrid` | Dense top 10 + BM25 top 10 | RRF (`k=60`) |
| `hybrid_rerank` | RRF candidate top 10 | CrossEncoder |

Shared benchmark conditions:

- final `top_k = 10`;
- embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`;
- reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`;
- fixed method order in the unified runner;
- initialization and warm-up measured separately from per-query latency.

## Datasets and Evaluation Split

| Split | OS | AI Papers | Total | Role |
|---|---:|---:|---:|---|
| Development | 20 | 12 | 32 | Pipeline exploration |
| Frozen held-out | 20 | 15 | 35 | Post-development evaluation |

The held-out set was built after development experimentation and frozen before its benchmark was run.

- development answer-page overlap: **0**;
- exact duplicate queries: **0**;
- manually identified near-duplicates: **0**;
- frozen commit: `ea47aa516a77a998aee5430f7daceb8c3ac07e75`;
- query text, ground-truth pages, and retrieval parameters were not revised after observing held-out results.

The [held-out dataset audit](eval/heldout_dataset_audit.md) validates PDF basenames, 1-based page ranges, query metadata, duplicate status, and ground-truth pages. It also records that retrieval had not been run at freeze time. The held-out queries cover definition, comparison, mechanism, reason, exact-terminology, similar-concept, and multi-page questions.

## Evaluation Metrics

| Metric | Interpretation |
|---|---|
| Hit@1 | At least one valid evidence page is ranked first |
| Hit@3 / Hit@5 / Hit@10 | Valid evidence appears within the corresponding cutoff |
| MRR | Reciprocal rank of the first valid evidence page, averaged across queries |
| Bootstrap 95% CI | Percentile interval from query-level resampling |
| Paired delta | Per-query metric difference between method or chunking pairs |
| Mean / P50 / P95 latency | Per-query runtime distribution after initialization and warm-up |

Bootstrap configuration:

```text
resamples = 10,000
seed = 42
confidence level = 95%
sampling unit = query
```

Paired comparisons resample aligned query-level differences. Reported bootstrap positive/zero/negative fractions are directional resample fractions, not posterior probabilities or formal p-values.

## Development Results

Each cell reports `Hit@1 / MRR`.

| Setting | Dense | BM25 | Dense + Rerank | Hybrid | Hybrid + Rerank |
|---|---:|---:|---:|---:|---:|
| OS paragraph | .6500 / .7875 | .7000 / .7917 | .8000 / .8917 | .7000 / .7954 | .8000 / .9000 |
| OS fixed | .7500 / .8338 | .7500 / .8125 | .7500 / .8750 | .7000 / .8181 | .7500 / .8750 |
| AI paragraph | .6667 / .7986 | .6667 / .7743 | .7500 / .8403 | .6667 / .8333 | .7500 / .8611 |
| AI section-aware | .5833 / .6948 | .5000 / .6500 | .7500 / .8250 | .5833 / .7500 | .8333 / .8889 |

Development observations:

- Dense reranking increased aggregate MRR in all four settings.
- Hybrid reranking also increased aggregate MRR over Hybrid in all four settings.
- Hybrid alone did not uniformly improve over Dense.
- Section-aware Dense was comparatively weak, while section-aware Hybrid + Rerank had the highest point estimate within that development setting.
- Aggregate bootstrap intervals were broad and often overlapping, so these point estimates were not treated as a final method ranking.

These results motivated a second question: **would the observed directions reproduce on queries that were not used during pipeline exploration?**

## Frozen Held-out Evaluation

Each cell reports `Hit@1 / MRR` for the frozen queries.

| Setting | Dense | BM25 | Dense + Rerank | Hybrid | Hybrid + Rerank |
|---|---:|---:|---:|---:|---:|
| OS paragraph | .7000 / .8017 | .7500 / .8167 | .7500 / .8417 | **.9000 / .9187** | .7000 / .8292 |
| OS fixed | .6500 / .7600 | .7000 / .7958 | .7500 / .8375 | **.8500 / .9000** | .7000 / .8292 |
| AI paragraph | .4000 / .5322 | **.6667 / .7206** | .6000 / .6796 | .5333 / .6617 | .5333 / .6337 |
| AI section-aware | .4000 / .4917 | **.6667 / .7500** | .4667 / .5417 | .4667 / .6106 | .5333 / .6911 |

Point-estimate patterns differed by domain:

- Hybrid had the highest held-out Hit@1 and MRR in both OS settings.
- BM25 had the highest held-out MRR in both AI-paper settings.
- Dense reranking improved MRR directionally over Dense in all four settings.
- Hybrid reranking was lower than Hybrid in OS paragraph, OS fixed, and AI paragraph, but higher in AI section-aware.

### Development vs held-out reranking direction

| Setting | Dev: Hybrid → Hybrid + Rerank ΔMRR | Held-out ΔMRR |
|---|---:|---:|
| OS paragraph | +.1046 | -.0896 |
| OS fixed | +.0569 | -.0708 |
| AI paragraph | +.0278 | -.0280 |
| AI section-aware | +.1389 | +.0806 |

The development improvement of Hybrid reranking did not consistently generalize. By contrast, held-out Dense → Dense + Rerank MRR deltas were positive in all four settings: `+.0400`, `+.0775`, `+.1474`, and `+.0500`. This is a more stable observed direction, not a claim of universal reranker behavior.

## Paired Method Comparison

Selected held-out paired results are shown below. `A → B` means the delta is `B - A`.

| Setting | Comparison | Metric | Delta | Paired 95% CI |
|---|---|---|---:|---:|
| OS paragraph | Dense → Hybrid | Hit@1 | +.2000 | [.0500, .4000] |
| OS paragraph | Dense → Hybrid | MRR | +.1171 | [.0229, .2300] |
| OS fixed | Dense → Hybrid | Hit@1 | +.2000 | [.0500, .4000] |
| OS fixed | Dense → Hybrid | MRR | +.1400 | [.0375, .2575] |
| AI section-aware | Dense + Rerank → Hybrid + Rerank | MRR | +.1494 | [.0333, .2972] |
| AI section-aware | BM25 → Hybrid | MRR | -.1394 | [-.3006, -.0033] |
| AI: paragraph → section-aware | Dense + Rerank | MRR | -.1380 | [-.2833, -.0222] |
| AI: paragraph → section-aware | Hybrid + Rerank | MRR | +.0574 | [.0167, .1089] |

These are percentile paired-bootstrap intervals from only 20 OS or 15 AI held-out queries. An interval excluding zero is informative for this paired sample, but it is not a formal p-value and does not remove small-sample or corpus-selection uncertainty.

### Latency

The table summarizes the range across the four held-out settings in milliseconds.

| Pipeline | Mean range | P50 range | P95 range |
|---|---:|---:|---:|
| Dense | 22.74–67.72 | 22.35–40.72 | 37.68–241.79 |
| BM25 | 0.77–1.07 | 0.70–1.04 | 1.13–1.31 |
| Dense + Rerank | 61.72–110.83 | 61.88–104.94 | 74.07–146.04 |
| Hybrid | 11.72–14.91 | 11.47–15.43 | 13.29–16.94 |
| Hybrid + Rerank | 49.46–95.24 | 41.41–94.51 | 72.12–129.12 |

BM25 was very fast for the current local corpus, while CrossEncoder reranking added a visible latency cost. These values depend on the specific hardware, runtime, corpus size, cache state, and local Chroma setup; they should not be generalized as algorithm-level performance.

## Systematic Error Analysis

Twenty-eight held-out comparison cases were manually audited against retrieved chunks and original PDF pages:

| Domain | Cases |
|---|---:|
| Operating Systems | 14 |
| AI Papers | 14 |
| **Total** | **28** |

Cases were selected purposively to cover large rank changes and distinct mechanisms. The counts below describe this manual sample, not the prevalence of failures in the full benchmark.

| Pipeline mechanism | Cases |
|---|---:|
| `lexical_rescue` | 5 |
| `lexical_distraction` | 2 |
| `fusion_rescue` | 2 |
| `fusion_degradation` | 2 |
| `reranker_rescue` | 4 |
| `reranker_inversion` | 5 |
| `chunking_improvement` | 4 |
| `chunking_degradation` | 4 |

Each annotation separates directly verifiable **observation** from evidence-based **interpretation**, and uses a second axis for document factors such as exact terminology, semantic neighbors, multi-page evidence, chunk granularity, section boundaries, or table/layout effects.

### Representative cases

| Mechanism | Case | Rank change | PDF-level observation |
|---|---|---:|---|
| Lexical rescue | `os_test_q17` | Dense 5 → BM25 1 | The evidence page directly contains the requested terms `Starvation` and `Aging`. |
| Reranker inversion | `ai_test_q07` | Dense 1 → Dense + Rerank 9 | General SELF-RAG descriptions outranked the appendix page containing the generator-data procedure. |
| Reranker rescue | `ai_test_q01` | Dense 5 → Dense + Rerank 1 | The CrossEncoder recovered the page defining Q-BLEU rather than general Jeopardy result pages. |
| Fusion rescue | `os_test_q20` | Dense 2 → Hybrid 1 | Hybrid replaced a short critical-section fragment with the slide listing all three required properties. |
| Chunking improvement | `ai_test_q12` | Paragraph Dense 2 → Section Dense 1 | One section chunk preserved the 2018/2020 Wikipedia corpus comparison and its rationale. |
| Chunking degradation | `ai_test_q13` | Paragraph Dense 1 → Section Dense 8 | Table residue was detected as a heading and mixed with the ISUSE explanation. |

### Error-analysis findings

1. **Exact terminology matters.** It was the most frequently observed primary factor among the manually selected cases, particularly for paper-specific names, labels, years, and lecture terms.
2. **Reranking errors were often subtle.** Inversions frequently occurred among semantic neighbors, partial answers, and closely related explanations rather than obviously unrelated passages.
3. **Chunking has no uniformly better direction.** Improvements and degradations depended on section preservation, fragment size, table remnants, incorrect heading detection, and multi-page splitting.
4. **Multi-page evidence exposes a metric limitation.** Page-level Hit/MRR can credit one valid page even when a complete answer requires complementary evidence from several pages.

## Key Findings

- Correct evidence was often already present in top-k; ranking quality was a major bottleneck.
- Dense candidate reranking had a positive held-out MRR direction across all settings, while reranking Hybrid candidates was not consistently beneficial.
- Hybrid retrieval improved Dense strongly in the OS held-out comparisons, but its benefit was less stable for AI papers.
- BM25 remained competitive for terminology-heavy scientific-paper queries and could also be distracted by repeated vocabulary.
- Chunking effectiveness depended jointly on document structure and the downstream retrieval/reranking method.
- Development-set improvements did not always reproduce on frozen held-out queries.

## Engineering and Demo

The experimental retrieval stack is exposed as a small application rather than remaining evaluation-only code.

| Layer | Implementation |
|---|---|
| PDF processing | PyMuPDF |
| Dense retrieval | SentenceTransformers + Chroma |
| Sparse retrieval | `rank-bm25` |
| Fusion | Reciprocal Rank Fusion |
| Reranking | CrossEncoder |
| Generation | Gemini via `google-genai` |
| API | FastAPI |
| UI | Streamlit |

API endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /` | Service metadata |
| `POST /rag/retrieve` | Retrieval debugging without generation |
| `POST /rag/query` | Retrieval, prompt construction, Gemini answer, and source citations |

Example retrieval request:

```bash
curl -X POST "http://127.0.0.1:8000/rag/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are first-fit and best-fit in contiguous allocation?",
    "collection": "trustdoc_os_paragraph",
    "chunks_path": "data/processed/chunks_paragraph_all.json",
    "retrieval_mode": "hybrid",
    "use_rerank": true,
    "top_k": 5,
    "initial_top_k": 10,
    "dense_top_k": 10,
    "bm25_top_k": 10
  }'
```

Compact response shape:

```json
{
  "query": "What are first-fit and best-fit in contiguous allocation?",
  "collection": "trustdoc_os_paragraph",
  "retrieval_mode": "hybrid",
  "use_rerank": true,
  "retrieved_sources": [
    {
      "rank": 1,
      "source_file": "2026-OS-L9A-MainMemory.pdf",
      "page_number": 13,
      "chunk_id": "13_para_0",
      "dense_rank": 6,
      "bm25_rank": 1,
      "hybrid_score": 0.03154,
      "rerank_score": 6.8276,
      "text_preview": "..."
    }
  ]
}
```

The Streamlit UI provides document-set selection, Dense/Hybrid choice, optional reranking, source tables, expandable retrieved chunks, and raw response inspection.

<img width="1512" height="857" alt="TrustDoc RAG Streamlit demo" src="https://github.com/user-attachments/assets/2e89df56-fbaf-4701-9b9e-0aa0fab4105c" />

## Repository Structure

```text
TrustDoc-RAG/
├── app/
│   ├── ingest/          # PDF loading and three chunking strategies
│   ├── retrieval/       # Chroma index construction and search
│   ├── rag/             # Dense, BM25, Hybrid, reranking, prompt, Gemini
│   ├── eval/            # Benchmark, metrics, paired comparison, error analysis
│   ├── api/             # FastAPI service
│   └── ui/              # Streamlit demo
├── configs/             # Four development and four held-out benchmark configs
├── eval/
│   ├── questions*.jsonl # Development and frozen held-out queries
│   └── heldout_dataset_audit.md
├── data/
│   ├── raw/             # Local PDFs
│   ├── processed/       # Local chunk artifacts
│   └── chroma/          # Local vector indexes
├── tests/
├── requirements.txt
└── README.md
```

`data/raw`, `data/processed`, `data/chroma`, `eval/results`, and `eval/analysis` are local artifacts excluded from Git. Evaluation tables in this README were regenerated from those frozen local artifacts.

## Installation

```bash
git clone https://github.com/SJA-03/TrustDoc_Rag.git
cd TrustDoc_Rag

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For answer generation, create `.env`:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Retrieval ingestion, indexing, and evaluation do not require a Gemini call.

## How to Reproduce

### 1. Place source PDFs

```text
data/raw/OS/*.pdf
data/raw/ai_papers/*.pdf
```

### 2. Parse and chunk

```bash
venv/bin/python app/ingest/batch_ingest.py \
  --input_dir data/raw/OS \
  --strategy paragraph \
  --output data/processed/chunks_paragraph_all.json

venv/bin/python app/ingest/batch_ingest.py \
  --input_dir data/raw/OS \
  --strategy fixed \
  --output data/processed/chunks_fixed_all.json

venv/bin/python app/ingest/batch_ingest.py \
  --input_dir data/raw/ai_papers \
  --strategy paragraph \
  --output data/processed/chunks_ai_papers_paragraph.json

venv/bin/python app/ingest/batch_ingest.py \
  --input_dir data/raw/ai_papers \
  --strategy section \
  --output data/processed/chunks_ai_papers_section.json
```

### 3. Build Chroma indexes

The chunk and collection names must match the selected benchmark config. Example:

```bash
venv/bin/python app/retrieval/build_index.py \
  --chunks data/processed/chunks_paragraph_all.json \
  --persist_dir data/chroma \
  --collection trustdoc_os_paragraph
```

Repeat for `trustdoc_os_fixed`, `trustdoc_ai_papers_paragraph`, and `trustdoc_ai_papers_section` with their corresponding chunk files.

### 4. Validate the frozen held-out set

This validates schema, exact duplicates, PDF presence, and page bounds without running retrieval:

```bash
PYTHONPATH=. venv/bin/python app/eval/validate_questions.py \
  --heldout eval/questions_os_heldout.jsonl eval/questions_ai_papers_heldout.jsonl \
  --dev eval/questions.jsonl eval/questions_ai_papers.jsonl \
  --source-root data/raw
```

### 5. Run the unified benchmark

Development example:

```bash
PYTHONPATH=. venv/bin/python app/eval/run_benchmark.py \
  --config configs/benchmark_os_paragraph.json \
  --bootstrap-resamples 10000 \
  --seed 42
```

Held-out example:

```bash
PYTHONPATH=. venv/bin/python app/eval/run_benchmark.py \
  --config configs/benchmark_os_paragraph_heldout.json \
  --bootstrap-resamples 10000 \
  --seed 42 \
  --output-root eval/results/heldout
```

Equivalent configs exist for OS fixed, AI paragraph, and AI section-aware. Each run writes `summary.json`, `summary.md`, `summary.csv`, and `per_query.json` under its output directory.

### 6. Run paired comparisons

```bash
PYTHONPATH=. venv/bin/python app/eval/compare_methods.py \
  --artifact os_paragraph=eval/results/heldout/os_paragraph/per_query.json \
  --artifact os_fixed=eval/results/heldout/os_fixed/per_query.json \
  --artifact ai_papers_paragraph=eval/results/heldout/ai_papers_paragraph/per_query.json \
  --artifact ai_papers_section=eval/results/heldout/ai_papers_section/per_query.json \
  --chunking-pair os_chunking=os_paragraph:os_fixed \
  --chunking-pair ai_chunking=ai_papers_paragraph:ai_papers_section \
  --resamples 10000 \
  --seed 42 \
  --confidence 0.95 \
  --output-dir eval/analysis/heldout
```

### 7. Validate and summarize manual error annotations

```bash
PYTHONPATH=. venv/bin/python app/eval/summarize_error_analysis.py \
  --annotations eval/analysis/error_analysis/manual_case_annotations.json \
  --output-dir eval/analysis/error_analysis
```

### 8. Run the API and UI

```bash
PYTHONPATH=. venv/bin/python -m uvicorn app.api.main:app --reload
```

In another terminal:

```bash
venv/bin/streamlit run app/ui/streamlit_app.py
```

- API documentation: `http://127.0.0.1:8000/docs`
- Streamlit: `http://localhost:8501`

## Limitations

- The development set has 32 queries and the held-out set has 35; both remain relatively small.
- Percentile paired-bootstrap intervals remain sensitive to the number and composition of queries.
- The document domains are limited to OS lecture slides and three RAG-related papers.
- Manual error analysis is a purposive 28-case sample, not an unbiased failure-distribution estimate.
- Page-level evidence metrics do not fully measure whether all evidence needed for a multi-page answer was jointly retrieved.
- PDF processing is text-based; no OCR or learned layout model is used.
- Section detection is heuristic, and tables or extracted text order can create malformed boundaries.
- BM25 uses simple regex tokenization rather than a language-specific analyzer.
- Latency measurements are specific to the local hardware, runtime, cache state, corpus, and index.
- Generation quality, answer faithfulness, and citation completeness have not been systematically evaluated.

## Future Work

### Evaluation

- Expand the held-out corpus and query set.
- Reassess interval robustness with more paired queries.
- Add answer-level faithfulness, citation correctness, and multi-page evidence-sufficiency evaluation.

### Retrieval

- Explore query-adaptive lexical/semantic fusion.
- Use the observed error taxonomy to design more selective reranking strategies.

### Document Processing

- Add layout-aware and table-aware parsing.
- Apply OCR only to documents or pages that require it.

The current retrieval experiments are treated as the final project result; future work should begin from the frozen benchmark and documented failure modes rather than tuning the existing held-out queries.
