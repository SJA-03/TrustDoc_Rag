# Held-out Retrieval Dataset Audit

This report freezes the held-out query design before any retrieval result is observed. Existing `eval/questions*.jsonl` files are treated as development sets. No dense, BM25, hybrid, rerank, or unified benchmark command was run against the held-out questions during construction or review.

## Dataset Summary

| Split | Queries |
|---|---:|
| OS held-out | 20 |
| AI Papers held-out | 15 |
| Total | 35 |

The development sets contain 20 OS and 12 AI Papers queries. Development and held-out results must remain separate in future reporting.

## Query Type Distribution

| Query type | OS | AI Papers |
|---|---:|---:|
| definition | 3 | 2 |
| comparison | 3 | 2 |
| mechanism | 3 | 2 |
| reason | 3 | 3 |
| exact_terminology | 3 | 2 |
| similar_concept | 3 | 2 |
| multi_page | 2 | 2 |
| **Total** | **20** | **15** |

Each query has one primary type even when it also has secondary characteristics.

## Language Distribution

| Language | OS | AI Papers |
|---|---:|---:|
| English (`en`) | 15 | 15 |
| Korean (`ko`) | 5 | 0 |

## Source Distribution

### OS held-out

| Source PDF | Queries |
|---|---:|
| `2026-OS-L2-OS-Structure.pdf` | 2 |
| `2026-OS-L2B-Structure-1.pdf` | 3 |
| `2026-OS-L3B-Processes_Part2.pdf` | 3 |
| `2026-OS-L4A-Threads_Part1_v2.pdf` | 4 |
| `2026-OS-L4B-Threads_Part2.pdf` | 2 |
| `2026-OS-L5A_Scheduling_Part1.pdf` | 3 |
| `2026-OS-L5B_Scheduling_Part2-1.pdf` | 2 |
| `2026-OS-L6A_Synchronization.pdf` | 1 |

### AI Papers held-out

| Source PDF | Queries |
|---|---:|
| `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf` | 6 |
| `SELF-RAG- LEARNING TO RETRIEVE, GENERATE, AND CRITIQUE THROUGH SELF-REFLECTION.pdf` | 7 |
| `Ragas- Automated Evaluation of Retrieval Augmented Generation.pdf` | 2 |

## Source Document Inventory

Page numbers use `app/ingest/pdf_loader.py`: PDF page index plus one.

### Operating Systems

| Source PDF | Pages | Development answer pages | Held-out answer pages |
|---|---:|---|---|
| `2026-OS-L10A-VirtualMemory.pdf` | 28 | 7, 8, 10, 12, 21, 24, 25, 26 | - |
| `2026-OS-L10B-VirtualMemory.pdf` | 13 | 3, 4, 6, 7, 12 | - |
| `2026-OS-L11-MassStorage.pdf` | 18 | 7, 16, 17 | - |
| `2026-OS-L12-IOSystems.pdf` | 13 | 12 | - |
| `2026-OS-L13-FileSystem-2.pdf` | 21 | - | - |
| `2026-OS-L2-OS-Structure.pdf` | 15 | - | 3, 5 |
| `2026-OS-L2B-Structure-1.pdf` | 22 | - | 7, 14, 18, 21 |
| `2026-OS-L3A-Processes_Part1.pdf` | 23 | 13, 15, 17 | - |
| `2026-OS-L3B-Processes_Part2.pdf` | 16 | - | 4, 5, 11, 13 |
| `2026-OS-L4A-Threads_Part1_v2.pdf` | 30 | - | 7, 13, 14, 16 |
| `2026-OS-L4B-Threads_Part2.pdf` | 19 | - | 17, 18 |
| `2026-OS-L5A_Scheduling_Part1.pdf` | 25 | - | 8, 19, 23 |
| `2026-OS-L5B_Scheduling_Part2-1.pdf` | 18 | - | 4, 5, 14 |
| `2026-OS-L6A_Synchronization.pdf` | 18 | - | 9 |
| `2026-OS-L6B_Synchronization.pdf` | 29 | 23, 24, 25 | - |
| `2026-OS-L7A_Synchronization.pdf` | 15 | - | - |
| `2026-OS-L7B_Synchronization.pdf` | 21 | - | - |
| `2026-OS-L8-Deadlocks.pdf` | 21 | 3, 6, 17, 18, 20 | - |
| `2026-OS-L8B-Deadlocks.pdf` | 14 | 2 | - |
| `2026-OS-L9A-MainMemory.pdf` | 17 | 10, 12, 13 | - |
| `2026-OS-L9B-MainMemory.pdf` | 21 | - | - |

### AI Papers

| Source PDF | Pages | Development answer pages | Held-out answer pages |
|---|---:|---|---|
| `Ragas- Automated Evaluation of Retrieval Augmented Generation.pdf` | 8 | 1, 2, 3, 5 | 4, 8 |
| `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf` | 19 | 1, 2, 3, 4, 7, 8, 9 | 5, 6, 10, 17, 18 |
| `SELF-RAG- LEARNING TO RETRIEVE, GENERATE, AND CRITIQUE THROUGH SELF-REFLECTION.pdf` | 30 | 1, 2, 3, 4, 9, 10, 17 | 18, 19, 20, 29, 30 |

## Answer Page Overlap

The ground-truth key is `(source_file, page)`.

| Comparison | Overlapping held-out queries |
|---|---:|
| OS held-out vs OS development | 0 |
| AI held-out vs AI Papers development | 0 |
| **Total** | **0** |

No development answer page was reused. Multi-page answers were assigned only when distinct pages were necessary to cover separate parts of the question.

## Duplicate Check

| Check | Count |
|---|---:|
| Exact duplicate against development queries | 0 |
| Exact duplicate within held-out queries | 0 |
| Manually identified near-duplicate | 0 |

Near-duplicate review compared question intent, not only wording. Candidates overlapping existing definition questions about demand paging, deadlock, LRU/FIFO, RAG/SELF-RAG definitions, reflection-token basics, or the general decision of when SELF-RAG retrieves were excluded.

## Ground-truth Verification

1. Enumerated all PDFs under `data/raw/OS` and `data/raw/ai_papers` and recorded page counts with `pdfinfo`.
2. Recomputed development answer-page usage from the two existing development JSONL files.
3. Read full candidate-page text directly from each original PDF with PyMuPDF; chunks and retrieval outputs were not used.
4. Rendered 37 candidate pages with Poppler and visually reviewed source-grouped contact sheets. Thirty-five unique pages were retained as final ground truth; two rejected candidate pages were not used.
5. Confirmed that each retained page directly supports its question and that multi-page annotations are individually valid evidence for the portion they cover.
6. Validated PDF basename and 1-based page bounds against the local source inventory.

## Freeze Policy

These files are frozen before retrieval evaluation. Once the held-out benchmark is run, questions must not be removed or rewritten because of method performance. A later correction is allowed only for a documented annotation error.

**Held-out retrieval benchmark was NOT run.**
