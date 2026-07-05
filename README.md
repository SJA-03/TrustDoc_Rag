# TrustDoc RAG

PDF 문서를 기반으로 질문에 대한 답변과 근거 페이지를 함께 제공하는 문서 기반 RAG(Retrieval-Augmented Generation) MVP입니다.

운영체제 강의자료 PDF 21개를 대상으로 PDF 파싱, chunking, embedding, vector DB indexing, retrieval, LLM 기반 답변 생성, citation 출력, retrieval evaluation, reranking evaluation, hybrid retrieval evaluation을 구현했습니다.

---

## Overview

TrustDoc RAG는 PDF 문서를 미리 검색 가능한 형태로 저장해두고, 사용자가 질문하면 관련 문서 조각을 검색한 뒤 LLM에게 근거로 제공하여 답변을 생성하는 시스템입니다.

```text
PDF 파일
→ 텍스트 추출
→ Chunking
→ Embedding
→ Chroma Vector DB 저장
→ 질문 입력
→ 관련 chunk 검색
→ Prompt 생성
→ Gemini 답변 생성
→ 근거 Source 출력
```

추가적으로 retrieval 품질 개선을 위해 다음 실험을 진행했습니다.

```text
Dense Retrieval
→ CrossEncoder Reranking
→ BM25 + Dense Hybrid Retrieval
→ Hybrid Retrieval + CrossEncoder Reranking
```

---

## Features

- PDF 문서 텍스트 추출
- page/paragraph 기반 chunking
- fixed-size chunking
- SentenceTransformer 기반 embedding 생성
- Chroma Vector DB indexing
- Query 기반 semantic retrieval
- Gemini API 기반 RAG 답변 생성
- 답변 내 source citation 출력
- Retrieval 성능 평가
- 복수 정답 페이지 지원
- Hit@1, Hit@3, Hit@5, Hit@10, MRR 평가
- CrossEncoder 기반 reranking 실험
- BM25 + Dense 기반 hybrid retrieval 실험
- RRF(Reciprocal Rank Fusion) 기반 rank fusion
- Hybrid retrieval + reranking 실험

---

## Tech Stack

| Area | Stack |
|---|---|
| Language | Python |
| PDF Parsing | PyMuPDF |
| Embedding | sentence-transformers |
| Vector DB | Chroma |
| LLM | Gemini API |
| Reranker | CrossEncoder |
| Keyword Retrieval | rank-bm25 |
| Rank Fusion | Reciprocal Rank Fusion |
| Evaluation | Hit@k, MRR |

---

## Project Structure

```text
TrustDoc-RAG/
├── app/
│   ├── ingest/
│   │   ├── pdf_loader.py
│   │   ├── chunkers.py
│   │   ├── run_ingest.py
│   │   └── batch_ingest.py
│   ├── retrieval/
│   │   ├── build_index.py
│   │   └── search.py
│   ├── rag/
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   ├── hybrid_retriever.py
│   │   ├── build_prompt.py
│   │   ├── llm_client.py
│   │   ├── run_rag.py
│   │   └── test_gemini.py
│   └── eval/
│       ├── evaluate_retrieval.py
│       ├── evaluate_rerank.py
│       ├── evaluate_hybrid.py
│       └── evaluate_hybrid_rerank.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── chroma/
├── eval/
│   └── questions.jsonl
├── requirements.txt
├── .gitignore
└── README.md
```

> `data/raw`, `data/processed`, `data/chroma`, `.env`는 GitHub에 포함하지 않습니다.

---

## RAG Pipeline

### 1. Indexing

```text
PDF
→ page별 텍스트 추출
→ chunk 생성
→ embedding 생성
→ Chroma Vector DB 저장
```

### 2. Retrieval + Generation

```text
사용자 질문
→ query embedding
→ Chroma에서 관련 chunk 검색
→ 검색 결과로 prompt 생성
→ Gemini 답변 생성
→ 근거 source 출력
```

### 3. Retrieval Improvement

```text
Dense Retrieval
→ candidate chunks

Dense Retrieval + CrossEncoder Reranking
→ rerank_score 기준 재정렬

BM25 + Dense Hybrid Retrieval
→ RRF 기반 rank fusion

BM25 + Dense Hybrid Retrieval + CrossEncoder Reranking
→ hybrid candidate 생성 후 reranker로 최종 재정렬
```

---

## Chunking Strategies

### 1. Paragraph / Page-based Chunking

페이지 단위의 문맥을 비교적 잘 보존합니다.  
강의자료 PDF처럼 한 페이지에 하나의 개념이 정리된 문서에서 유리할 수 있습니다.

### 2. Fixed-size Chunking

일정한 글자 수 기준으로 chunk를 나눕니다.  
특정 키워드나 짧은 개념 검색에는 유리할 수 있지만, 문장이나 개념이 중간에 끊길 수 있습니다.

---

## Installation

```bash
git clone https://github.com/SJA-03/TrustDoc_Rag.git
cd TrustDoc_Rag

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Environment Variables

Gemini API를 사용하기 위해 `.env` 파일을 생성합니다.

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

---

## How to Run

### 1. Single PDF Ingestion

```bash
python app/ingest/run_ingest.py \
  --pdf "data/raw/2026-OS-L8-Deadlocks.pdf" \
  --strategy paragraph \
  --output data/processed/chunks_paragraph_test.json
```

### 2. Batch PDF Ingestion

```bash
python app/ingest/batch_ingest.py \
  --input_dir data/raw \
  --strategy paragraph \
  --output data/processed/chunks_paragraph_all.json
```

```bash
python app/ingest/batch_ingest.py \
  --input_dir data/raw \
  --strategy fixed \
  --output data/processed/chunks_fixed_all.json
```

### 3. Build Chroma Index

```bash
python app/retrieval/build_index.py \
  --chunks data/processed/chunks_paragraph_all.json \
  --collection trustdoc_os_paragraph
```

```bash
python app/retrieval/build_index.py \
  --chunks data/processed/chunks_fixed_all.json \
  --collection trustdoc_os_fixed
```

### 4. Search Test

```bash
python app/retrieval/search.py \
  --query "What are the four necessary conditions for deadlock?" \
  --collection trustdoc_os_paragraph \
  --top_k 5
```

### 5. Run RAG

```bash
python app/rag/run_rag.py \
  --query "What are the four necessary conditions for deadlock?" \
  --collection trustdoc_os_paragraph \
  --top_k 5
```

---

## Example Output

```text
Question:
What are the four necessary conditions for deadlock?

Answer:
For a deadlock to occur, four necessary conditions must hold simultaneously:

1. Mutual exclusion
2. Hold and wait
3. No preemption
4. Circular wait

Evidence
- [Source 1, 2026-OS-L8-Deadlocks.pdf, page 17]
- [Source 4, 2026-OS-L8-Deadlocks.pdf, page 6]
```

---

## Evidence vs Retrieved Sources

TrustDoc RAG는 retrieved source와 evidence를 구분합니다.

```text
Retrieved Sources:
검색 시스템이 top-k로 가져온 후보 chunk 목록

Evidence:
LLM이 최종 답변에서 실제로 사용했다고 명시한 근거 source
```

이 구분을 통해 단순 검색 결과와 실제 답변 근거를 분리해서 확인할 수 있습니다.

---

## Retrieval Evaluation

평가는 20개 질문으로 진행했습니다.  
하나의 질문에 여러 페이지가 유효한 근거가 될 수 있기 때문에 복수 정답 페이지를 지원합니다.

```json
{
  "id": "q1",
  "query": "What are the four necessary conditions for deadlock?",
  "answers": [
    {
      "file": "2026-OS-L8-Deadlocks.pdf",
      "page": 6
    },
    {
      "file": "2026-OS-L8-Deadlocks.pdf",
      "page": 17
    }
  ]
}
```

### Evaluation Metrics

| Metric | Meaning |
|---|---|
| Hit@1 | 정답 근거가 검색 결과 1등에 포함되는 비율 |
| Hit@3 | 정답 근거가 상위 3개 안에 포함되는 비율 |
| Hit@5 | 정답 근거가 상위 5개 안에 포함되는 비율 |
| Hit@10 | 정답 근거가 상위 10개 안에 포함되는 비율 |
| MRR | 정답 근거가 얼마나 높은 순위에 등장하는지를 반영하는 지표 |

---

## Baseline Retrieval

Baseline retrieval은 Chroma 기반 dense retrieval만 사용했습니다.

```bash
python app/eval/evaluate_retrieval.py \
  --questions eval/questions.jsonl \
  --collection trustdoc_os_paragraph \
  --top_k 10
```

```bash
python app/eval/evaluate_retrieval.py \
  --questions eval/questions.jsonl \
  --collection trustdoc_os_fixed \
  --top_k 10
```

| Collection | top_k | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| paragraph | 10 | 0.6500 | 0.9000 | 0.9500 | 1.0000 | 0.7875 |
| fixed-size | 10 | 0.7500 | 0.9000 | 0.9500 | 1.0000 | 0.8338 |

20개 질문 기준의 baseline retrieval에서는 fixed-size chunking이 paragraph chunking보다 Hit@1과 MRR에서 더 높은 결과를 보였습니다.

다만 두 방식 모두 Hit@10은 1.0000으로, 정답 근거가 top10 안에는 모두 포함되었습니다.  
따라서 현재 문제는 retrieval 자체의 실패라기보다, 정답 근거를 더 상위로 올리는 ranking 품질 문제에 가깝습니다.

---

## Reranking Experiment

기존 embedding search는 정답 근거를 top10 안에는 잘 포함했지만, 일부 질문에서는 정답 chunk가 top1/top3로 충분히 올라오지 못했습니다.

이를 개선하기 위해 `cross-encoder/ms-marco-MiniLM-L-6-v2` 기반 CrossEncoder reranker를 추가했습니다.

### Reranking Pipeline

```text
query
→ embedding search로 후보 top10 검색
→ CrossEncoder reranker로 query-chunk pair 재점수화
→ rerank_score 기준 재정렬
→ 최종 retrieval 평가
```

### Run Rerank Evaluation

```bash
PYTHONPATH=. python app/eval/evaluate_rerank.py \
  --questions eval/questions.jsonl \
  --collection trustdoc_os_paragraph \
  --initial_top_k 10
```

```bash
PYTHONPATH=. python app/eval/evaluate_rerank.py \
  --questions eval/questions.jsonl \
  --collection trustdoc_os_fixed \
  --initial_top_k 10
```

### Reranking Result

| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| paragraph baseline | 0.6500 | 0.9000 | 0.9500 | 1.0000 | 0.7875 |
| paragraph + rerank | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 0.8917 |
| fixed-size baseline | 0.7500 | 0.9000 | 0.9500 | 1.0000 | 0.8338 |
| fixed-size + rerank | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 0.8750 |

Reranking 적용 후 두 chunking 전략 모두 Hit@3와 Hit@5가 1.0000으로 개선되었습니다.

특히 paragraph chunking은 다음과 같이 성능이 크게 개선되었습니다.

```text
Hit@1 : 0.6500 → 0.8000
MRR   : 0.7875 → 0.8917
```

---

## Hybrid Retrieval Experiment

Dense retrieval은 의미 기반 검색에 강하지만, 정확한 키워드가 중요한 질문에서는 약할 수 있습니다.  
이를 보완하기 위해 BM25 keyword retrieval과 Dense retrieval을 결합한 Hybrid Retrieval을 실험했습니다.

Hybrid Retrieval에서는 BM25 점수와 dense distance를 직접 더하지 않고, 순위 기반 결합 방식인 RRF(Reciprocal Rank Fusion)를 사용했습니다.

### Hybrid Retrieval Pipeline

```text
query
→ Dense retrieval top-k 검색
→ BM25 retrieval top-k 검색
→ RRF로 두 결과의 순위 결합
→ hybrid_score 기준 재정렬
→ 최종 retrieval 평가
```

### Run Hybrid Evaluation

```bash
PYTHONPATH=. python app/eval/evaluate_hybrid.py \
  --questions eval/questions.jsonl \
  --chunks data/processed/chunks_paragraph_all.json \
  --collection trustdoc_os_paragraph \
  --top_k 10 \
  --dense_top_k 10 \
  --bm25_top_k 10
```

```bash
PYTHONPATH=. python app/eval/evaluate_hybrid.py \
  --questions eval/questions.jsonl \
  --chunks data/processed/chunks_fixed_all.json \
  --collection trustdoc_os_fixed \
  --top_k 10 \
  --dense_top_k 10 \
  --bm25_top_k 10
```

### Hybrid Retrieval Result

| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| paragraph baseline | 0.6500 | 0.9000 | 0.9500 | 1.0000 | 0.7875 |
| paragraph + hybrid | 0.7000 | 0.8500 | 0.9500 | 1.0000 | 0.7954 |
| fixed-size baseline | 0.7500 | 0.9000 | 0.9500 | 1.0000 | 0.8338 |
| fixed-size + hybrid | 0.7000 | 0.9000 | 0.9500 | 1.0000 | 0.8181 |

Hybrid Retrieval은 일부 키워드형 질문에서는 도움이 되었지만, 전체 성능을 안정적으로 개선하지는 못했습니다.

예를 들어 `first-fit`, `best-fit`, `contiguous allocation`처럼 정확한 용어가 중요한 질문에서는 BM25가 정답 chunk를 더 높은 순위로 끌어올렸습니다.

하지만 일반 개념형 질문에서는 BM25가 반복적으로 등장하는 키워드에 끌려 오히려 ranking을 흐리는 경우도 있었습니다.

따라서 Hybrid Retrieval은 단독 최종 ranking 전략으로는 불안정했습니다.

---

## Hybrid + Reranking Experiment

Hybrid Retrieval을 최종 ranking 전략으로 사용하는 대신, reranker 앞단의 candidate generation 단계로 활용하는 실험을 진행했습니다.

### Hybrid + Reranking Pipeline

```text
query
→ Dense retrieval top-k 검색
→ BM25 retrieval top-k 검색
→ RRF로 후보군 결합
→ CrossEncoder reranker로 query-chunk pair 재점수화
→ rerank_score 기준 최종 재정렬
→ retrieval 평가
```

### Run Hybrid + Rerank Evaluation

```bash
PYTHONPATH=. python app/eval/evaluate_hybrid_rerank.py \
  --questions eval/questions.jsonl \
  --chunks data/processed/chunks_paragraph_all.json \
  --collection trustdoc_os_paragraph \
  --hybrid_top_k 10 \
  --dense_top_k 10 \
  --bm25_top_k 10
```

```bash
PYTHONPATH=. python app/eval/evaluate_hybrid_rerank.py \
  --questions eval/questions.jsonl \
  --chunks data/processed/chunks_fixed_all.json \
  --collection trustdoc_os_fixed \
  --hybrid_top_k 10 \
  --dense_top_k 10 \
  --bm25_top_k 10
```
---

## FastAPI Serving

TrustDoc RAG는 CLI 실행뿐 아니라 FastAPI 기반 API serving도 지원합니다.

API는 retrieval debugging용 endpoint와 최종 RAG 답변 생성 endpoint를 분리했습니다.

```text
POST /rag/retrieve
→ retrieval 결과만 반환
→ LLM 호출 없음
→ dense / hybrid / rerank 결과 디버깅용

POST /rag/query
→ retrieval 수행
→ optional reranking
→ prompt 생성
→ Gemini 답변 생성
→ answer + retrieved_sources 반환
```

### Run API Server

```bash
PYTHONPATH=. python -m uvicorn app.api.main:app --reload
```

서버 실행 후 Swagger UI에서 API를 테스트할 수 있습니다.

```text
http://127.0.0.1:8000/docs
```

### Request Schema

```json
{
  "query": "What are first-fit and best-fit in contiguous allocation?",
  "collection": "trustdoc_os_paragraph",
  "chunks_path": "data/processed/chunks_paragraph_all.json",
  "retrieval_mode": "hybrid",
  "top_k": 5,
  "initial_top_k": 10,
  "dense_top_k": 10,
  "bm25_top_k": 10,
  "use_rerank": true
}
```

### Retrieval-only API

```bash
curl -X POST "http://127.0.0.1:8000/rag/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are first-fit and best-fit in contiguous allocation?",
    "collection": "trustdoc_os_paragraph",
    "chunks_path": "data/processed/chunks_paragraph_all.json",
    "retrieval_mode": "hybrid",
    "top_k": 5,
    "initial_top_k": 10,
    "dense_top_k": 10,
    "bm25_top_k": 10,
    "use_rerank": true
  }'
```

Example response:

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
      "rerank_score": 6.8276
    }
  ]
}
```

### RAG Query API

```bash
curl -X POST "http://127.0.0.1:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are first-fit and best-fit in contiguous allocation?",
    "collection": "trustdoc_os_paragraph",
    "chunks_path": "data/processed/chunks_paragraph_all.json",
    "retrieval_mode": "hybrid",
    "top_k": 5,
    "initial_top_k": 10,
    "dense_top_k": 10,
    "bm25_top_k": 10,
    "use_rerank": true
  }'
```

Example response:

```json
{
  "query": "What are first-fit and best-fit in contiguous allocation?",
  "collection": "trustdoc_os_paragraph",
  "retrieval_mode": "hybrid",
  "use_rerank": true,
  "answer": "In contiguous allocation, first-fit allocates the first hole that is large enough, while best-fit allocates the smallest hole that is large enough...",
  "retrieved_sources": [
    {
      "rank": 1,
      "source_file": "2026-OS-L9A-MainMemory.pdf",
      "page_number": 13,
      "chunk_id": "13_para_0",
      "dense_rank": 6,
      "bm25_rank": 1,
      "hybrid_score": 0.03154,
      "rerank_score": 6.8276
    }
  ]
}
```

### Supported Retrieval Modes

| retrieval_mode | Description |
|---|---|
| `dense` | Chroma vector DB 기반 semantic retrieval |
| `hybrid` | Dense retrieval + BM25 retrieval + RRF fusion |

### Reranking Option

| use_rerank | Description |
|---|---|
| `false` | retrieval 결과를 그대로 사용 |
| `true` | CrossEncoder reranker로 최종 순위 재정렬 |

현재 실험 기준 가장 좋은 설정은 다음과 같습니다.

```json
{
  "retrieval_mode": "hybrid",
  "use_rerank": true,
  "collection": "trustdoc_os_paragraph",
  "chunks_path": "data/processed/chunks_paragraph_all.json"
}
```

## Streamlit Demo UI

TrustDoc RAG includes a thin Streamlit demo UI that calls the existing FastAPI backend.

Backend:

```bash
PYTHONPATH=. python -m uvicorn app.api.main:app --reload
```

Frontend:

```bash
streamlit run app/ui/streamlit_app.py
```

`/rag/retrieve` is for retrieval debugging without an LLM call.

`/rag/query` is for final RAG answer generation.

Recommended default demo setting:

```text
retrieval_mode: hybrid
use_rerank: true
collection: trustdoc_os_paragraph
chunks_path: data/processed/chunks_paragraph_all.json
```

### Final Evaluation Result

| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| paragraph baseline | 0.6500 | 0.9000 | 0.9500 | 1.0000 | 0.7875 |
| paragraph + hybrid | 0.7000 | 0.8500 | 0.9500 | 1.0000 | 0.7954 |
| paragraph + rerank | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 0.8917 |
| paragraph + hybrid + rerank | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 0.9000 |
| fixed-size baseline | 0.7500 | 0.9000 | 0.9500 | 1.0000 | 0.8338 |
| fixed-size + hybrid | 0.7000 | 0.9000 | 0.9500 | 1.0000 | 0.8181 |
| fixed-size + rerank | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 0.8750 |
| fixed-size + hybrid + rerank | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 0.8750 |

최종적으로 현재 실험에서는 `paragraph + hybrid + rerank` 조합이 가장 높은 MRR을 보였습니다.

```text
Best method:
paragraph + hybrid + rerank

Hit@1 : 0.8000
Hit@3 : 1.0000
Hit@5 : 1.0000
Hit@10: 1.0000
MRR   : 0.9000
```

다만 `paragraph + rerank` 대비 MRR만 `0.8917 → 0.9000`으로 소폭 개선되었기 때문에, hybrid의 추가 효과는 제한적이었습니다.

---

## Key Findings

- 작은 평가셋만으로 chunking 전략의 우열을 판단하면 위험합니다.
- 초기 5개 질문 평가에서는 paragraph chunking이 더 좋아 보였지만, 20개 질문으로 확장하자 fixed-size chunking이 baseline 기준 Hit@1과 MRR에서 더 좋은 결과를 보였습니다.
- 두 baseline 방식 모두 Hit@10은 1.0000으로, 정답 근거를 top10 안에는 모두 포함했습니다.
- 따라서 현재 문제는 검색 실패보다는 ranking 품질 개선 문제에 가깝습니다.
- CrossEncoder reranking을 적용하자 Hit@3, Hit@5가 모두 1.0000으로 개선되었습니다.
- Hybrid Retrieval 단독은 일부 키워드형 질문에는 도움이 되었지만, 전체적으로 안정적인 성능 개선을 만들지는 못했습니다.
- BM25는 정확한 용어가 중요한 질문에서는 유리했지만, 반복 키워드가 많은 개념형 질문에서는 순위를 흐릴 수 있었습니다.
- Hybrid Retrieval을 최종 ranking 전략으로 쓰기보다는, reranker 앞단의 candidate generation 단계로 활용하는 편이 더 안정적이었습니다.
- 최종 실험에서는 paragraph + hybrid + rerank 조합이 가장 높은 MRR을 보였습니다.
- RAG에서는 chunking 전략을 embedding retrieval 성능만으로 판단하기보다, candidate generation, reranking까지 포함한 최종 retrieval pipeline 기준으로 평가해야 합니다.

---

## Limitations

- 평가 질문이 20개로 아직 작습니다.
- 운영체제 강의자료 PDF에 한정된 실험입니다.
- OCR, Table OCR, Layout-aware parsing은 아직 구현하지 않았습니다.
- Reranker는 성능을 개선하지만, embedding search보다 느립니다.
- Hybrid Retrieval의 RRF 파라미터와 top-k 설정을 충분히 튜닝하지 않았습니다.
- 현재 BM25 tokenization은 간단한 regex 기반이며, 한국어 형태소 분석은 적용하지 않았습니다.
- 현재는 CLI 중심이며 별도의 웹 UI는 없습니다.
- LLM 답변 품질에 대한 자동 평가는 아직 포함하지 않았습니다.

---

## Future Work

- 평가 질문 30~50개 이상으로 확장
- Query expansion 실험
- BM25 tokenization 개선
- RRF 파라미터 및 dense/BM25 candidate top-k 튜닝
- OCR / Table OCR / Layout-aware chunking 적용
- RAG 답변 품질 평가 추가
- FastAPI 기반 serving
- Streamlit 또는 React 기반 UI 추가
- 업로드한 PDF에 대해 즉시 질의응답 가능한 구조로 확장

---

## Status

현재 구현된 범위:

```text
PDF Parsing
Chunking
Embedding
Chroma Indexing
Semantic Retrieval
Gemini RAG Answer Generation
Source Citation
Retrieval Evaluation
Multiple Answer Page Evaluation
Hit@10 Metric
CrossEncoder Reranking
Rerank Evaluation
BM25 Retrieval
RRF-based Hybrid Retrieval
Hybrid Retrieval Evaluation
Hybrid + Rerank Evaluation
```

TrustDoc RAG는 현재 문서 기반 RAG MVP에서 한 단계 더 나아가, chunking 전략, dense retrieval, keyword retrieval, reranking, hybrid candidate generation을 정량 평가할 수 있는 실험형 RAG 프로젝트입니다.
