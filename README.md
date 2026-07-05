# TrustDoc RAG

PDF 문서를 기반으로 질문에 대한 답변과 근거 페이지를 함께 제공하는 문서 기반 RAG(Retrieval-Augmented Generation) 시스템입니다.

운영체제 강의자료 PDF를 대상으로 PDF 파싱, chunking, embedding, Chroma indexing, dense retrieval, BM25 hybrid retrieval, CrossEncoder reranking, Gemini 기반 답변 생성, retrieval evaluation, FastAPI serving, Streamlit demo UI까지 구현했습니다.

---

## Demo

TrustDoc RAG는 Streamlit UI를 통해 질문 입력, retrieval mode 선택, reranking 옵션 변경, 답변 확인, retrieved source 분석을 한 화면에서 수행할 수 있습니다.

<img width="1512" height="857" alt="스크린샷 2026-07-05 오후 11 30 31" src="https://github.com/user-attachments/assets/2e89df56-fbaf-4701-9b9e-0aa0fab4105c" />

---

## Overview

TrustDoc RAG는 PDF 문서를 미리 검색 가능한 형태로 저장한 뒤, 사용자가 질문하면 관련 문서 조각을 검색하고 LLM에게 근거로 제공하여 답변을 생성하는 시스템입니다.

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

최종적으로 API serving과 데모 UI까지 구현하여 다음 흐름으로 사용할 수 있습니다.

```text
Streamlit UI
→ FastAPI API
→ Dense / Hybrid Retrieval
→ Optional Reranking
→ Gemini Answer Generation
→ Answer + Retrieved Sources
```

---

## Key Features

- PDF 문서 텍스트 추출
- page/paragraph 기반 chunking
- fixed-size chunking
- paper section-aware chunking
- SentenceTransformer 기반 embedding 생성
- Chroma Vector DB indexing
- Chroma 기반 dense retrieval
- BM25 기반 keyword retrieval
- RRF 기반 hybrid retrieval
- CrossEncoder 기반 reranking
- Gemini API 기반 RAG 답변 생성
- 답변 내 source citation 출력
- retrieval-only API 제공
- RAG query API 제공
- Streamlit demo UI 제공
- Hit@1, Hit@3, Hit@5, Hit@10, MRR 기반 retrieval evaluation
- 복수 정답 페이지 평가 지원

---

## Tech Stack

| Area | Stack |
|---|---|
| Language | Python |
| PDF Parsing | PyMuPDF |
| Embedding | sentence-transformers |
| Vector DB | Chroma |
| Keyword Retrieval | rank-bm25 |
| Rank Fusion | Reciprocal Rank Fusion |
| Reranker | CrossEncoder |
| LLM | Gemini API |
| API Server | FastAPI, Uvicorn |
| Demo UI | Streamlit |
| Evaluation | Hit@k, MRR |

---

## Project Structure

```text
TrustDoc-RAG/
├── app/
│   ├── api/
│   │   └── main.py
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
│   ├── ui/
│   │   └── streamlit_app.py
│   └── eval/
│       ├── evaluate_retrieval.py
│       ├── evaluate_rerank.py
│       ├── evaluate_hybrid.py
│       └── evaluate_hybrid_rerank.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── chroma/
├── docs/
│   └── images/
│       └── streamlit-demo.png
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

### 2. Retrieval

```text
사용자 질문
→ query embedding
→ Chroma에서 관련 chunk 검색
→ top-k candidate 반환
```

### 3. Reranking

```text
query
→ dense retrieval top-k 후보 검색
→ CrossEncoder가 query-chunk pair 재점수화
→ rerank_score 기준 재정렬
```

### 4. Hybrid Retrieval

```text
query
→ Dense retrieval top-k 검색
→ BM25 retrieval top-k 검색
→ RRF로 candidate 결합
→ hybrid_score 기준 재정렬
```

### 5. Hybrid + Reranking

```text
query
→ Dense retrieval 후보
→ BM25 retrieval 후보
→ RRF 기반 hybrid candidate 생성
→ CrossEncoder reranker로 최종 재정렬
```

### 6. Generation

```text
최종 top-k chunk
→ prompt 생성
→ Gemini 답변 생성
→ source citation 포함 답변 반환
```

---

## Chunking Strategies

### 1. Paragraph / Page-based Chunking

페이지 단위의 문맥을 비교적 잘 보존합니다.

강의자료 PDF처럼 한 페이지에 하나의 개념이 정리된 문서에서 유리할 수 있습니다.

### 2. Fixed-size Chunking

일정한 글자 수 기준으로 chunk를 나눕니다.

특정 키워드나 짧은 개념 검색에는 유리할 수 있지만, 문장이나 개념이 중간에 끊길 수 있습니다.

### 3. Section-aware Chunking

논문형 PDF를 위한 실험적 chunking 전략입니다.

텍스트에서 `Abstract`, `Introduction`, `Related Work`, `Methods`, `Experiments`, `Conclusion` 같은 section heading과 `3.1 Problem Formalization and Overview` 같은 numbered heading을 감지하고, 각 chunk에 `section_title` 메타데이터를 추가합니다.

레이아웃 모델을 사용하는 방식은 아니며, PDF에서 추출된 텍스트를 기반으로 section을 추적하는 단순한 text-based 전략입니다.

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

## Data Preparation

PDF 파일은 GitHub에 포함하지 않습니다.

로컬 환경에서 PDF 파일을 다음 경로에 넣습니다.

```text
data/raw/
```

예시:

```text
data/raw/2026-OS-L8-Deadlocks.pdf
data/raw/2026-OS-L9A-MainMemory.pdf
data/raw/2026-OS-L10A-VirtualMemory.pdf
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

Paragraph chunking:

```bash
python app/ingest/batch_ingest.py \
  --input_dir data/raw \
  --strategy paragraph \
  --output data/processed/chunks_paragraph_all.json
```

Fixed-size chunking:

```bash
python app/ingest/batch_ingest.py \
  --input_dir data/raw \
  --strategy fixed \
  --output data/processed/chunks_fixed_all.json
```

### 3. Build Chroma Index

Paragraph collection:

```bash
python app/retrieval/build_index.py \
  --chunks data/processed/chunks_paragraph_all.json \
  --collection trustdoc_os_paragraph
```

Fixed-size collection:

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

### 5. Run CLI RAG

```bash
python app/rag/run_rag.py \
  --query "What are the four necessary conditions for deadlock?" \
  --collection trustdoc_os_paragraph \
  --top_k 5
```

---

## FastAPI Serving

TrustDoc RAG는 FastAPI 기반 API serving을 지원합니다.

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
      "distance": 0.9602,
      "dense_rank": 6,
      "bm25_rank": 1,
      "hybrid_score": 0.03154,
      "rerank_score": 6.8276,
      "text_preview": "..."
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

---

## Streamlit Demo UI

FastAPI endpoint를 호출하는 Streamlit 기반 데모 UI를 제공합니다.

### Run Backend

```bash
PYTHONPATH=. python -m uvicorn app.api.main:app --reload
```

### Run Frontend

```bash
streamlit run app/ui/streamlit_app.py
```

실행 후 접속:

```text
http://localhost:8501
```

### UI Features

- 질문 입력
- 예시 질문 버튼
- document set 선택
  - Operating Systems
  - AI Papers
  - Custom
- API base URL 설정
- `RAG Answer` / `Retrieve Only` 모드 선택
- `dense` / `hybrid` retrieval mode 선택
- reranking 사용 여부 선택
- document set에 따른 collection / chunks path 자동 설정
- Custom document set의 collection / chunks path 직접 입력
- `top_k`, `initial_top_k`, `dense_top_k`, `bm25_top_k` 조정
- 답변 출력
- retrieved sources table 출력
- source별 expandable card 출력
- raw API response 확인

### Supported Document Sets

| Document Set | Collection | Chunks Path |
|---|---|---|
| Operating Systems | `trustdoc_os_paragraph` | `data/processed/chunks_paragraph_all.json` |
| AI Papers | `trustdoc_ai_papers_paragraph` | `data/processed/chunks_ai_papers_paragraph.json` |
| Custom | 사용자 입력 | 사용자 입력 |

Document set을 선택하면 Streamlit UI가 해당 collection과 chunks path를 자동으로 API payload에 사용합니다.

### Example Questions

Operating Systems:

```text
What are the four necessary conditions for deadlock?
What is demand paging?
What are first-fit and best-fit in contiguous allocation?
What is thrashing in virtual memory?
```

AI Papers:

```text
What is Retrieval-Augmented Generation?
How does Self-RAG decide when to retrieve?
What does RAGAS evaluate in a RAG pipeline?
How does Self-RAG differ from standard RAG?
```

### Recommended Demo Setting

```json
{
  "document_set": "Operating Systems",
  "endpoint_mode": "RAG Answer",
  "retrieval_mode": "hybrid",
  "use_rerank": true,
  "collection": "trustdoc_os_paragraph",
  "chunks_path": "data/processed/chunks_paragraph_all.json",
  "top_k": 5,
  "initial_top_k": 10,
  "dense_top_k": 10,
  "bm25_top_k": 10
}
```

Example question:

```text
What are first-fit and best-fit in contiguous allocation?
```

Expected result:

```text
Answer generated from 2026-OS-L9A-MainMemory.pdf page 13
rank 1 source = 2026-OS-L9A-MainMemory.pdf p.13
dense_rank = 6
bm25_rank = 1
rerank_score ≈ 6.8276
```

---

## Example RAG Output

Question:

```text
What are first-fit and best-fit in contiguous allocation?
```

Answer:

```text
In contiguous allocation, first-fit and best-fit are methods used to satisfy a request of size n from a list of free holes.

First-fit allocates the first hole that is large enough.
Best-fit allocates the smallest hole that is large enough.

Evidence
- [Source 1, 2026-OS-L9A-MainMemory.pdf, page 13]
```

Retrieved source:

```text
rank: 1
source_file: 2026-OS-L9A-MainMemory.pdf
page_number: 13
chunk_id: 13_para_0
dense_rank: 6
bm25_rank: 1
rerank_score: 6.8276
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

### Evaluation Reporting & Error Analysis

초기 evaluation script는 Hit@k, MRR, detailed results를 terminal에 출력하는 방식이었습니다. 이제 `evaluate_hybrid_rerank.py`는 evaluation 결과를 JSON과 Markdown으로 저장할 수 있습니다. 이를 통해 실험 재현성과 weak-case 분석이 쉬워집니다.

JSON report는 다음 top-level 구조를 사용합니다.

```text
run_metadata
summary
detailed_results
weak_cases
```

Weak case는 `answer_rank`가 `None`이거나 `answer_rank > 1`인 질문입니다. Weak case를 보면 retrieval ranking이 불안정한 query를 빠르게 찾을 수 있습니다. Report에는 top1 source, page, `section_title`이 있는 경우의 section title, 그리고 retrieved text preview가 포함됩니다.

```bash
PYTHONPATH=. python app/eval/evaluate_hybrid_rerank.py \
  --questions eval/questions_ai_papers.jsonl \
  --chunks data/processed/chunks_ai_papers_section.json \
  --collection trustdoc_ai_papers_section \
  --hybrid_top_k 10 \
  --dense_top_k 10 \
  --bm25_top_k 10 \
  --output eval/results/ai_papers_section_hybrid_rerank.json \
  --markdown_output eval/results/ai_papers_section_hybrid_rerank.md
```

`eval/results/`는 Git에서 제외됩니다. 생성된 report는 로컬 실험 artifact이며, source document의 text preview를 포함할 수 있기 때문에 GitHub에 commit하지 않습니다.

예를 들어 section-aware + hybrid + rerank 실험에서는 `ai_q1`, `ai_q6` 같은 weak case가 자동으로 식별되었습니다. 이를 통해 top1 retrieval이 왜 잘못되었거나 부분적으로만 맞았는지 더 쉽게 확인할 수 있었습니다.

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

20개 질문 기준 baseline retrieval에서는 fixed-size chunking이 paragraph chunking보다 Hit@1과 MRR에서 더 높은 결과를 보였습니다.

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

## AI Papers Evaluation

The AI Papers document set contains three RAG-related papers:

- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- SELF-RAG
- Ragas

This document set is used to test whether the RAG pipeline works beyond the Operating Systems lecture slides. The AI Papers document set was chunked into 260 paragraph chunks.

```text
Chroma collection: trustdoc_ai_papers_paragraph
Evaluation file: eval/questions_ai_papers.jsonl
```

### Run Dense Baseline

```bash
PYTHONPATH=. python app/eval/evaluate_retrieval.py \
  --questions eval/questions_ai_papers.jsonl \
  --collection trustdoc_ai_papers_paragraph \
  --top_k 10
```

### Run Dense + Rerank

```bash
PYTHONPATH=. python app/eval/evaluate_rerank.py \
  --questions eval/questions_ai_papers.jsonl \
  --collection trustdoc_ai_papers_paragraph \
  --initial_top_k 10
```

### Run Hybrid

```bash
PYTHONPATH=. python app/eval/evaluate_hybrid.py \
  --questions eval/questions_ai_papers.jsonl \
  --chunks data/processed/chunks_ai_papers_paragraph.json \
  --collection trustdoc_ai_papers_paragraph \
  --top_k 10 \
  --dense_top_k 10 \
  --bm25_top_k 10
```

### Run Hybrid + Rerank

```bash
PYTHONPATH=. python app/eval/evaluate_hybrid_rerank.py \
  --questions eval/questions_ai_papers.jsonl \
  --chunks data/processed/chunks_ai_papers_paragraph.json \
  --collection trustdoc_ai_papers_paragraph \
  --hybrid_top_k 10 \
  --dense_top_k 10 \
  --bm25_top_k 10
```

### AI Papers Result

| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| dense baseline | 0.6667 | 0.9167 | 1.0000 | 1.0000 | 0.7986 |
| dense + rerank | 0.7500 | 0.9167 | 1.0000 | 1.0000 | 0.8403 |
| hybrid | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 0.8333 |
| hybrid + rerank | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 0.8611 |

Dense retrieval already found correct evidence within top5/top10 for all questions. However, Hit@1 was only 0.6667, so top-rank quality was limited. CrossEncoder reranking improved Hit@1 and MRR.

Hybrid retrieval improved Hit@3 to 1.0000, showing BM25 helps stabilize keyword-heavy paper retrieval. Hybrid + rerank achieved the best MRR of 0.8611. This suggests that for paper-style documents, combining keyword-based candidate generation with reranking is more reliable than dense retrieval alone.

Compared with the OS document set, the AI Papers document set has a different structure and style. OS slides and AI papers are different document types, but both document sets showed that correct evidence usually appears in top-k candidates. The main challenge is ranking the best evidence at the top. Reranking and hybrid candidate generation help improve this ranking quality.

### Section-aware Chunking Experiment

Section-aware chunking is an experimental text-based chunking strategy for paper-style PDFs. It detects section headings such as `Abstract`, `Introduction`, `Related Work`, `Evaluation Strategies`, `Experiments`, `Conclusion`, `References`, and numbered headings. It adds `section_title` metadata to each chunk.

The final cleaned section-aware AI Papers chunk file had 269 chunks.

```text
Chroma collection: trustdoc_ai_papers_section
```

The first version of heading detection incorrectly treated numbered list items and metric/table rows as section titles. The detector was refined to avoid single-number list items, decimal metric rows, and table/figure captions.

Create section-aware chunks:

```bash
python app/ingest/batch_ingest.py \
  --input_dir data/raw/ai_papers \
  --strategy section \
  --output data/processed/chunks_ai_papers_section.json
```

Build section-aware Chroma index:

```bash
python app/retrieval/build_index.py \
  --chunks data/processed/chunks_ai_papers_section.json \
  --collection trustdoc_ai_papers_section
```

Dense baseline evaluation:

```bash
PYTHONPATH=. python app/eval/evaluate_retrieval.py \
  --questions eval/questions_ai_papers.jsonl \
  --collection trustdoc_ai_papers_section \
  --top_k 10
```

Hybrid + rerank evaluation:

```bash
PYTHONPATH=. python app/eval/evaluate_hybrid_rerank.py \
  --questions eval/questions_ai_papers.jsonl \
  --chunks data/processed/chunks_ai_papers_section.json \
  --collection trustdoc_ai_papers_section \
  --hybrid_top_k 10 \
  --dense_top_k 10 \
  --bm25_top_k 10
```

| AI Papers Chunking | Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---:|---:|---:|---:|---:|
| paragraph | hybrid + rerank | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 0.8611 |
| section-aware | dense baseline | 0.5833 | 0.7500 | 0.8333 | 1.0000 | 0.6948 |
| section-aware | hybrid + rerank | 0.8333 | 0.9167 | 0.9167 | 1.0000 | 0.8889 |

Section-aware dense retrieval performed worse than paragraph dense retrieval. This suggests that simply adding section-aware chunking does not automatically improve semantic retrieval.

However, section-aware + hybrid + rerank achieved the highest Hit@1 and MRR on AI Papers. The improvement suggests that section-aware chunks can be useful when combined with keyword-based candidate generation and CrossEncoder reranking. However, Hit@3 and Hit@5 decreased because one broad definition question had the correct source at rank 6. Therefore, section-aware chunking should be considered experimental rather than the default.

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
- Streamlit UI는 로컬 데모용이며, production deployment는 아직 고려하지 않았습니다.
- LLM 답변 품질에 대한 자동 평가는 아직 포함하지 않았습니다.

---

## Future Work

- 평가 질문 30~50개 이상으로 확장
- Query expansion 실험
- BM25 tokenization 개선
- RRF 파라미터 및 dense/BM25 candidate top-k 튜닝
- OCR / Table OCR / Layout-aware chunking 적용
- RAG 답변 품질 평가 추가
- 파일 업로드 기반 ingestion API 추가
- FastAPI deployment 구조 개선
- Streamlit UI 개선 및 데모 시나리오 추가

---

## Current Status

현재 구현된 범위:

```text
PDF Parsing
Chunking
Embedding
Chroma Indexing
Dense Retrieval
BM25 Hybrid Retrieval
RRF Fusion
CrossEncoder Reranking
Retrieval Evaluation
Hybrid + Rerank Evaluation
Multi-document-set Evaluation
AI Papers Evaluation Set
Section-aware Chunking
Section-aware Chunking Evaluation
Evaluation Report Export
Weak-case Analysis
Gemini RAG Answer Generation
Source Citation
FastAPI Serving
Retrieval-only API
RAG Query API
Streamlit Demo UI
Swagger UI
```

TrustDoc RAG는 현재 문서 기반 RAG MVP에서 한 단계 더 나아가, chunking 전략, dense retrieval, keyword retrieval, reranking, hybrid candidate generation을 정량 평가하고 API/UI로 시연할 수 있는 실험형 RAG 프로젝트입니다.
