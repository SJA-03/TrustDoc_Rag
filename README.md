# TrustDoc RAG

PDF 문서를 기반으로 질문에 대한 답변과 근거 페이지를 함께 제공하는 문서 기반 RAG(Retrieval-Augmented Generation) MVP입니다.

운영체제 강의자료 PDF 21개를 대상으로 PDF 파싱, chunking, embedding, vector DB indexing, retrieval, LLM 기반 답변 생성, citation 출력, retrieval evaluation, reranking evaluation을 구현했습니다.

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
│   │   ├── build_prompt.py
│   │   ├── llm_client.py
│   │   ├── run_rag.py
│   │   └── test_gemini.py
│   └── eval/
│       ├── evaluate_retrieval.py
│       └── evaluate_rerank.py
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

### Baseline Retrieval Result

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

최종적으로 현재 실험에서는 `paragraph + rerank` 조합이 가장 높은 성능을 보였습니다.

---

## Key Findings

- 작은 평가셋만으로 chunking 전략의 우열을 판단하면 위험합니다.
- 초기 5개 질문 평가에서는 paragraph chunking이 더 좋아 보였지만, 20개 질문으로 확장하자 fixed-size chunking이 baseline 기준 Hit@1과 MRR에서 더 좋은 결과를 보였습니다.
- 두 baseline 방식 모두 Hit@10은 1.0000으로, 정답 근거를 top10 안에는 모두 포함했습니다.
- 따라서 현재 문제는 검색 실패보다는 ranking 품질 개선 문제에 가깝습니다.
- CrossEncoder reranking을 적용하자 Hit@3, Hit@5가 모두 1.0000으로 개선되었습니다.
- 최종 실험에서는 paragraph + rerank 조합이 가장 좋은 성능을 보였습니다.
- RAG에서는 chunking 전략을 embedding retrieval 성능만으로 판단하기보다, reranker까지 포함한 최종 retrieval pipeline 기준으로 평가해야 합니다.

---

## Limitations

- 평가 질문이 20개로 아직 작습니다.
- 운영체제 강의자료 PDF에 한정된 실험입니다.
- OCR, Table OCR, Layout-aware parsing은 아직 구현하지 않았습니다.
- Reranker는 성능을 개선하지만, embedding search보다 느립니다.
- 현재는 CLI 중심이며 별도의 웹 UI는 없습니다.
- LLM 답변 품질에 대한 자동 평가는 아직 포함하지 않았습니다.

---

## Future Work

- 평가 질문 30~50개 이상으로 확장
- Query expansion 실험
- Hybrid retrieval BM25 + dense retrieval 적용
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
```

TrustDoc RAG는 현재 문서 기반 RAG MVP에서 한 단계 더 나아가, chunking 전략과 reranking 전략을 정량 평가할 수 있는 실험형 RAG 프로젝트입니다.