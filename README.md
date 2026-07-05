# TrustDoc RAG

PDF 문서를 기반으로 질문에 대한 답변과 근거 페이지를 함께 제공하는 문서 기반 RAG(Retrieval-Augmented Generation) MVP입니다.

운영체제 강의자료 PDF 21개를 대상으로 PDF 파싱, chunking, embedding, vector DB indexing, retrieval, LLM 기반 답변 생성, citation 출력, retrieval evaluation을 구현했습니다.

---

## Overview

TrustDoc RAG는 사용자의 질문에 대해 PDF 문서에서 관련 chunk를 검색하고, 검색된 근거를 LLM context로 제공해 답변을 생성하는 시스템입니다.

```text
PDF
→ Text Extraction
→ Chunking
→ Embedding
→ Chroma Vector DB
→ Retrieval
→ Prompt Construction
→ Gemini Answer Generation
→ Evidence Citation
→ Retrieval Evaluation
```

이 프로젝트의 목적은 단순 LLM API 호출이 아니라, 문서 기반 RAG 시스템의 핵심 흐름을 직접 구현하고 평가하는 것입니다.

---

## Features

- PDF 페이지별 텍스트 추출
- 파일명, 페이지 번호 metadata 유지
- Paragraph/Page-based chunking
- Fixed-size chunking
- Sentence-transformers 기반 embedding
- Chroma vector DB indexing
- Query 기반 top-k retrieval
- Gemini API 기반 RAG 답변 생성
- Source file / page number citation 출력
- Hit@1, Hit@3, Hit@5, MRR 기반 retrieval evaluation
- Paragraph chunking vs fixed-size chunking 비교

---

## Tech Stack

| Category | Stack |
|---|---|
| Language | Python |
| PDF Parsing | PyMuPDF |
| Embedding | sentence-transformers |
| Vector DB | Chroma |
| LLM | Gemini API |
| Env | python-dotenv |
| Evaluation | Hit@1, Hit@3, Hit@5, MRR |

Embedding model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

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
│   │
│   ├── retrieval/
│   │   ├── build_index.py
│   │   └── search.py
│   │
│   ├── rag/
│   │   ├── retriever.py
│   │   ├── build_prompt.py
│   │   ├── llm_client.py
│   │   ├── run_rag.py
│   │   └── test_gemini.py
│   │
│   └── eval/
│       └── evaluate_retrieval.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── chroma/
│
├── eval/
│   └── questions.jsonl
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## RAG Pipeline

### 1. Indexing

질문이 들어오기 전에 PDF 문서를 검색 가능한 형태로 변환합니다.

```text
PDF → Text Extraction → Chunking → Embedding → Chroma 저장
```

PDF에서 추출한 데이터는 파일명과 페이지 번호를 metadata로 유지합니다.

```json
{
  "source_file": "2026-OS-L8-Deadlocks.pdf",
  "page_number": 6,
  "chunk_id": "6_para_0",
  "chunking_strategy": "paragraph",
  "text": "Deadlock can arise if four conditions hold simultaneously..."
}
```

### 2. Retrieval & Generation

사용자의 질문이 들어오면 관련 chunk를 검색하고, 이를 LLM context로 제공해 답변을 생성합니다.

```text
User Query
→ Query Embedding
→ Top-k Retrieval
→ Context Construction
→ Gemini API
→ Answer with Evidence
```

---

## Chunking Strategies

이 프로젝트에서는 두 가지 chunking 전략을 비교했습니다.

| Strategy | Chunks | Description |
|---|---:|---|
| Paragraph/Page-based | 435 | 슬라이드 PDF의 페이지 단위 의미 보존에 유리 |
| Fixed-size | 572 | 일정 길이로 세밀하게 분할하지만 문맥 단절 가능 |

Fixed-size chunking 설정:

```text
chunk_size = 600
overlap = 100
```

실험 결과, 현재 운영체제 슬라이드 PDF에서는 paragraph/page 기반 chunking이 문맥 보존과 검색 품질 측면에서 더 안정적이었습니다.

---

## Installation

```bash
git clone <repository-url>
cd TrustDoc-RAG

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt`

```txt
pymupdf
python-dotenv
tqdm
chromadb
sentence-transformers
google-genai
```

---

## Environment Variables

프로젝트 루트에 `.env` 파일을 생성합니다.

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

`.env`, 원본 PDF, Chroma DB는 GitHub에 업로드하지 않습니다.

```gitignore
.env
data/raw/
data/chroma/
```

---

## How to Run

### 1. Build Chroma Index

Paragraph chunking index:

```bash
python app/retrieval/build_index.py \
  --chunks data/processed/chunks_paragraph_all.json \
  --collection trustdoc_os_paragraph
```

Fixed-size chunking index:

```bash
python app/retrieval/build_index.py \
  --chunks data/processed/chunks_fixed_all.json \
  --collection trustdoc_os_fixed
```

---

### 2. Test Retrieval

```bash
python app/retrieval/search.py \
  --query "What are the four necessary conditions for deadlock?" \
  --collection trustdoc_os_paragraph \
  --top_k 5
```

Example retrieval result:

```text
[1] 2026-OS-L8-Deadlocks.pdf p.17
[2] 2026-OS-L8-Deadlocks.pdf p.18
[3] 2026-OS-L8-Deadlocks.pdf p.3
[4] 2026-OS-L8-Deadlocks.pdf p.6
[5] 2026-OS-L8B-Deadlocks.pdf p.2
```

---

### 3. Run RAG

English query:

```bash
python app/rag/run_rag.py \
  --query "What are the four necessary conditions for deadlock?" \
  --collection trustdoc_os_paragraph \
  --top_k 5
```

Korean query:

```bash
python app/rag/run_rag.py \
  --query "deadlock의 necessary conditions는 무엇인가?" \
  --collection trustdoc_os_paragraph \
  --top_k 5
```

---

### 4. Run Retrieval Evaluation

Paragraph collection:

```bash
python app/eval/evaluate_retrieval.py \
  --questions eval/questions.jsonl \
  --collection trustdoc_os_paragraph \
  --top_k 5
```

Fixed-size collection:

```bash
python app/eval/evaluate_retrieval.py \
  --questions eval/questions.jsonl \
  --collection trustdoc_os_fixed \
  --top_k 5
```

---

## Example Output

Question:

```text
What are the four necessary conditions for deadlock?
```

Retrieved chunks:

```text
[1] distance=0.4991 | 2026-OS-L8-Deadlocks.pdf p.17 | 17_para_0
[2] distance=0.5199 | 2026-OS-L8-Deadlocks.pdf p.18 | 18_para_0
[3] distance=0.6438 | 2026-OS-L8-Deadlocks.pdf p.3  | 3_para_0
[4] distance=0.6475 | 2026-OS-L8-Deadlocks.pdf p.6  | 6_para_0
[5] distance=0.6583 | 2026-OS-L8B-Deadlocks.pdf p.2 | 2_para_0
```

Generated answer:

```text
For a deadlock to occur, four necessary conditions must hold simultaneously:

1. Mutual exclusion
2. Hold and wait
3. No preemption
4. Circular wait

Evidence:
- [Source 1, 2026-OS-L8-Deadlocks.pdf, page 17]
- [Source 4, 2026-OS-L8-Deadlocks.pdf, page 6]
```

---

## Evidence vs Retrieved Sources

이 프로젝트에서는 `Retrieved Sources`와 `Evidence`를 구분했습니다.

```text
Retrieved Sources = retrieval 단계에서 검색된 후보 문서
Evidence = LLM이 답변에 실제 사용했다고 명시한 근거
```

검색된 모든 문서를 정답 근거처럼 보여주면 혼동될 수 있기 때문에, 답변에 사용한 근거와 검색 후보를 분리했습니다.

---

## Retrieval Evaluation

평가는 20개 질문으로 진행했으며, 하나의 질문에 여러 페이지가 유효한 근거가 될 수 있어 복수 정답 페이지를 지원했습니다.

| Collection | top_k | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| paragraph | 10 | 0.6500 | 0.9000 | 0.9500 | 1.0000 | 0.7875 |
| fixed-size | 10 | 0.7500 | 0.9000 | 0.9500 | 1.0000 | 0.8338 |

20개 질문 기준으로는 fixed-size chunking이 paragraph/page-based chunking보다 Hit@1과 MRR에서 더 높은 결과를 보였습니다. 다만 두 방식 모두 Hit@10은 1.0000으로, 정답 근거가 top10 안에는 모두 포함되었습니다.

따라서 현재 문제는 retrieval 자체의 실패라기보다, 정답 근거를 더 상위로 올리는 ranking 품질 개선 문제로 볼 수 있습니다.

## Key Findings

- RAG 품질은 LLM 성능만이 아니라 PDF 파싱, chunking, embedding, retrieval, prompt 설계에 영향을 받습니다.
- Top1이 항상 가장 자세한 근거는 아니므로 top-k retrieval이 중요합니다.
- Fixed-size chunking은 구현이 단순하지만 문맥이 끊길 수 있습니다.
- 슬라이드 PDF에서는 paragraph/page 기반 chunking이 근거 표시와 문맥 보존에 유리했습니다.
- Retrieval 성능을 정량적으로 보기 위해 Hit@k, MRR 평가를 적용했습니다.
- 단일 정답 페이지보다 복수 정답 페이지를 허용하는 평가 방식이 실제 문서 검색에 더 자연스럽습니다.
- 평가 질문을 5개에서 20개로 확장하자 chunking 전략 비교 결과가 달라졌습니다. 이는 RAG 실험에서 작은 평가셋만으로 결론을 내리면 위험하며, 질문셋을 확장해 반복 평가하는 과정이 필요하다는 점을 보여줍니다.
- top_k=10에서는 두 chunking 전략 모두 Hit@10 1.0000을 보였으므로, 이후 개선 방향은 문서 파싱보다 reranking 또는 query expansion을 통한 상위 순위 개선에 가깝습니다.

---

## Limitations

현재 프로젝트는 MVP 단계이며 다음과 같은 한계가 있습니다.

- 평가 질문 수가 5개로 적음
- Deadlock 단원 중심으로 평가됨
- 이미지, 표, 다이어그램이 포함된 PDF 처리 부족
- OCR 미적용
- Table OCR / Layout-aware parsing 미적용
- Reranking 미적용
- Web UI 미구현

---

## Future Work

- 평가 질문을 15~30개 이상으로 확장
- Memory, Virtual Memory, File System, I/O 등 다른 단원 포함
- Reranker 도입
- Query expansion 적용
- 한국어 질문을 영어로 변환 후 검색하는 방식 실험
- OCR / Table OCR / Layout-aware parsing 적용
- FastAPI 기반 API 서버 구현
- Streamlit 또는 React 기반 UI 구현

---

## Status

현재 구현 완료:

- PDF parsing
- Chunking
- Chroma indexing
- Semantic retrieval
- Gemini 기반 RAG answer generation
- Citation output
- Retrieval evaluation
- Paragraph vs fixed-size comparison

다음 목표:

- 평가 질문 확장
- Reranking 실험
- OCR / Layout-aware parsing 실험
- 간단한 UI 또는 API 서버 구현