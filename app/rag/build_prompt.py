from typing import List, Dict, Any


def build_context(chunks: List[Dict[str, Any]], max_chars: int = 5000) -> str:
    """
    검색된 chunk들을 LLM에 넣을 context 문자열로 변환합니다.
    """
    context_parts = []
    total_chars = 0

    for chunk in chunks:
        meta = chunk["metadata"]

        header = (
            f"[Source {chunk['rank']}]\n"
            f"file: {meta['source_file']}\n"
            f"page: {meta['page_number']}\n"
            f"chunk_id: {meta['chunk_id']}\n"
        )

        body = chunk["text"].strip()
        part = f"{header}\n{body}\n"

        if total_chars + len(part) > max_chars:
            break

        context_parts.append(part)
        total_chars += len(part)

    return "\n---\n".join(context_parts)


def build_rag_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    context = build_context(chunks)

    prompt = f"""
You are a helpful assistant for answering questions based on lecture PDF documents.

Rules:
1. Use only the provided context.
2. If the answer is not supported by the context, say you cannot find enough evidence in the provided documents.
3. Do not make up facts that are not in the context.
4. Answer in the same language as the question.
5. When using evidence, cite it using this format: [Source N, file name, page number].
6. At the end, include an "Evidence" section listing only the sources you actually used.
7. Do not list sources that were retrieved but not used in the final answer.

[Context]
{context}

[Question]
{query}

[Answer]
""".strip()

    return prompt