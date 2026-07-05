from llm_client import GeminiClient


def main():
    client = GeminiClient()
    answer = client.generate("RAG를 한 문장으로 설명해줘.")
    print(answer)


if __name__ == "__main__":
    main()