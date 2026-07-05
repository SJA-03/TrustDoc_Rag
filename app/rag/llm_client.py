import os
from dotenv import load_dotenv
from google import genai


class GeminiClient:
    def __init__(self, model_name: str | None = None):
        load_dotenv()

        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = genai.Client()

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        if not response.text:
            return "답변을 생성하지 못했습니다."

        return response.text