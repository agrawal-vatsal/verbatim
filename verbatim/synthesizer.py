from pathlib import Path
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class Synthesizer:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.2
            )  # Using a smarter model for synthesis

        prompt_path = Path("prompts/synthesis.txt")
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_path.read_text()),
                ("user", "Question: {question}")
            ]
        )

        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def generate_answer(self, question: str, chunks: list[dict[str, Any]]) -> str:
        # Format the DB chunks into a single string for the prompt
        context_text = "\n\n".join(
            [
                f"Source: {c['company']} | {c['fy']} {c['quarter']} | Page {c['page_number']}\nContent: {c['content']}"
                for c in chunks
            ]
        )

        return self.chain.invoke(
            {
                "question": question,
                "context": context_text
            }
        )