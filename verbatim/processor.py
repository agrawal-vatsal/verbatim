from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

if TYPE_CHECKING:
    from verbatim.db import Database


class QueryProcessor:
    def __init__(self, db: "Database") -> None:
        self.db = db
        # Initialize the LangChain LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Load prompt from file
        prompt_path = Path("prompts/extraction.txt")
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_path.read_text()),
                ("user", "{user_input}")
            ]
        )

        # Define the chain
        self.chain = self.prompt_template | self.llm | JsonOutputParser()

    def process_query(self, user_input: str) -> dict[str, Any]:
        valid_companies = ", ".join(self.db.get_unique_companies())

        return cast(dict[str, Any], self.chain.invoke(
            {
                "valid_companies": valid_companies,
                "user_input": user_input
            }
        ))