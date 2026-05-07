from verbatim.db import Database
from scripts.ingest_data import get_embeddings
import sys


def ask_verbatim(question: str):
    db = Database()

    print(f"🔍 Question: {question}")

    # 1. Turn the question into a vector
    # (We wrap it in a list because our get_embeddings function expects a list)
    question_vector = get_embeddings([question])[0]

    # 2. Search the database
    results = db.search_similar_chunks(question_vector, limit=3)

    # 3. Display the findings
    print("\n📚 Top Relevant Chunks Found:")
    print("-" * 50)
    for i, res in enumerate(results, 1):
        print(f"RESULT {i} | Distance: {res['distance']}")
        print(f"Source: {res['metadata']}")
        print(f"Content: {res['content'][:300]}...")
        print("-" * 50)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        ask_verbatim(user_query)
    else:
        # Default test query
        ask_verbatim("What are the key highlights of HDFC Bank's Q3 performance?")