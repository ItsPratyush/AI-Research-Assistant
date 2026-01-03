print("running query.py")

from rag_pipeline import RAGPipeline

def main():
    rag = RAGPipeline(k=5)

    print("--AI Research Assistant (Groq + RAG)")
    print("Type 'exit' to quit\n")

    while True:
        query = input(" Your question: ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        result = rag.answer(query)

        print("Answer:\n")
        print(result["answer"])

        print("Sources used:\n")
        for i, (_, meta) in enumerate(result["context"], 1):
            print(f"  SOURCE {i}: {meta['source']} (page {meta['page']})")
        print("\n" + "-" * 60)

if __name__ == "__main__":
    main()