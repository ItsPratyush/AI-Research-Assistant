import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "research_papers"

class RAGPipeline:

    def __init__(self, k: int = 5): #self makes it so that variables are "instanced", and can be used outside the __init__ constructor later without confining those variables to the constructor.
        self.k = k
        self.client = chromadb.Client(Settings(
            persist_directory=CHROMA_DIR,
            anonymized_telemetry=False #opting out to being tracked by library devs.
        ))
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    

    def retrieve(self, query: str):
        query_emb = self.embed_model.encode([query]).tolist() #vectorization
        results = self.collection.query(
            query_embeddings = query_emb,
            n_results = self.k
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return list(zip(docs, metas))


    def build_prompt(self, query, context):
        context_text = ""
        for i, (doc, meta) in enumerate(context, 1):
            context_text += (
                f"[SOURCE {i} | {meta['source']} | page {meta['page']}]\n"
                f"{doc}\n\n"
            )
        return f"""
You are an AI Research Assistant.
Answer the question using ONLY the provided context.
If the answer is not present, say that it doesn't exist in the documents provided.
-----CONTEXT-----
{context_text}
-----END CONTEXT-----
Question: {query}
Rules:
    - Be consise and accurate
    - Cite sources like [SOURCE 1], [SOURCE 2]
"""
    

    def answer(self, query: str):
        context = self.retrieve(query)
        prompt = self.build_prompt(query, context)

        response = self.llm.chat.completions.create(
            model = "llama3-70b-8192",
            messages = [
                {"role": "user", "content": prompt}
            ],
            temperature = 0.2
        )

        return {
            "answer": response.choices[0].message.content,
            "context": content
        }