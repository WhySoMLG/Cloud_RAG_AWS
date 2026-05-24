import os
import uuid
import logging
from pinecone import Pinecone, ServerlessSpec
from google import genai
from multimodal_rag_pipeline import MultimodalRAGPipeline

logger = logging.getLogger("RAGConnector")

class PineconeStore:
    def __init__(self, index_name="multimodal-rag-v2", dimension=3072):
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required.")
        
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pc.Index(self.index_name)

    def upsert(self, chunk: dict, embedding: list[float], namespace: str):
        metadata = {
            "text": chunk.get("text", ""),
            "source": chunk.get("metadata", {}).get("source", "unknown"),
        }
        self.index.upsert(
            vectors=[{
                "id": chunk.get("id", str(uuid.uuid4())),
                "values": embedding,
                "metadata": metadata
            }],
            namespace=namespace
        )

    def query(self, query_vector: list[float], namespace: str, top_k=5):
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        return [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in results.matches]

class RAGConnector:
    def __init__(self):
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        
        self.client = genai.Client()
        self.vector_db = PineconeStore(index_name="multimodal-rag-v2", dimension=3072)
        self.pipeline = MultimodalRAGPipeline()
        
        self.embed_model = 'gemini-embedding-2' 
        self.llm_model = 'gemini-2.5-flash'

    def index(self, file_path: str, session_id: str):
        chunks = self.pipeline.ingest(file_path)
        for chunk in chunks:
            result = self.client.models.embed_content(
                model=self.embed_model, 
                contents=chunk["text"]
            )
            vector = result.embeddings[0].values
            self.vector_db.upsert(chunk, vector, namespace=session_id)

    def query(self, question: str, session_id: str, top_k=5):
        result = self.client.models.embed_content(
            model=self.embed_model, 
            contents=question
        )
        query_vector = result.embeddings[0].values
        
        candidates = self.vector_db.query(query_vector, namespace=session_id, top_k=top_k)
        
        context_texts = [f"Source: {c['metadata'].get('source')}\nContent: {c['metadata'].get('text')}" for c in candidates]
        context_blob = "\n\n".join(context_texts)
        
        prompt = f"Use the following context to answer the user's question.\n\nContext:\n{context_blob}\n\nQuestion: {question}"
        
        response = self.client.models.generate_content(
            model=self.llm_model,
            contents=prompt
        )
        
        return {
            "question": question,
            "answer": response.text,
            "sources": candidates
        }