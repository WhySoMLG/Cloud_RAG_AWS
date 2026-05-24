import os
import uuid
import logging
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from multimodal_rag_pipeline import MultimodalRAGPipeline

logger = logging.getLogger("RAGConnector")

class PineconeStore:
    # Notice the new dimension (1536) and new index name
    def __init__(self, index_name="github-rag-v1", dimension=1536): 
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
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN environment variable is missing.")
        
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=token,
        )
        self.vector_db = PineconeStore(index_name="github-rag-v1", dimension=1536)
        self.pipeline = MultimodalRAGPipeline()
        
        # Top-tier Microsoft/OpenAI models
        self.embed_model = 'text-embedding-3-small' 
        self.llm_model = 'gpt-4o-mini'

    def index(self, file_path: str, session_id: str, mime_type: str = None):
        chunks = self.pipeline.ingest(file_path, mime_type)
        if not chunks:
            return
            
        chunk_texts = [chunk["text"] for chunk in chunks]

        # Send ONE batch request to embed all chunks instantly
        response = self.client.embeddings.create(
            input=chunk_texts,
            model=self.embed_model
        )
        
        for chunk, embedding_data in zip(chunks, response.data):
            vector = embedding_data.embedding
            self.vector_db.upsert(chunk, vector, namespace=session_id)

    def query(self, question: str, session_id: str, top_k=5):
        # 1. Embed the question
        embed_response = self.client.embeddings.create(
            input=question,
            model=self.embed_model
        )
        query_vector = embed_response.data[0].embedding
        
        # 2. Search Pinecone
        candidates = self.vector_db.query(query_vector, namespace=session_id, top_k=top_k)
        
        context_texts = [f"Source: {c['metadata'].get('source')}\nContent: {c['metadata'].get('text')}" for c in candidates]
        context_blob = "\n\n".join(context_texts)
        
        prompt = f"Use the following context to answer the user's question.\n\nContext:\n{context_blob}\n\nQuestion: {question}"
        
        # 3. Generate Answer
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return {
            "question": question,
            "answer": response.choices[0].message.content,
            "sources": candidates
        }