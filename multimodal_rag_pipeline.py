import os
import uuid
import logging
import time
from pathlib import Path
from google import genai

logger = logging.getLogger("MultimodalPipeline")

class MultimodalRAGPipeline:
    def __init__(self):
        # The new SDK automatically picks up GEMINI_API_KEY from your .env file
        self.client = genai.Client()

    def process_file_with_gemini(self, file_path: Path) -> str:
        logger.info(f"Uploading {file_path.name} to Gemini for multimodal extraction...")
        
        # New syntax for uploading files
        uploaded_file = self.client.files.upload(file=str(file_path))
        
        # Wait for video/audio to finish processing on Google's servers
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = self.client.files.get(name=uploaded_file.name)
            
        if uploaded_file.state.name == "FAILED":
            raise Exception("Failed to process the media file.")
            
        prompt = (
            "You are a document and media extraction assistant. "
            "Analyze this file and provide a complete, highly detailed Markdown representation of its contents. "
            "If it's a video or audio, transcribe the speech and describe key visuals. "
            "If it's a document, extract the text cleanly."
        )
        
        # New syntax for generation
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[uploaded_file, prompt]
        )
        
        # Clean up cloud storage
        self.client.files.delete(name=uploaded_file.name) 
        return response.text

    def ingest(self, file_path: str) -> list[dict]:
        path = Path(file_path)
        markdown_text = self.process_file_with_gemini(path)
        
        words = markdown_text.split()
        chunk_size = 350
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i+chunk_size])
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": {"source": path.name}
            })
            
        return chunks