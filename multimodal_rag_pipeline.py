import os
import uuid
import logging
import base64
from pathlib import Path
from openai import OpenAI
import docx
from pypdf import PdfReader

logger = logging.getLogger("MultimodalPipeline")

class MultimodalRAGPipeline:
    def __init__(self):
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN environment variable is required.")
        
        # Connect the OpenAI SDK to GitHub's free model inference endpoint
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=token,
        )

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def extract_text(self, file_path: Path, mime_type: str = None) -> str:
        ext = file_path.suffix.lower()
        
        # 1. Handle Word Documents (Zero RAM impact)
        if ext == '.docx':
            logger.info(f"Extracting DOCX: {file_path.name}")
            doc = docx.Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        
        # 2. Handle PDFs (Zero RAM impact)
        elif ext == '.pdf':
            logger.info(f"Extracting PDF: {file_path.name}")
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
            
        # 3. Handle Text/Markdown
        elif ext in ['.txt', '.md', '.csv']:
            logger.info(f"Extracting Text: {file_path.name}")
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        # 4. Handle Images (Pass to GPT-4o-mini Vision)
        elif ext in ['.png', '.jpg', '.jpeg', '.webp']:
            logger.info(f"Describing Image: {file_path.name}")
            base64_image = self.encode_image(file_path)
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in high detail. Extract any text present in the image."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Unsupported file format: {ext}. Please use PDF, DOCX, TXT, or Images.")

    def ingest(self, file_path: str, mime_type: str = None) -> list[dict]:
        path = Path(file_path)
        extracted_text = self.extract_text(path, mime_type)
        
        if not extracted_text.strip():
            return []
        
        words = extracted_text.split()
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