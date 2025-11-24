import re

import ollama
from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> list[tuple[str, int]]:
    """Extract text from PDF, returning list of (text, page_num) tuples."""
    text_chunks = []
    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text = re.sub(r'\s+', ' ', text)
                text_chunks.append((text, i + 1))
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text_chunks


def semantic_chunk(text: str, model_name: str = 'qwen3:4b-instruct-2507-q4_K_M') -> list[str]:
    """Chunk text using LLM semantic parsing."""
    BATCH_SIZE = 2000
    text_parts = [text[i:i+BATCH_SIZE] for i in range(0, len(text), BATCH_SIZE)]
    processed_text = ""
    
    for part in text_parts:
        try:
            response = ollama.chat(model=model_name, messages=[
                {
                    'role': 'system',
                    'content': "You are a semantic parser. Rewrite the following text exactly as is, but insert '¶' where the topic shifts significantly. Output ONLY the processed text."
                },
                {'role': 'user', 'content': part}
            ])
            processed_text += response['message']['content']
        except Exception as e:
            print(f"Error processing chunk: {e}")
            processed_text += part
    
    return [c.strip() for c in processed_text.split('¶') if c.strip()]
