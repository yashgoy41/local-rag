import os
import uuid
import ollama
import lancedb
from pypdf import PdfReader
from tqdm import tqdm
from lancedb.pydantic import LanceModel, Vector

# Schema definition
class Doc(LanceModel):
    id: str
    text: str
    source: str
    page: int
    vector: Vector(2560)

def extract_text_from_pdf(file_path):
    text_chunks = []
    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Normalize whitespace: replace multiple spaces with single space
                import re
                text = re.sub(r'\s+', ' ', text)
                text_chunks.append((text, i + 1))
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text_chunks

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [(f.read(), 0)]
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return []

def semantic_chunk(text):
    # Split into batches if too long to avoid context limits
    BATCH_SIZE = 2000
    
    # Split text into chunks of BATCH_SIZE
    text_parts = [text[i:i+BATCH_SIZE] for i in range(0, len(text), BATCH_SIZE)]
    
    processed_text = ""
    
    for part in text_parts:
        try:
            response = ollama.chat(model='llama3.1:8b', messages=[
                {
                    'role': 'system',
                    'content': "You are a semantic parser. Rewrite the following text exactly as is, but insert the special token '¶' at every point where the topic or semantic meaning shifts significantly. Do not change, summarize, or add any other words. Just the original text with '¶' inserted. Do not include any introductory or concluding remarks. Output ONLY the processed text."
                },
                {
                    'role': 'user',
                    'content': part
                }
            ])
            processed_text += response['message']['content']
        except Exception as e:
            print(f"Error processing chunk with Ollama: {e}")
            processed_text += part # Fallback to original text if fail
            
    # Split by the special token
    raw_chunks = processed_text.split('¶')
    return [c.strip() for c in raw_chunks if c.strip()]

def setup_database():
    db_path = "./data/lancedb"
    os.makedirs(db_path, exist_ok=True)
    db = lancedb.connect(db_path)
    
    # Create table if not exists
    try:
        table = db.create_table("docs", schema=Doc, exist_ok=True)
    except Exception as e:
        # Fallback/Handling if table exists with different schema or other issues
        # For this script, we assume we can open it if it exists
        table = db.open_table("docs")
        
    return table

def main():
    # Ensure documents directory exists
    if not os.path.exists("./documents"):
        os.makedirs("./documents")
        print("Created ./documents folder. Please add files there.")
        return

    files = [f for f in os.listdir("./documents") if f.lower().endswith(('.pdf', '.txt'))]
    
    if not files:
        print("No PDF or TXT files found in ./documents.")
        return

    table = setup_database()
    
    print(f"Found {len(files)} files. Starting ingestion...")
    
    for filename in tqdm(files, desc="Processing files"):
        file_path = os.path.join("./documents", filename)
        
        extracted_data = []
        if filename.lower().endswith('.pdf'):
            extracted_data = extract_text_from_pdf(file_path)
        elif filename.lower().endswith('.txt'):
            extracted_data = extract_text_from_txt(file_path)
            
        for text, page_num in extracted_data:
            if not text.strip():
                continue
                
            # Semantic chunking
            chunks = semantic_chunk(text)
            
            # Save chunks to debug file
            import json
            debug_file = "./data/chunks_debug.jsonl"
            with open(debug_file, "a", encoding="utf-8") as f:
                for chunk in chunks:
                    if chunk:
                        json.dump({"source": filename, "page": page_num, "chunk": chunk}, f, ensure_ascii=False)
                        f.write("\n")
            
            # Prepare data for insertion
            data_to_insert = []
            for chunk in chunks:
                if not chunk:
                    continue
                    
                # Dummy vector for now as requested
                dummy_vector = [0.0] * 2560 
                
                doc = Doc(
                    id=str(uuid.uuid4()),
                    text=chunk,
                    source=filename,
                    page=page_num,
                    vector=dummy_vector
                )
                data_to_insert.append(doc)
            
            if data_to_insert:
                table.add(data_to_insert)
    
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
