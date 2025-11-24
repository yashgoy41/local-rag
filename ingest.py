import os
import uuid
import time
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
    total_prompt_tokens = 0
    total_eval_tokens = 0
    total_prompt_time = 0
    total_eval_time = 0
    
    for part in text_parts:
        try:
            print(f"    Processing batch with LLM...", end='', flush=True)
            response = ollama.chat(model='qwen3:4b-instruct-2507-q4_K_M', messages=[
                {
                    'role': 'system',
                    'content': "You are a semantic parser. Rewrite the following text exactly as is, but insert the special token '¶' at every point where the topic or semantic meaning shifts significantly. Do not change, summarize, or add any other words. Just the original text with '¶' inserted. Do not include any introductory or concluding remarks. Output ONLY the processed text."
                },
                {
                    'role': 'user',
                    'content': part
                }
            ])
            
            # Extract Ollama's native performance metrics
            prompt_eval_count = response.get('prompt_eval_count', 0)
            prompt_eval_duration = response.get('prompt_eval_duration', 0) / 1e9  # Convert to seconds
            eval_count = response.get('eval_count', 0)
            eval_duration = response.get('eval_duration', 0) / 1e9  # Convert to seconds
            
            total_prompt_tokens += prompt_eval_count
            total_eval_tokens += eval_count
            total_prompt_time += prompt_eval_duration
            total_eval_time += eval_duration
            
            # Calculate tok/s for this batch
            prompt_tps = prompt_eval_count / prompt_eval_duration if prompt_eval_duration > 0 else 0
            eval_tps = eval_count / eval_duration if eval_duration > 0 else 0
            
            print(f" done ({eval_duration:.1f}s, {eval_count} tokens, {eval_tps:.1f} tok/s)")
            
            processed_text += response['message']['content']
        except Exception as e:
            print(f"\n    Error processing chunk with Ollama: {e}")
            processed_text += part # Fallback to original text if fail
    
    if total_eval_time > 0:
        avg_prompt_tps = total_prompt_tokens / total_prompt_time if total_prompt_time > 0 else 0
        avg_eval_tps = total_eval_tokens / total_eval_time if total_eval_time > 0 else 0
        print(f"  LLM Performance:")
        print(f"    Prompt: {total_prompt_tokens} tokens in {total_prompt_time:.2f}s ({avg_prompt_tps:.1f} tok/s)")
        print(f"    Generation: {total_eval_tokens} tokens in {total_eval_time:.2f}s ({avg_eval_tps:.1f} tok/s)")
            
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
    total_start = time.time()
    total_chunks = 0
    
    for file_idx, filename in enumerate(files, 1):
        file_start = time.time()
        print(f"\n{'='*60}")
        print(f"[{file_idx}/{len(files)}] Processing: {filename}")
        print(f"{'='*60}")
        
        file_path = os.path.join("./documents", filename)
        
        extracted_data = []
        if filename.lower().endswith('.pdf'):
            print("  Extracting text from PDF...")
            extract_start = time.time()
            extracted_data = extract_text_from_pdf(file_path)
            print(f"  Extracted {len(extracted_data)} pages in {time.time() - extract_start:.2f}s")
        elif filename.lower().endswith('.txt'):
            print("  Reading text file...")
            extract_start = time.time()
            extracted_data = extract_text_from_txt(file_path)
            print(f"  Read file in {time.time() - extract_start:.2f}s")
        
        file_chunks = 0
        for page_idx, (text, page_num) in enumerate(extracted_data, 1):
            if not text.strip():
                continue
            
            print(f"\n  Page {page_idx}/{len(extracted_data)}:")
            print(f"    Text length: {len(text)} characters")
                
            # Semantic chunking
            chunk_start = time.time()
            chunks = semantic_chunk(text)
            print(f"    Generated {len(chunks)} chunks")
            
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
                file_chunks += len(data_to_insert)
                total_chunks += len(data_to_insert)
        
        file_time = time.time() - file_start
        print(f"\n  ✓ File complete: {file_chunks} chunks in {file_time:.2f}s ({file_time/file_chunks:.2f}s/chunk)")
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total files: {len(files)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average: {total_time/total_chunks:.2f}s/chunk")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
