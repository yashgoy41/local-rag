#!/usr/bin/env python3
"""
Main script to run the complete RAG pipeline:
1. Ingest documents
2. Generate embeddings
3. Run retrieval test
"""

import sys
import subprocess
from pathlib import Path

def run_script(script_name: str, description: str) -> bool:
    """
    Run a Python script and return success status.
    
    Args:
        script_name: Name of the script to run
        description: Human-readable description
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            cwd=Path(__file__).parent
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed: {e}")
        return False

def main():
    """Run the complete RAG pipeline."""
    print("="*60)
    print("LOCAL RAG PIPELINE")
    print("="*60)
    
    # Check if documents directory exists and has files
    docs_dir = Path("./documents")
    if not docs_dir.exists() or not any(docs_dir.iterdir()):
        print("\n⚠️  Warning: No documents found in ./documents/")
        print("Please add PDF or TXT files to the documents/ directory first.")
        return 1
    
    # Step 1: Ingestion
    if not run_script("ingest.py", "Step 1: Document Ingestion"):
        print("\n❌ Pipeline failed at ingestion step")
        return 1
    
    # Step 2: Embedding
    if not run_script("embed.py", "Step 2: Embedding Generation"):
        print("\n❌ Pipeline failed at embedding step")
        return 1
    
    # Step 3: Retrieval Test
    if not run_script("retriever.py", "Step 3: Retrieval Test"):
        print("\n❌ Pipeline failed at retrieval step")
        return 1
    
    # Success
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print("\nYour RAG system is ready to use!")
    print("\nNext steps:")
    print("  - Modify the query in retriever.py to test different searches")
    print("  - Import RAGRetriever in your own code")
    print("  - View data with Lance Data Viewer (see README)")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
