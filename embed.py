import ollama


def get_embeddings(texts: list[str], model_name: str = 'qwen3-embedding:4b') -> list[list[float]]:
    """Generate embeddings for a list of texts using Ollama."""
    embeddings = []
    for text in texts:
        response = ollama.embeddings(model=model_name, prompt=text)
        embeddings.append(response['embedding'])
    return embeddings
